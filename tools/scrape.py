#!/usr/bin/env python3

import os
import re
import copy
import logging
import tempfile
import subprocess

from common.headers import make_headers
from common.io import (lang, read_tree, read_map, read_list,
                       write_map, write_list)
from common.parser import ArgumentParser
from common.map import Map, translate


regex_perl_sub = re.compile('\nsub (\w+)\s+{([^}]*)}')
regex_subst = re.compile('\ssubst\("(\w+)", "(\w+)", "(\w+)"\);')

def _find_subst(txt, name):
    for sub in regex_perl_sub.findall(txt):
        if sub[0] == name:
            return regex_subst.findall(sub[1])
    return []


def update_maps(args, id_maps, triplets, known_ids):
    if args.verbose:
        print('Update translation maps:')
    count = 0
    for hop, hip, cuda in triplets:
        # cuda -> hip translation
        _hop = hop
        if cuda in id_maps['source']['cuda']:
            _hop = id_maps['source']['cuda'][cuda]
        elif cuda in known_ids and not translate.is_default_cuda(_hop, cuda):
            id_maps['source']['cuda'][cuda] = _hop
            count += 1
            if args.verbose:
                print('  New mapping: {} -> {}'.format(cuda, _hop))
        if hip in known_ids and _hop not in id_maps['target']['hip']:
            if not translate.is_default_hip(_hop, hip):
                id_maps['target']['hip'][_hop] = hip
                count += 1
                if args.verbose:
                    print('  New mapping: {} -> {}'.format(_hop, hip))
        # hip -> cuda translation
        _hop = hop
        if hip in id_maps['source']['hip']:
            _hop = id_maps['source']['hip'][hip]
        elif hip in known_ids and not translate.is_default_hip(_hop, hip):
            id_maps['source']['hip'][hip] = _hop
            count += 1
            if args.verbose:
                print('  New mapping: {} -> {}'.format(hip, _hop))
        if cuda in known_ids and _hop not in id_maps['target']['cuda']:
            if not translate.is_default_cuda(_hop, cuda):
                id_maps['target']['cuda'][_hop] = cuda
                count += 1
                if args.verbose:
                    print('  New mapping: {} -> {}'.format(_hop, cuda))
    if args.verbose:
        print('  New mapping chains:', count)
    return count


def scrape_hipify(args, path, known_ids):
    if args.verbose:
        print('Scrape hipify: {}'.format(path))
    txt = open(path).read()
    subs = []
    subs.extend(_find_subst(txt, 'simpleSubstitutions'))
    if args.include_experimental:
        subs.extend(_find_subst(txt, 'experimentalSubstitutions'))

    if args.exclude:
        # exclude IDs with prefix
        regex_exclude = re.compile('^({})'.format('|'.join(args.exclude)))
        exclude = lambda x: regex_exclude.match(x)
    else:
        exclude = lambda x: False

    triplets = []
    for cuda, hip, group in subs:
        if group in args.exclude_group:
            continue
        elif exclude(cuda) or exclude(hip):
            continue
        elif not args.include_unknown and \
                (cuda not in known_ids or hip not in known_ids):
            continue
        hop = translate.to_hop(hip)
        triplets.append((hop, hip, cuda))
    if args.verbose:
        print('  Substitutions found: {}'.format(len(triplets)))
        print('')
    return triplets


def _regex_lang(path):
    _lang = lang(path)
    if not _lang:
        raise ValueError('Unable to guess header language: {}'.format(path))
    prefix = [_lang.lower(), _lang.upper(), _lang.capitalize()]
    if _lang == 'CUDA':
        prefix.extend(['cu', 'CU', 'Cu'])
    return re.compile('^(.*?)({})'.format('|'.join(prefix)))


def _filename(path):
    dirname, filename = os.path.split(path)
    if lang(path) == 'HIP':
        subname = os.path.basename(dirname)
        filename = os.path.join(subname, filename)
    return filename


def _root(path):
    dirname, filename = os.path.split(path)
    if lang(path) == 'HIP':
        dirname = os.path.dirname(dirname)
    return dirname


def _remove_id(name, id_list):
    for filename in id_list:
        if name in id_list[filename]:
            id_list[filename].remove(name)


def _ctags(path):
    with tempfile.NamedTemporaryFile(prefix='tmp-hop-scrape-',
                                     suffix='.h') as fp:
        define = ''
        if lang(path) == 'HIP':
            define = '-D__HIP_PLATFORM_AMD__'
        # preprocess to get also included files
        cpp = 'cpp -I{} {} {} > {}'.format(_root(path), define, path, fp.name)
        # get only identifiers that are visible externally
        ctags = 'ctags -x --c-kinds=defgtuvp --file-scope=no {}'.format(fp.name)
        cmd = cpp + ';' + ctags
        status, output = subprocess.getstatusoutput(cmd)
        if status:
            raise OSError('Subprocess failed. Abort.')
    return output.split('\n')


def _includes(path):
    regex_include = re.compile('^#include [<"]([^>"]*)[">]')
    includes = []
    for line in open(path):
        if regex_include.match(line):
            includes.append(regex_include.match(line).group(1))
    return includes


def _tree_expand(tree, label, filename):
    names = tree[label].get(filename, []).copy()
    for name in names.copy():
        names.extend(_tree_expand(tree, label, name))
    return names

def _tree_includes(tree, label, parent):
    if label != 'hop':
        label = 'source/' + label
    return _tree_expand(tree, label, parent)


def _included_ids(path, tree, id_lists):
    label = lang(path).lower()
    filename = _filename(path)
    ids = id_lists[label].get(filename, []).copy()
    for include in _includes(path):
        logging.debug('{} includes {}'.format(filename, include))
        ids.extend(id_lists[label].get(include, []))
    for include in _tree_includes(tree, label, filename):
        logging.debug('{} tree includes {}'.format(filename, include))
        ids.extend(id_lists[label].get(include, []))
    return ids


def _add_identifier(args, filename, name, label, id_lists, known_ids, count):
    id_lists[label].setdefault(filename, [])
    if name in id_lists[label][filename]:
        return
    if name in known_ids:
        _remove_id(name, id_lists[label])
        if args.verbose:
            print('  Moved identifier: ', name)
        count['move'] += 1
    else:
        known_ids.append(name)
        if args.verbose:
            print('  New identifier: ', name)
        count['new'] += 1
    id_lists[label][filename].append(name)


def _all_hop_ids(tree, id_lists, filename):
    ids = id_lists['hop'].get(filename, []).copy()
    for name in tree['hop'].get(filename, []):
        ids.extend(id_lists['hop'].get(name, []))
    return ids


def _add_hop(args, path, name, id_lists, known_ids, tree, count):
    filename = translate.translate(os.path.basename(_filename(path)), 'hop')
    name = translate.to_hop(name)
    if name not in _all_hop_ids(tree, id_lists, filename):
        return _add_identifier(args, filename, name, 'hop', id_lists,
                               known_ids, count)


def scrape_header(args, path, tree, id_lists, known_ids, known_maps, count):
    """
    cpp -I. -DN
    ctags -x --c-kinds=defgtuvp --file-scope=no
    """
    label = lang(path).lower()
    filename = _filename(path)
    regex_lang = _regex_lang(path)
    if args.verbose:
        print('Scrape header: {}'.format(filename))
    included_ids = _included_ids(path, tree, id_lists)
    logging.debug('included_ids={}'.format(included_ids))
    for line in _ctags(path):
        if not line:
            break
        name = line.split()[0]
        if name not in known_maps:
            continue
        if (name.startswith('_')
                or name.endswith('_H')
                or not regex_lang.match(name)):
            continue
        if name in included_ids:
            count['old'] += 1
            continue
        _add_identifier(args, filename, name, label, id_lists, known_ids, count)
        _add_hop(args, filename, name, id_lists, known_ids, tree, count)
    if args.verbose:
        print('  Old identifiers:   {}'.format(count['old']))
        print('  New identifiers:   {}'.format(count['new']))
        print('  Moved identifiers: {}'.format(count['move']))
        print('')


def _all_identifiers(id_lists):
    ids = []
    for label in id_lists:
        for filename in id_lists[label]:
            ids.extend(id_lists[label][filename])
    return ids


def _known_maps(id_maps, triplets):
    ids = []
    for direction in id_maps.values():
        for lang in direction.values():
            ids.extend(lang.keys())
            ids.extend(lang.values())
    for hop, hip, cuda in triplets:
        ids.append(hop)
        ids.append(hip)
        ids.append(cuda)
    return ids

def _known_triplets(triplets, known_ids):
    known = []
    for hop, hip, cuda in triplets:
        if hop in known_ids or hip in known_ids or cuda in known_ids:
            known.append((hop, hip, cuda))
    return known


def scrape(args):
    tree = read_tree('data/file.tree')
    id_maps = {
            'source': read_map('data/source.map', source=True),
            'target': read_map('data/target.map'),
            }
    id_lists = {
            'hop': read_list('data/hop.list'),
            'hip': read_list('data/hip.list'),
            'cuda': read_list('data/cuda.list'),
            }
    known_ids = _all_identifiers(id_lists)
    logging.debug('tree={}'.format(tree))
    logging.debug('id_maps={}'.format(id_maps))
    logging.debug('id_lists={}'.format(id_lists))

    orig_maps = copy.deepcopy(id_maps)
    orig_lists = copy.deepcopy(id_lists)

    path = os.path.join(args.hipify, 'bin/hipify-perl')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    triplets = scrape_hipify(args, path, known_ids)
    known_maps = _known_maps(id_maps, triplets)

    count = {
            'old': 0,
            'new': 0,
            'move': 0,
            }
    for path in args.files:
        basename = os.path.basename(path)
        if basename.endswith('.h') or basename.endswith('.hpp'):
            scrape_header(args, path, tree, id_lists, known_ids, known_maps,
                          count)
        else:
            print('Unable to scrape: {}'.format(path))
    triplets = _known_triplets(triplets, known_ids)
    count['map'] = update_maps(args, id_maps, triplets, known_ids)
    print('')
    print('Old identifiers:    {}'.format(count['old']))
    print('New identifiers:    {}'.format(count['new']))
    print('Moved identifiers:  {}'.format(count['move']))
    print('New mapping chains: {}'.format(count['map']))

    logging.debug('id_maps={}'.format(id_maps))
    logging.debug('id_lists={}'.format(id_lists))
    todo = []
    if not id_maps['source'] == orig_maps['source']:
        todo.append('data/source.map')
    if not id_maps['target'] == orig_maps['target']:
        todo.append('data/target.map')
    if not id_lists['hop'] == orig_lists['hop']:
        todo.append('data/hop.list')
    if not id_lists['hip'] == orig_lists['hip']:
        todo.append('data/hip.list')
    if not id_lists['cuda'] == orig_lists['cuda']:
        todo.append('data/cuda.list')
    if not todo:
        return
    print('')
    print('Updated metadata:')
    print('  ' + '\n  '.join(todo))
    if args.force or input('Overwrite file(s)? [Y/n] ') in ['y', 'yes', '']:
        for filename in todo:
            base, ext = os.path.splitext(filename)
            label = os.path.basename(base)
            if ext == '.map':
                source = True if label == 'source' else False
                if not args.dry_run:
                    write_map(filename, id_maps[label], source, force=True)
            elif ext == '.list':
                if not args.dry_run:
                    write_list(filename, id_lists[label], force=True)
            else:
                raise ValueError('Unknown file type: {}'.format(filename))


if __name__ == '__main__':
    usage = '%(prog)s [options] hipify file.h {file2.h ...}'
    desc = 'Scrape files to suggest new identifiers and mappings.'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('hipify',
            help='path to HIPIFY git repository / installation')
    parser.add_argument('files', nargs='+',
            help='header files to scrape')
    parser.add_argument('-e', '--include-experimental',
            action='store_true', default=False,
            help='(hipify) include experimental substitutions')
    parser.add_argument('-u', '--include-unknown',
            action='store_true', default=False,
            help='(hipify) include unknown identifiers')
    parser.add_argument('-x', '--exclude', action='append', default=[],
            help='(hipify) exclude identifiers with this prefix')
    parser.add_argument('-g', '--exclude-group', action='append', default=[],
            help='(hipify) exclude substitution group (library, ...)')
    parser.add_argument('--force', action='store_true', default=False,
            help='force overwriting of existing files')
    parser.add_argument('-d', '--dry-run', action='store_true', default=False,
            help='run without modifying any files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
            help='display additional information while running')
    parser.add_argument('--debug', action='store_true', default=False,
            help='display additional information while running')

    args = parser.parse_args()

    # configure logging
    config = {'format': '[%(levelname)s] %(message)s'}
    if args.debug:
        config['level'] = logging.DEBUG
    logging.basicConfig(**config)
    logging.debug('args={}'.format(args))

    scrape(args)
