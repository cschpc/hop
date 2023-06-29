#!/usr/bin/env python3

import os
import re
import tempfile
import subprocess

from common.headers import make_headers
from common.io import lang, read_tree, read_map, read_list, write_header
from common.parser import ArgumentParser
from common.map import Map


regex_perl_sub = re.compile('\nsub (\w+)\s+{([^}]*)}')
regex_subst = re.compile('\ssubst\("(\w+)", "(\w+)", "(\w+)"\);')

def _find_subst(txt, name):
    for sub in regex_perl_sub.findall(txt):
        if sub[0] == name:
            return regex_subst.findall(sub[1])
    return []


regex_lower = re.compile('^(.*?)(hop|hip|Hip|cuda|Cuda|cu|Cu)')
regex_upper = re.compile('^(.*?)(HOP|HIP|CUDA|CU)')

def translate(identifier, target):
    if regex_lower.match(identifier):
        return regex_lower.sub(r'\1' + target, identifier)
    if regex_upper.match(identifier):
        return regex_upper.sub(r'\1' + target.upper(), identifier)
    return identifier


def known_mapping(id_maps, ):
    if cuda in id_maps['source']['cuda']:
        if not id_maps['source']['cuda'][cuda] == hop:
            return True
    if hip in id_maps['target']['hip']:
        if not id_maps['target']['hip'][hop] == hip:
            return True

    hop = src_map[key]
    return tgt_map[src_map[key]]


def update_maps(id_maps, triplets, verbose=False):
    count = 0
    for hop, hip, cuda in triplets:
        # cuda -> hip translation
        _hop = hop
        if cuda in id_maps['source']['cuda']:
            _hop = id_maps['source']['cuda'][cuda]
        if _hop not in id_maps['target']['hip']:
            id_maps['source']['cuda'][cuda] = _hop
            id_maps['target']['hip'][_hop] = hip
            count += 1
            print('New mapping: {} -> {} -> {}'.format(cuda, _hop, hip))
        # hip -> cuda translation
        _hop = hop
        if hip in id_maps['source']['hip']:
            _hop = id_maps['source']['hip'][hip]
        if _hop not in id_maps['target']['cuda']:
            id_maps['source']['hip'][hip] = _hop
            id_maps['target']['cuda'][_hop] = cuda
            count += 1
            if verbose:
                print('New mapping: {} -> {} -> {}'.format(hip, _hop, cuda))
    return count


def scrape_hipify(args, path, known_ids, id_maps):
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
        hop = translate(hip, 'gpu')
        triplets.append((hop, hip, cuda))
    count = update_maps(id_maps, triplets, args.verbose)
    if args.verbose:
        print('')
        print('Substitutions: found: {}'.format(len(triplets)))
        print('New mapping chains:  {}'.format(count))


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


def _remove_id(name, known_ids, id_list):
    for filename in id_list:
        if name in id_list[filename]:
            id_list[filename].remove(name)
    known_ids.remove(name)


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


def scrape_header(args, path, known_ids, id_lists):
    """
    cpp -I. -DN
    ctags -x --c-kinds=defgtuvp --file-scope=no
    """
    label = lang(path).lower()
    filename = _filename(path)
    regex_lang = _regex_lang(path)
    count = 0
    for line in _ctags(path):
        name = line.split()[0]
        if (name.startswith('_')
                or name.endswith('_H')
                or not regex_lang.match(name)
                or name in id_lists[label].get(filename, [])):
            continue
        if name in known_ids:
            _remove_id(name, known_ids, id_lists[label])
        id_lists[label].setdefault(filename, [])
        id_lists[label][filename].append(name)
        if args.verbose:
            print('New identifier: ', name)
        count += 1
    print('New identifiers: {}'.format(count))


def _all_identifiers(id_lists):
    ids = []
    for label in id_lists:
        for filename in id_lists[label]:
            ids.extend(id_lists[label][filename])
    return ids


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

    for path in args.files:
        basename = os.path.basename(path)
        if basename == 'hipify-perl':
            scrape_hipify(args, path, known_ids, id_maps)
        elif basename.endswith('.h'):
            scrape_header(args, path, known_ids, id_lists)
        else:
            print('Unable to scrape {}'.format(path))


if __name__ == '__main__':
    usage = '%(prog)s [options] file {file2 ...}'
    desc = 'Scrape files to suggest new identifiers and mappings.'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('files', nargs='+',
            help='files to scrape')
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
    parser.add_argument('-d', '--dry-run', action='store_true', default=False,
            help='run without modifying any files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
            help='display additional information while running')

    args = parser.parse_args()

    scrape(args)
