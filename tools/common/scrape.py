import os
import re
import logging
import tempfile
import subprocess

from common.io import header_name, header_root, lang
from common.metadata import translate


regex_perl_sub = re.compile('\nsub (\w+)\s+{([^}]*)}')
regex_subst = re.compile('\ssubst\("(\w+)", "(\w+)", "(\w+)"\);')

def _find_subst(txt, name):
    for sub in regex_perl_sub.findall(txt):
        if sub[0] == name:
            return regex_subst.findall(sub[1])
    return []


# mistaken IDs in hipify
_errata_hipify = {
        'hipDeviceAttributeMaxBlocksPerMultiprocessor': 'hipDeviceAttributeMaxBlocksPerMultiProcessor',
        }

def scrape_hipify(path, verbose=False, experimental=False,
                  exclude=[], exclude_group=[]):
    if verbose:
        print('Scrape hipify: {}'.format(path))
    txt = open(path).read()
    subs = []
    subs.extend(_find_subst(txt, 'simpleSubstitutions'))
    if experimental:
        subs.extend(_find_subst(txt, 'experimentalSubstitutions'))

    if exclude:
        # exclude IDs with prefix
        regex_exclude = re.compile('^({})'.format('|'.join(exclude)))
        _exclude = lambda x: regex_exclude.match(x)
    else:
        _exclude = lambda x: False

    triplets = []
    for cuda, hip, group in subs:
        # correct for any mistaken IDs in hipify
        hip = _errata_hipify.get(hip, hip)
        cuda = _errata_hipify.get(cuda, cuda)
        hop = translate.to_hop(hip)
        logging.debug('scrape_hipify: ({}, {}, {})'.format(hop, hip, cuda))
        # skip excluded IDs
        if group in exclude_group:
            logging.debug('  ignore (group)')
            continue
        elif _exclude(cuda) or _exclude(hip):
            logging.debug('  ignore (exclude)')
            continue
        triplets.append((hop, hip, cuda))
    if verbose:
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


def _remove_id(name, id_list):
    for filename in id_list:
        if name in id_list[filename]:
            id_list[filename].remove(name)


def _ctags(args, path):
    with tempfile.NamedTemporaryFile(prefix='tmp-hop-scrape-',
                                     suffix='.h') as fp:
        define = ''
        if lang(path) == 'HIP':
            define += ' -D__HIP_PLATFORM_AMD__'
            if args.cpp_macros:
                define += ' -D__HIPCC__'
        else:
            if args.cpp_macros:
                define += ' -D__CUDACC__'
        # preprocess to get also included files
        cpp = 'cpp '
        if args.cpp_macros:
            cpp = 'c++ -E '
        cpp += '-I{} {} {} > {}'.format(header_root(path), define, path,
                                        fp.name)
        # get only identifiers that are visible externally
        ctags = 'ctags -x --c-kinds=defgtuvp --extras=-F {}'.format(fp.name)
        cmd = cpp + ' ; ' + ctags
        logging.debug('_ctags command: ' + cmd)
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


def _included_ids(path, metadata):
    label = lang(path).lower()
    filename = header_name(path)
    ids = metadata['list'][label].get(filename, []).copy()
    for include in _includes(path):
        logging.debug('{} includes {}'.format(filename, include))
        ids.extend(metadata['list'][label].get(include, []))
    if label == 'hop':
        root = label
    else:
        root = 'source/' + label
    if filename in metadata['tree'][root]:
        node = metadata['tree'][root][filename].link or filename
        for include in _tree_expand(metadata['tree'], root, node):
            logging.debug('{} tree includes {}'.format(filename, include))
            ids.extend(metadata['list'][label].get(include, []))
    return ids


def _known_maps(metadata, triplets):
    ids = []
    for direction in metadata['map'].values():
        for lang in direction.values():
            ids.extend(lang.keys())
            ids.extend(lang.values())
    for hop, hip, cuda in triplets:
        ids.append(hop)
        ids.append(hip)
        ids.append(cuda)
    return ids


def _add_identifier(args, filename, name, label, metadata, known_ids, count):
    metadata['list'][label].setdefault(filename, [])
    if name in metadata['list'][label][filename]:
        return
    if name in known_ids:
        if not args.ignore_moved:
            _remove_id(name, metadata['list'][label])
            if args.verbose:
                print('  Moved identifier: ', name)
            count['move'] += 1
    else:
        known_ids.append(name)
        if args.verbose:
            print('  New identifier: ', name)
        count['new'] += 1
    metadata['list'][label][filename].append(name)


def _all_hop_ids(metadata, filename):
    ids = metadata['list']['hop'].get(filename, []).copy()
    for name in metadata['tree']['hop'].get(filename, []):
        ids.extend(metadata['list']['hop'].get(name, []))
    return ids


def _find_hop(triplets, name, label):
    if label == 'hip':
        index = 1
    else:
        index = 2
    logging.debug('_find_hop: name={}  label={}  index={}'.format(
                  name, label, index))
    for triplet in triplets:
        if triplet[index] == name:
            logging.debug('triplet={}'.format(triplet))
            return triplet[0]
    return None


def _add_hop(args, path, name, label, metadata, known_ids, triplets, count):
    filename = translate.translate(os.path.basename(path), 'hop')
    if name in metadata['map']['source'][label]:
        hop = metadata['map']['source'][label][name]
    else:
        hop = _find_hop(triplets, name, label)
    if not hop:
        if label == 'hip':
            hop = translate.to_hop(name)
        else:
            return
    logging.debug('_add_hop: hop={}'.format(hop))
    if (hop not in known_ids
            or (label == 'hip'
                and hop not in _all_hop_ids(metadata, filename))):
        _add_identifier(args, filename, hop, 'hop', metadata, known_ids, count)


def scrape_header(args, path, metadata, known_ids, triplets, count):
    label = lang(path).lower()
    filename = header_name(path)
    regex_lang = _regex_lang(path)
    if args.verbose:
        print('Scrape header: {}'.format(filename))
    included_ids = _included_ids(path, metadata)
    logging.debug('included_ids={}'.format(included_ids))
    known_maps = _known_maps(metadata, triplets)
    for line in _ctags(args, path):
        if not line:
            break
        name = line.split()[0]
        logging.debug('scrape_header: name={}'.format(name))
        if name not in known_maps:
            logging.debug('  ignore (known_maps)')
            continue
        if (name.startswith('_')
                or name.endswith('_H')
                or not regex_lang.match(name)):
            logging.debug('  ignore (_ | regex)')
            continue
        if name in included_ids:
            count['old'] += 1
            continue
        _add_identifier(args, filename, name, label, metadata, known_ids, count)
        _add_hop(args, filename, name, label, metadata, known_ids, triplets,
                 count)
    if args.verbose:
        print('  Old identifiers:   {}'.format(count['old']))
        print('  New identifiers:   {}'.format(count['new']))
        print('  Moved identifiers: {}'.format(count['move']))
        print('')
