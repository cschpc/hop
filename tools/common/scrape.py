import os
import re
import logging
import tempfile
import subprocess

from common.io import header_name, header_root, lang
from common.metadata import known_map_ids, translate


regex_perl_sub = re.compile('\nsub (\w+)\s+{([^}]*)}')
regex_subst = re.compile('\ssubst\("(\w+)", "(\w+)", "(\w+)"\);')

def _find_subst(txt, name):
    for sub in regex_perl_sub.findall(txt):
        if sub[0] == name:
            return regex_subst.findall(sub[1])
    return []


regex_perl_hash = re.compile('\nmy %(\w+)\s+=\s+\(([^)]*)\)')
regex_key_value = re.compile('"(\w+)"\s+=>\s+"([\w.]+)"')

def _find_key_values(txt, name):
    for blob in regex_perl_hash.findall(txt):
        logging.debug('blob={}'.format(blob))
        if blob[0] == name:
            return regex_key_value.findall(blob[1])
    return []


def _find_keys(txt, name):
    return [x[0] for x in _find_key_values(txt, name)]


def _version(string):
    logging.debug('_version < {}'.format(string))
    bits = []
    for bit in string.split('.'):
        try:
            bits.append(int(bit))
        except ValueError:
            bits.append(0)
    while len(bits) < 4:
        bits.append(0)
    logging.debug('_version > {}'.format(tuple(bits)))
    return tuple(bits)


def obsolete_ids(path, version=None):
    logging.debug('obsolete_ids < path={} version={}'.format(path, version))
    if version:
        ids = []
        for key, value in _find_key_values(open(path).read(), 'removed_funcs'):
            logging.debug('obsolete_ids: key={} value={}'.format(key, value))
            if _version(value) <= _version(version):
                logging.debug('obsolete_ids: {} <= {}'.format(value, version))
                ids.append(key)
        return ids
    return _find_keys(open(path).read(), 'removed_funcs')


# mistaken IDs in hipify
_errata_hipify = {
        'hipDeviceAttributeMaxBlocksPerMultiprocessor': 'hipDeviceAttributeMaxBlocksPerMultiProcessor',
        }

def scrape_hipify(path, verbose=False, experimental=False,
                  exclude=[], exclude_group=[], cuda_version=None):
    if verbose:
        print('Scrape hipify: {}'.format(path))
    txt = open(path).read()
    subs = []
    subs.extend(_find_subst(txt, 'simpleSubstitutions'))
    if experimental:
        subs.extend(_find_subst(txt, 'experimentalSubstitutions'))
    obsolete = obsolete_ids(txt, cuda_version)
    logging.debug('obsolete={}'.format(obsolete))

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
        elif cuda in obsolete:
            logging.debug('  ignore (obsolete)')
            continue
        triplets.append((hop, hip, cuda))
    if verbose:
        print('  Substitutions found: {}'.format(len(triplets)))
        print('')
    return triplets


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
        if not args.expand_macros:
            cpp += '-fdirectives-only '
        cpp += '-I{} {} {} > {}'.format(header_root(path), define, path,
                                        fp.name)
        # get only identifiers that are visible externally
        ctags = 'ctags -x --c-kinds=defgstuvp --extras=-F {}'.format(fp.name)
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
    if label == 'hop':
        root = label
    else:
        root = 'source/' + label
    filename = header_name(path)
    ids = metadata['list'][label].get(filename, []).copy()
    # grep header for includes
    for include in _includes(path):
        logging.debug('{} includes {}'.format(filename, include))
        if include in metadata['tree'][root]:
            node = metadata['tree'][root][include].link or include
        else:
            node = include
        logging.debug('include node {}'.format(node))
        n = len(ids)
        ids.extend(metadata['list'][label].get(node, []))
        logging.debug('{} IDs added'.format(len(ids) - n))
        for include in _tree_expand(metadata['tree'], root, node):
            logging.debug('{} expands to include {}'.format(node, include))
            n = len(ids)
            ids.extend(metadata['list'][label].get(include, []))
            logging.debug('{} IDs added'.format(len(ids) - n))
    # expand dependencies in the file tree
    if filename in metadata['tree'][root]:
        node = metadata['tree'][root][filename].link or filename
        logging.debug('include node {}'.format(node))
        n = len(ids)
        ids.extend(metadata['list'][label].get(node, []))
        logging.debug('{} IDs added'.format(len(ids) - n))
        for include in _tree_expand(metadata['tree'], root, node):
            logging.debug('{} tree includes {}'.format(filename, include))
            n = len(ids)
            ids.extend(metadata['list'][label].get(include, []))
            logging.debug('{} IDs added'.format(len(ids) - n))
    return ids


def _known_triplet_ids(triplets):
    ids = []
    for hop, hip, cuda in triplets:
        ids.append(hop)
        ids.append(hip)
        ids.append(cuda)
    return ids


def _add_identifier(args, filename, name, label, metadata, known_ids, count):
    metadata['list'][label].setdefault(filename, [])
    if name in metadata['list'][label][filename]:
        return
    if name in known_ids[label]:
        if not args.ignore_moved:
            _remove_id(name, metadata['list'][label])
            if args.verbose:
                print('  Moved identifier: ', name)
            count['move'] += 1
    else:
        known_ids[label].append(name)
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
    if (hop not in known_ids[label]
            or (label == 'hip'
                and hop not in _all_hop_ids(metadata, filename))):
        _add_identifier(args, filename, hop, 'hop', metadata, known_ids, count)


def scrape_header(args, path, metadata, known_ids, triplets, count):
    label = lang(path).lower()
    filename = header_name(path)
    if args.verbose:
        print('Scrape header: {}'.format(filename))
    included_ids = _included_ids(path, metadata)
    logging.debug('included_ids={}'.format(included_ids))
    known_maps = known_map_ids(metadata) + known_triplet_ids(triplets)
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
                or not translate.match(name)):
            logging.debug('  ignore (_ | regex)')
            continue
        if name in included_ids:
            count['old'] += 1
            logging.debug('  ignore (included_ids)')
            continue
        _add_identifier(args, filename, name, label, metadata, known_ids, count)
        _add_hop(args, filename, name, label, metadata, known_ids, triplets,
                 count)
    if args.verbose:
        print('  Old identifiers:   {}'.format(count['old']))
        print('  New identifiers:   {}'.format(count['new']))
        print('  Moved identifiers: {}'.format(count['move']))
        print('')
