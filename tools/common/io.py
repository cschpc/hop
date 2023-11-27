import re
import os
import string
import inspect
import logging
import pathlib

from common.abc import UniqueList
from common.metadata import Map, Node, Include, Embed, Special

# match '[file path] ...' grouped as '[\1] \2'
regex_block = re.compile('\s*\[([^]]+)\]\s*([^[]+)')
# match an identifier, e.g. 'id(foo, bar)' or 'id'
regex_id = re.compile('(\w+(\([\w\s,]*\))?)')


def _root_path():
    src = inspect.getsourcefile(lambda:42)
    path = os.path.join(os.path.dirname(src), '../..')
    return os.path.abspath(path)


def file_path(filename):
    return os.path.join(_root_path(), filename)


def _in_hip_root(path):
    hip = os.path.join(path, 'hip')
    return os.path.exists(hip) and os.path.isdir(hip)


def _split_at_hip_root(path):
    subs = []
    while path and not _in_hip_root(path):
        path, tail = os.path.split(path)
        logging.debug('_split_at_hip_root: path={} tail={}'.format(path, tail))
        subs.append(tail)
    if subs:
        sub = os.path.join(*reversed(subs))
    else:
        sub = None
    logging.debug('_split_at_hip_root > ({}, {})'.format(path, sub))
    return (path, sub)


def header_name(path):
    dirname, filename = os.path.split(path)
    if lang(path) == 'HIP':
        dirname, subname = _split_at_hip_root(dirname)
        if subname:
            filename = os.path.join(subname, filename)
    logging.debug('header_name < {} > {}'.format(path, filename))
    return filename


def header_root(path):
    dirname, filename = os.path.split(path)
    if lang(path) == 'HIP':
        dirname, subname = _split_at_hip_root(dirname)
    logging.debug('header_root < {} > {}'.format(path, dirname))
    return dirname


def corename(filename):
    core = re.sub('^(hop_|hop|hip_|hip|cuda_|cuda|cu)', '',
                  os.path.basename(filename))
    return re.sub('.h$', '', core).lower()


def lang(filename):
    basename = os.path.basename(filename).lower()
    if basename.startswith('hip'):
        return 'HIP'
    elif basename.startswith('cu'):
        return 'CUDA'
    elif basename.startswith('hop'):
        return 'HOP'
    elif filename:
        return lang(os.path.dirname(filename))
    else:
        return None


def read_tree(filename):
    tree = {}
    try:
        with open(file_path(filename)) as fp:
            txt = fp.read()
    except FileNotFoundError:
        return tree

    for block in regex_block.finditer(txt):
        root = block.group(1).strip()
        content = block.group(2).strip()

        tree[root] = {}
        node = None
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elif line.startswith('+'):
                if node is None:
                    raise SyntaxError('Orphaned file: ' + line)
                node.append(Include(line[1:].strip()))
            elif line.startswith('-'):
                if node is None:
                    raise SyntaxError('Orphaned link: ' + line)
                node.append(Embed(line[1:].strip()))
            elif line.startswith('='):
                if node is None:
                    raise SyntaxError('Orphaned link: ' + line)
                node.link = line[1:].strip()
                node = None
            elif line.startswith('~'):
                if node is None:
                    raise SyntaxError('Orphaned file: ' + line)
                node.append(Special(line[1:].strip()))
            else:
                node = Node(name=line)
                tree[root][node.name] = node
    return tree


def _find_identifiers(line):
    return [x[0] for x in regex_id.findall(line)]


def read_map(filename, source=False):
    id_maps = {
            'hip': Map(label='hip', source=source),
            'cuda': Map(label='cuda', source=source),
            }
    try:
        with open(file_path(filename)) as fp:
            txt = fp.read()
    except FileNotFoundError:
        return id_maps

    # if source translation, reverse order of mapping
    if source:
        order = lambda k,v: reversed((k, v))
    else:
        order = lambda k,v: (k,v)

    for block in regex_block.finditer(txt):
        label = block.group(1).strip().lower()
        content = block.group(2).strip()

        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elif label == '*':
                hop, hip, cuda = _find_identifiers(line)
            elif label == 'hip':
                hop, hip = _find_identifiers(line)
                cuda = None
            elif label == 'cuda':
                hop, cuda = _find_identifiers(line)
                hip = None
            else:
                raise ValueError(
                        'Unknown label [{0}] in {1}'.format(label, filename))

            for lang, other in zip(['hip', 'cuda'], [hip, cuda]):
                if other is not None:
                    key, value = order(hop, other)
                    id_maps[lang][key] = value
    return id_maps


def read_list(filename):
    id_lists = {}
    try:
        with open(file_path(filename)) as fp:
            txt = fp.read()
    except FileNotFoundError:
        return id_lists

    for block in regex_block.finditer(txt):
        filename = block.group(1).strip()
        content = block.group(2).strip()

        id_lists.setdefault(filename, UniqueList())
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            id_lists[filename].append(line)
    return id_lists


def read_template(filename):
    with open(file_path(filename)) as fp:
        template = string.Template(fp.read())
    return template


def read_license():
    with open(file_path('LICENSE')) as fp:
        txt = fp.read()
    return re.sub('^MIT License\s*', '', txt)


def read_metadata(base='data'):
    _base = lambda x: os.path.join(base, x)
    metadata = {}
    metadata['tree'] = read_tree(_base('file.tree'))
    metadata['map'] = {
            'source': read_map(_base('source.map'), source=True),
            'target': read_map(_base('target.map')),
            }
    metadata['list'] = {
            'hop': read_list(_base('hop.list')),
            'hip': read_list(_base('hip.list')),
            'cuda': read_list(_base('cuda.list')),
            }
    logging.debug('read metadata={}'.format(metadata))
    return metadata


def _ok_to_overwrite(path):
    answer = input('File {} exists. Overwrite? [Y/n] '.format(path))
    if answer.lower() in ['n', 'no']:
        return False
    return True


def _create_directory(path, force=False):
    if not force:
        answer = input("Create directory {} ? [Y/n] ".format(path))
        if answer.lower() in ['n', 'no']:
            return False
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def write_header(path, content, force=False):
    path = file_path(path)
    if not force and os.path.exists(path) and not _ok_to_overwrite(path):
        return
    if not os.path.exists(os.path.dirname(path)):
        _create_directory(os.path.dirname(path), force)
    with open(path, 'w') as fp:
        fp.write(content)


def write_tree(filename, tree, force=False):
    path = file_path(filename)
    if not force and os.path.exists(path) and not _ok_to_overwrite(path):
        return
    with open(path, 'w') as fp:
        print('# HOP file tree', file=fp)
        for root in tree:
            print('', file=fp)
            print('[{}]'.format(root), file=fp)
            for node in tree[root].values():
                print('', file=fp)
                print(node.name, file=fp)
                if node.link:
                    print('= {}'.format(node.link), file=fp)
                    continue
                for name in node:
                    print('+ {}'.format(name), file=fp)


def _join_map(id_map, source=False):
    # if source map, order is reversed
    if source:
        order = lambda x: [(v,k) for k,v in x]
    else:
        order = lambda x: x

    joint_map = {}
    for label in id_map:
        for key, value in order(id_map[label].items()):
            joint_map.setdefault(key, {})
            joint_map[key].setdefault(label, [])
            joint_map[key][label].append(value)
    return joint_map


def _closest_match(hop, mapping, key):
    if key not in mapping or not len(mapping[key]):
        return None
    if len(mapping[key]) == 1:
        return mapping[key][0]
    regex_core = re.compile('^(gpu|hip|cuda|cu)')
    # identical core name with default prefix (hip | cuda)
    best = regex_core.sub(key, hop)
    if best in mapping[key]:
        return best
    # identical core name
    core = regex_core.sub('', hop)
    for other in mapping[key]:
        if core == regex_core.sub('', other):
            return other
    # no idea, just return the first
    return mapping[key][0]


def _max_width(data):
    if not data:
        return 0
    width = [0] * len(data[0])
    for line in data:
        for i, column in enumerate(line):
            if len(column) > width[i]:
                width[i] = len(column)
    return width


def _format_columns(values, width):
    if len(width) == 3:
        template = '{:%d}  {:%d}  {:%d}' % tuple(width)
    else:
        template = '{:%d}  {:%d}' % tuple(width)
    return template.format(*values)


def write_map(filename, id_map, source=False, force=False):
    path = file_path(filename)
    if not force and os.path.exists(path) and not _ok_to_overwrite(path):
        return
    output = {
            '*': [],
            'hip': [],
            'cuda': [],
            }
    for hop, mapping in _join_map(id_map, source=source).items():
        hip = _closest_match(hop, mapping, 'hip')
        cuda = _closest_match(hop, mapping, 'cuda')
        if hip and cuda:
            output['*'].append((hop, hip, cuda))
            mapping['hip'].remove(hip)
            mapping['cuda'].remove(cuda)
        for label in mapping:
            for other in mapping[label]:
                output[label].append((hop, other))
    with open(path, 'w') as fp:
        print('# HOP identifier map', file=fp)
        for label in ['*', 'hip', 'cuda']:
            print('', file=fp)
            print('[{}]'.format(label), file=fp)
            width = _max_width(output[label])
            for ids in sorted(output[label]):
                line = _format_columns(ids, width)
                print(line.rstrip(), file=fp)


def write_all_maps(id_maps, force=False):
    write_map('data/source.map', id_maps['source'], source=True, force=force)
    write_map('data/target.map', id_maps['target'], force=force)


def write_list(filename, id_list, force=False):
    base, _ = os.path.splitext(filename)
    label = os.path.basename(base)
    path = file_path(filename)
    if not force and os.path.exists(path) and not _ok_to_overwrite(path):
        return
    with open(path, 'w') as fp:
        print('# HOP identifier list for: {}'.format(label.upper()), file=fp)
        for key in sorted(id_list):
            print('', file=fp)
            print('[{}]'.format(key), file=fp)
            for identifier in sorted(set(id_list[key])):
                print(identifier, file=fp)


def write_all_lists(id_lists, force=False):
    for key in id_lists:
        filename = 'data/{}.list'.format(key)
        write_list(filename, id_lists[key], force=force)


def write_metadata(metadata, force=False, dry_run=False):
    orig_metadata = read_metadata()
    todo = []
    if not metadata['map']['source'] == orig_metadata['map']['source']:
        todo.append('data/source.map')
    if not metadata['map']['target'] == orig_metadata['map']['target']:
        todo.append('data/target.map')
    if not metadata['list']['hop'] == orig_metadata['list']['hop']:
        todo.append('data/hop.list')
    if not metadata['list']['hip'] == orig_metadata['list']['hip']:
        todo.append('data/hip.list')
    if not metadata['list']['cuda'] == orig_metadata['list']['cuda']:
        todo.append('data/cuda.list')
    if not todo:
        return
    print('')
    print('Updated metadata:')
    print('  ' + '\n  '.join(todo))
    if force or input('Overwrite file(s)? [Y/n] ') in ['y', 'yes', '']:
        for filename in todo:
            base, ext = os.path.splitext(filename)
            label = os.path.basename(base)
            if ext == '.map':
                source = True if label == 'source' else False
                if not dry_run:
                    write_map(filename, metadata['map'][label], source,
                              force=True)
            elif ext == '.list':
                if not dry_run:
                    write_list(filename, metadata['list'][label], force=True)
            else:
                raise ValueError('Unknown file type: {}'.format(filename))
