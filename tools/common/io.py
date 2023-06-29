import re
import os
import string
import inspect

from common.map import Map

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


def corename(filename):
    core = re.sub('^(hop_|hop|hip_|hip|cuda_|cuda|cu)', '',
                  os.path.basename(filename))
    return re.sub('.h$', '', core)


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
    with open(file_path(filename)) as fp:
        txt = fp.read()

    tree = {}
    for block in regex_block.finditer(txt):
        root = block.group(1).strip()
        content = block.group(2).strip()

        tree[root] = {}
        parent = None
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elif line.startswith('+'):
                if parent is None:
                    raise SyntaxError('Orphaned file: ', line)
                tree[root][parent].append(line[1:].strip())
            else:
                parent = line
                tree[root][parent] = []
    return tree


def _find_identifiers(line):
    return [x[0] for x in regex_id.findall(line)]


def read_map(filename, source=False):
    with open(file_path(filename)) as fp:
        txt = fp.read()

    # if source translation, reverse order of mapping
    if source:
        order = lambda k,v: reversed((k, v))
    else:
        order = lambda k,v: (k,v)

    id_maps = {
            'hip': Map(label='hip', source=source),
            'cuda': Map(label='cuda', source=source),
            }
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
    with open(file_path(filename)) as fp:
        txt = fp.read()

    id_lists = {}
    for block in regex_block.finditer(txt):
        filename = block.group(1).strip()
        content = block.group(2).strip()

        id_lists.setdefault(filename, [])
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


def _ok_to_overwrite(path):
    answer = input('File {} exists. Overwrite? [Y/n] '.format(path))
    if answer.lower() in ['n', 'no']:
        return False
    return True


def write_header(path, content, force=False):
    path = file_path(path)
    if not force and os.path.exists(path) and not _ok_to_overwrite(path):
        return
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
            for parent in tree[root]:
                print(parent, file=fp)
                for name in tree[root][parent]:
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
                print(line, file=fp)


def write_all_maps(id_maps, force=False):
    write_map('data/source.map', id_maps['source'], source=True, force=force)
    write_map('data/target.map', id_maps['target'], force=force)


def write_list(filename, id_list, force=False):
    label, _ = os.path.splitext(filename)
    path = file_path(filename)
    if not force and os.path.exists(path) and not _ok_to_overwrite(path):
        return
    with open(path, 'w') as fp:
        print('# HOP identifier list for: {}'.format(label.upper()), file=fp)
        for key in id_list:
            print('', file=fp)
            print('[{}]'.format(key), file=fp)
            for identifier in sorted(id_list[key]):
                print(identifier, file=fp)


def write_all_lists(id_lists, force=False):
    for key in id_lists:
        filename = 'data/{}.list'.format(key)
        write_list(filename, id_lists[key], force=force)
