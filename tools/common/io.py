import re

# match '[file path] ...' grouped as '[\1] \2'
regex_block = re.compile('\s*\[([^]]+)\]\s*([^[]+)')
# match an identifier, e.g. 'id(foo, bar)' or 'id'
regex_id = re.compile('(\w+(\([\w\s,]*\))?)')


def read_tree(filename):
    with open(filename) as fp:
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
    with open(filename) as fp:
        txt = fp.read()

    # if source translation, reverse order of mapping
    if source:
        order = lambda k,v: reversed((k, v))
    else:
        order = lambda k,v: (k,v)

    id_maps = {'hip': {}, 'cuda': {}}
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
    with open(filename) as fp:
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
