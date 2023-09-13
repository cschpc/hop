import re
import os
import string
import inspect

from common import io


_license = io.read_license()


def _custom_template(filename, default):
    name = filename.replace('/', '.')
    if name.startswith('hop.'):
        name = name.replace('hop.', 'target.')
    template = os.path.join('data/templates', 'template.' + name)
    if os.path.exists(template):
        return template
    return os.path.join('data/templates', default)


def _fill_template(template, args):
    template = io.read_template('data/templates/' + template)
    args['template'] = template.safe_substitute(args)
    header = io.read_template('data/templates/template.h')
    return header.safe_substitute(args)


def source_header(filename, content):
    sentinel = os.path.basename(filename).replace('.', '_').upper()
    args = {
            'license': _license,
            'sentinel': 'HOP_SOURCE_{}'.format(sentinel),
            'content': content,
            'include': 'hop_{}.h'.format(io.corename(filename)),
            'lang': io.lang(filename),
            }
    return _fill_template(_custom_template(filename, 'template.source'), args)


def target_header(filename, include, content):
    sentinel = os.path.basename(filename).replace('.', '_').upper()
    args = {
            'license': _license,
            'sentinel': sentinel,
            'content': content,
            'include': include,
            }
    return _fill_template(_custom_template(filename, 'template.target'), args)


def hop_header(filename, hipname, cudaname):
    sentinel = os.path.basename(filename).replace('.', '_').upper()
    args = {
            'license': _license,
            'sentinel': sentinel,
            'corename': io.corename(filename),
            'cudaname': cudaname,
            'hipname': hipname,
            }
    return _fill_template(_custom_template(filename, 'template.hop'), args)


def _format_define(src, tgt):
    if len(src) > 32:
        return '#define {}  \\\n        {}'.format(src, tgt)
    else:
        return '#define {:32} {}'.format(src, tgt)


def _defines(id_list, id_map):
    defs = []
    for src in id_list:
        tgt = id_map[src]
        defs.append(_format_define(src, tgt))
    return defs


def _includes(node):
    return ['#include <{}>'.format(x) for x in node]


def content(node, id_map, id_lists):
    lines = _includes(node)
    lines.append('')
    lines.extend(_defines(id_lists.get(node.name, []), id_map))
    return '\n'.join(lines)


def _coretree(tree):
    cores = {}
    for root in tree:
        cores[root] = {}
        for filename in tree[root]:
            cores[root][io.corename(filename)] = filename
    return cores


def make_headers(tree, id_maps, id_lists):
    coretree = _coretree(tree)
    headers = {}
    branch = tree['hop']
    for node in branch.values():
        corename = io.corename(node.name)
        coresubs = [io.corename(x) for x in node]

        # main hop header
        path = os.path.join('hop', node.name)
        hipname = coretree['source/hip'][corename]
        cudaname = coretree['source/cuda'][corename]
        headers[path] = hop_header(path, hipname, cudaname)

        # target header for hip
        path_hip = path.replace('.h', '_hip.h')
        content_hip = content(node, id_maps['target']['hip'], id_lists['hop'])
        headers[path_hip] = target_header(path_hip, hipname, content_hip)

        # target header for cuda
        path_cuda = path.replace('.h', '_cuda.h')
        content_cuda = content(node, id_maps['target']['cuda'], id_lists['hop'])
        headers[path_cuda] = target_header(path_cuda, cudaname, content_cuda)

    # source header for HIP
    branch = tree['source/hip']
    for node in branch.values():
        path = os.path.join('source/hip', node.name)
        content_hip = content(node, id_maps['source']['hip'], id_lists['hip'])
        headers[path] = source_header(path, content_hip)

    # source header for CUDA
    branch = tree['source/cuda']
    for node in branch.values():
        path = os.path.join('source/cuda', node.name)
        content_cuda = content(node, id_maps['source']['cuda'],
                               id_lists['cuda'])
        headers[path] = source_header(path, content_cuda)

    return headers
