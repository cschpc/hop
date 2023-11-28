import re
import os
import string
import inspect
import logging

from common import io
from common.metadata import Include, Embed


_license = io.read_license()


def _custom_template(filename, default):
    logging.debug('_custom_template() < {} {}'.format(filename, default))
    name = filename.replace('.', '_').replace('/', '.')
    if name.startswith('hop.source.hip'):
        _, ext = os.path.splitext(name)
        name = 'source.hip' + ext
    elif name.startswith('hop.source.cuda'):
        name = name.replace('hop.', '')
    elif name.startswith('hop.'):
        name = name.replace('hop.', 'target.')
    template = os.path.join('data/templates', 'template.' + name)
    logging.debug('template={}'.format(template))
    if os.path.exists(template):
        return template
    return os.path.join('data/templates', default)


def _fill_template(template, args):
    template = io.read_template(template)
    args['template'] = template.safe_substitute(args)
    header = io.read_template('data/templates/template.h')
    return header.safe_substitute(args)


def source_header(filename, content):
    sentinel = os.path.basename(filename).replace('.', '_').upper()
    args = {
            'license': _license,
            'sentinel': 'HOP_SOURCE_{}'.format(sentinel),
            'content': content,
            'include': 'hop/hop_{}.h'.format(io.corename(filename)),
            'lang': io.lang(filename),
            }
    logging.debug('source_header: args={}'.format(args))
    return _fill_template(_custom_template(filename, 'template.source'), args)


def target_header(filename, include, content):
    sentinel = os.path.basename(filename).replace('.', '_').upper()
    args = {
            'license': _license,
            'sentinel': sentinel,
            'content': content,
            'include': include,
            }
    logging.debug('target_header: args={}'.format(args))
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
    logging.debug('hop_header: args={}'.format(args))
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
    return ['#include <{}>'.format(x) for x in node if type(x) is Include]


def _embed(node):
    return [x for x in node if type(x) is Embed]


def content(node, id_map, id_list):
    logging.debug('content() < node={}'.format(node))
    lines = _includes(node)
    if len(lines):
        lines.append('')
    lines.extend(_defines(id_list.get(node.name, []), id_map))
    for name in _embed(node):
        if name not in id_list:
            continue
        if len(lines):
            lines.append('')
        lines.append('/* {} */'.format(name))
        lines.extend(_defines(id_list[name], id_map))
    if len(lines):
        lines.insert(0, '')
        if lines[-1]:
            lines.append('')
    return '\n'.join(lines)


def _coretree(tree):
    cores = {}
    for root in tree:
        cores[root] = {}
        for filename in tree[root]:
            cores[root][io.corename(filename)] = filename
    return cores


def make_headers(metadata):
    coretree = _coretree(metadata['tree'])
    headers = {}
    branch = metadata['tree']['hop']
    for node in branch.values():
        if node.link:
            continue
        corename = io.corename(node.name)
        coresubs = [io.corename(x) for x in node]

        # main hop header
        path = os.path.join('hop', node.name)
        hipname = coretree['source/hip'][corename]
        cudaname = coretree['source/cuda'][corename]
        headers[path] = hop_header(path, hipname, cudaname)

        # target header for hip
        path_hip = path.replace('.h', '_hip.h')
        content_hip = content(node, metadata['map']['target']['hip'],
                              metadata['list']['hop'])
        headers[path_hip] = target_header(path_hip, hipname, content_hip)

        # target header for cuda
        path_cuda = path.replace('.h', '_cuda.h')
        content_cuda = content(node, metadata['map']['target']['cuda'],
                               metadata['list']['hop'])
        headers[path_cuda] = target_header(path_cuda, cudaname, content_cuda)

    # source header for HIP
    branch = metadata['tree']['source/hip']
    for node in branch.values():
        if node.link:
            continue
        path = os.path.join('hop/source/hip', node.name)
        content_hip = content(node, metadata['map']['source']['hip'],
                              metadata['list']['hip'])
        headers[path] = source_header(path, content_hip)

    # source header for CUDA
    branch = metadata['tree']['source/cuda']
    for node in branch.values():
        if node.link:
            continue
        path = os.path.join('hop/source/cuda', node.name)
        content_cuda = content(node, metadata['map']['source']['cuda'],
                               metadata['list']['cuda'])
        headers[path] = source_header(path, content_cuda)

    return headers
