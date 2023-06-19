import re
import os
import string
import inspect

from common import io


_license = io.read_license()


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
            'lang': io.lang(filename),
            'content': content,
            'include': 'hop_{}.h'.format(io.corename(filename)),
            }
    return _fill_template('template.source', args)


def target_header(filename, include, content):
    sentinel = os.path.basename(filename).replace('.', '_').upper()
    args = {
            'license': _license,
            'sentinel': sentinel,
            'include': include,
            'content': content,
            }
    return _fill_template('template.target', args)


def hop_header(filename, source_hip, source_cuda):
    sentinel = os.path.basename(filename).replace('.', '_').upper()
    args = {
            'license': _license,
            'sentinel': sentinel,
            'source_hip': source_hip,
            'source_cuda': source_cuda,
            'corename': io.corename(filename),
            }
    return _fill_template('template.hop', args)


def _format_define(src, tgt):
    if len(src) > 32:
        return '#define {}  \\\n        {}'.format(src, tgt)
    else:
        return '#define {:32} {}'.format(src, tgt)


def defines(id_list, id_map):
    defs = []
    for src in id_list:
        tgt = id_map[src]
        defs.append(_format_define(src, tgt))
    return defs


def content(filename, branch, id_map, id_lists):
    lines = defines(id_lists[filename], id_map)
    for sub in branch[filename]:
        lines.append('')
        lines.append('/* {} */'.format(sub))
        lines.extend(defines(id_lists[sub], id_map))
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
    for filename in branch:
        corename = io.corename(filename)

        # main hop header
        path = os.path.join('hop', filename)
        source_hip = coretree['source/hip'][corename]
        source_cuda = coretree['source/cuda'][corename]
        headers[path] = hop_header(filename, source_hip, source_cuda)

        # target header for hip
        path_hip = path.replace('.h', '_hip.h')
        target_hip = filename.replace('.h', '_hip.h')
        content_hip = content(filename, branch, id_maps['target']['hip'],
                              id_lists['hop'])
        headers[path_hip] = target_header(target_hip, source_hip, content_hip)

        # target header for cuda
        path_cuda = path.replace('.h', '_cuda.h')
        target_cuda = filename.replace('.h', '_cuda.h')
        content_cuda = content(filename, branch, id_maps['target']['cuda'],
                               id_lists['hop'])
        headers[path_cuda] = target_header(target_cuda, source_cuda,
                                           content_cuda)

    # source header for HIP
    branch = tree['source/hip']
    for filename in branch:
        path = os.path.join('source/hip', filename)
        content_hip = content(filename, branch, id_maps['source']['hip'],
                              id_lists['hip'])
        headers[path] = source_header(filename, content_hip)

    # source header for CUDA
    branch = tree['source/cuda']
    for filename in branch:
        path = os.path.join('source/cuda', filename)
        content_cuda = content(filename, branch, id_maps['source']['cuda'],
                               id_lists['cuda'])
        headers[path] = source_header(filename, content_cuda)

    return headers
