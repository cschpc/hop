#!/usr/bin/env python3

import os
import pathlib

from common.io import read_tree, read_map, read_list, file_path
from common.parser import ArgumentParser


def _check_regular_file(path, warn):
    if not os.path.exists(path):
        warn('File {} does not exist.'.format(path))
    elif not os.path.isfile(path):
        warn('File {} is not a regular file.'.format(path))


def _check_symbolic_link(link, target, warn):
    linked = pathlib.Path(link).resolve()
    if not os.path.exists(link):
        warn('Link {} does not exist.'.format(link))
    elif not os.path.islink(link):
        warn('Link {} is not a symbolic link.'.format(link))
    elif not linked.exists():
        warn('Link target {} does not exists.'.format(linked))
    elif not linked.is_file():
        warn('Link target {} is not a regular file.'.format(linked))
    elif linked != pathlib.Path(target).resolve():
        warn('Link {} does not point to {}.'.format(link, target))


def check_tree(tree):
    warnings = []
    warn = lambda x: warnings.append(x)
    for root in tree:
        for parent in tree[root]:
            path = file_path(os.path.join(root, parent))
            _check_regular_file(path, warn)
            for name in tree[root][parent]:
                link = file_path(os.path.join(root, name))
                _check_symbolic_link(link, path, warn)
    return warnings


def check_maps(tree, id_maps):
    warnings = []
    warn = lambda x: warnings.append(x)
    return warnings


def _all_files_in_tree(tree):
    files = {}
    for root in tree:
        label = root.replace('source/', '', 1)
        files.setdefault(label, [])
        for parent in tree[root]:
            files[label].append(parent)
            for name in tree[root][parent]:
                files[label].append(name)
    return files


def check_lists(tree, id_lists):
    warnings = []
    warn = lambda x: warnings.append(x)
    files = _all_files_in_tree(tree)
    for label in id_lists:
        if label == 'hop':
            root = label
        else:
            root = 'source/' + label
        for filename in id_lists[label]:
            path = file_path(os.path.join(root, filename))
            _check_regular_file(path, warn)
            if not filename in files[label]:
                warn('File {} not in file tree.'.format(filename))
    return warnings


def check(args):
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

    warnings = []
    warnings.extend(check_tree(tree))
    warnings.extend(check_lists(tree, id_lists))
    print('Warnings: {}'.format(len(warnings)))
    for msg in warnings:
        print(' ', msg)


if __name__ == '__main__':
    usage = '%(prog)s [options]'
    desc = 'Check consistency of metadata (in ../data/).'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('-d', '--dry-run', action='store_true', default=False,
            help='run without modifying any files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
            help='display additional information while running')

    args = parser.parse_args()

    check(args)
