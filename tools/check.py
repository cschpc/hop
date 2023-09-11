#!/usr/bin/env python3

import os
import pathlib
import logging

from common.io import read_tree, read_map, read_list, file_path
from common.abc import UniqueList
from common.parser import ArgumentParser
from common.reference import reference_map


warnings = UniqueList()
warn = lambda x: warnings.append(x)


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
    for root in tree:
        for node in tree[root].values():
            path = file_path(os.path.join(root, node.name))
            _check_regular_file(path, warn)
            if node.link:
                target = file_path(os.path.join(root, node.link))
                _check_symbolic_link(path, target, warn)
                continue
            for name in node:
                path = file_path(os.path.join(root, name))
                _check_regular_file(path, warn)


def check_maps(tree, id_maps, reference):
    if not reference:
        return warnings
    for cuda, hop in id_maps['source']['cuda'].items():
        hip = id_maps['target']['hip'][hop]
        if cuda not in reference['cuda']:
            warn('No reference mapping for {}'.format(cuda))
            continue
        if hip != reference['cuda'][cuda]:
            warn('Incorrect mapping: {} -> {} -> {}'.format(cuda, hop, hip))
    for hip, hop in id_maps['source']['hip'].items():
        cuda = id_maps['target']['cuda'][hop]
        if hip not in reference['hip']:
            warn('No reference mapping for {}'.format(hip))
            continue
        if cuda not in reference['hip'][hip]:
            warn('Incorrect mapping: {} -> {} -> {}'.format(hip, hop, cuda))


def _all_files_in_tree(tree):
    files = {}
    for root in tree:
        label = root.replace('source/', '', 1)
        files.setdefault(label, UniqueList())
        for node in tree[root].values():
            files[label].append(node.name)
            files[label].extend(node)
            logging.debug('node={}'.format(repr(node)))
    logging.debug('_all_files_in_tree > {}'.format(files))
    return files


def _check_id_list(id_list, label, id_maps, reference, warn, wishlist):
    for name in id_list:
        if label == 'hip':
            hip = name
            hop = id_maps['source']['hip'][hip]
            cuda = id_maps['target']['cuda'][hop]
            if cuda not in reference['hip'][hip]:
                warn('Incorrect mapping: {} -> {} -> {}'.format(hip, hop, cuda))
            wishlist['cuda'].append(cuda)
        elif label == 'cuda':
            cuda = name
            hop = id_maps['source']['cuda'][cuda]
            hip = id_maps['target']['hip'][hop]
            if hip != reference['cuda'][cuda]:
                warn('Incorrect mapping: {} -> {} -> {}'.format(cuda, hop, hip))
            wishlist['hip'].append(hip)
        else:
            hop = name
            hip = id_maps['target']['hip'][hop]
            cuda = id_maps['target']['cuda'][hop]
            wishlist['hip'].append(hip)
            wishlist['cuda'].append(cuda)


def _known_identifiers(id_lists):
    known_ids = {
            'hip': [],
            'cuda': [],
            }
    for label in known_ids:
        for filename in id_lists[label]:
            known_ids[label].extend(id_lists[label][filename])
    return known_ids


def check_lists(tree, id_lists, id_maps, reference):
    files = _all_files_in_tree(tree)
    known_ids = _known_identifiers(id_lists)
    wishlist = {
            'hip': [],
            'cuda': [],
            }
    for label in id_lists:
        if label == 'hop':
            root = label
        else:
            root = 'source/' + label
        for filename in id_lists[label]:
            path = file_path(os.path.join(root, filename))
            _check_regular_file(path, warn)
            if not filename in files[label]:
                warn('File [{}] {} missing from the file tree.'.format(
                    root, filename))
            if not reference:
                continue
            _check_id_list(id_lists[label][filename], label, id_maps,
                           reference, warn, wishlist)
    for label in wishlist:
        for tgt in sorted(set(wishlist[label])):
            if tgt not in known_ids[label]:
                warn('Unknown target ID: {}'.format(tgt))


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

    if args.hipify:
        hipify = os.path.join(os.path.expanduser(args.hipify),
                              'bin/hipify-perl')
        if not os.path.exists(hipify):
            raise FileNotFoundError(hipify)
        reference = reference_map(hipify)
    else:
        print('Unable to scrape for reference mappings (cf. --hipify option)')
        reference = None

    check_tree(tree)
    check_lists(tree, id_lists, id_maps, reference)
    check_maps(tree, id_maps, reference)
    print('Warnings: {}'.format(len(warnings)))
    for msg in sorted(warnings):
        print(' ', msg)


if __name__ == '__main__':
    usage = '%(prog)s [options]'
    desc = 'Check consistency of metadata (in ../data/).'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('-i', '--hipify', default=None,
            help='path to HIPIFY git repository / installation')
    parser.add_argument('-d', '--dry-run', action='store_true', default=False,
            help='run without modifying any files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
            help='display additional information while running')
    parser.add_argument('--debug', action='store_true', default=False,
            help='display additional information while running')

    args = parser.parse_args()

    # configure logging
    config = {'format': '[%(levelname)s] %(message)s'}
    if args.debug:
        config['level'] = logging.DEBUG
    logging.basicConfig(**config)
    logging.debug('args={}'.format(args))

    check(args)
