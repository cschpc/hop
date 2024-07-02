#!/usr/bin/env python3

import os
import pathlib
import logging

from common.io import read_metadata, file_path
from common.abc import UniqueList
from common.parser import ArgumentParser
from common.metadata import known_list_ids, make_triplet, translate, VersionedID
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


def check_tree(metadata):
    for root in metadata['tree']:
        for node in metadata['tree'][root].values():
            path = file_path(os.path.join(root, node.name))
            _check_regular_file(path, warn)
            if node.link:
                target = file_path(os.path.join(root, node.link))
                _check_symbolic_link(path, target, warn)
                continue
            for name in node:
                path = file_path(os.path.join(root, name))
                _check_regular_file(path, warn)


def check_maps(metadata, reference):
    if not reference:
        return warnings
    for cuda, hop in metadata['map']['source']['cuda'].items():
        hip = metadata['map']['target']['hip'][hop]
        if cuda not in reference['cuda']:
            warn('No reference mapping for {}'.format(cuda))
            continue
        if hip != reference['cuda'][cuda]:
            warn('Incorrect mapping: {} -> {} -> {}'.format(cuda, hop, hip))
    for hip, hop in metadata['map']['source']['hip'].items():
        cuda = metadata['map']['target']['cuda'][hop]
        if hip not in reference['hip']:
            warn('No reference mapping for {}'.format(hip))
            continue
        if cuda not in reference['hip'][hip]:
            warn('Incorrect mapping: {} -> {} -> {}'.format(hip, hop, cuda))


def _all_files_in_tree(metadata):
    files = {}
    for root in metadata['tree']:
        label = root.replace('source/', '', 1)
        files.setdefault(label, UniqueList())
        for node in metadata['tree'][root].values():
            files[label].append(node.name)
            files[label].extend(node)
            logging.debug('node={}'.format(repr(node)))
    logging.debug('_all_files_in_tree > {}'.format(files))
    return files


def _check_id_list(filename, label, metadata, reference, warn, wishlist):
    for name in metadata['list'][label][filename]:
        hop, hip, cuda = make_triplet(metadata, label, name)
        if label == 'hip':
            if cuda not in reference['hip'].get(hip, []):
                warn('Incorrect mapping: {} -> {} -> {}'.format(hip, hop, cuda))
            ref = reference['hip'].get(hip, [])
            vid = VersionedID(cuda)
            if vid.has_version_suffix():
                for x in vid.supercedes():
                    if x in ref:
                        ref.remove(x)
            if len(ref) > 1 and not translate.is_lib(hop) \
                    and not translate.is_default_cuda(hop, cuda):
                warn('Non-default mapping: {} -> {} -> {} <> {}'.format(
                    hip, hop, cuda, translate.default(hip, 'cuda')))
            wishlist['cuda'].append(cuda)
        elif label == 'cuda':
            if hip != reference['cuda'].get(cuda):
                warn('Incorrect mapping: {} -> {} -> {}'.format(cuda, hop, hip))
            wishlist['hip'].append(hip)
        else:
            wishlist['hip'].append(hip)
            wishlist['cuda'].append(cuda)


def check_lists(metadata, reference):
    files = _all_files_in_tree(metadata)
    wishlist = {
            'hip': [],
            'cuda': [],
            }
    for label in metadata['list']:
        if label == 'hop':
            root = label
        else:
            root = 'source/' + label
        for filename in metadata['list'][label]:
            path = file_path(os.path.join(root, filename))
            _check_regular_file(path, warn)
            if not filename in files[label]:
                warn('File [{}] {} missing from the file tree.'.format(
                    root, filename))
            if not reference:
                continue
            _check_id_list(filename, label, metadata, reference, warn,
                           wishlist)
    known_ids = known_list_ids(metadata)
    for label in wishlist:
        for tgt in sorted(set(wishlist[label])):
            if tgt not in known_ids[label]:
                warn('Unknown target ID: {}'.format(tgt))


def check(args):
    metadata = read_metadata()

    if args.hipify:
        hipify = os.path.join(os.path.expanduser(args.hipify),
                              'bin/hipify-perl')
        if not os.path.exists(hipify):
            raise FileNotFoundError(hipify)
        reference = reference_map(hipify)
    else:
        print('Unable to scrape for reference mappings (cf. --hipify option)')
        reference = None

    check_tree(metadata)
    check_lists(metadata, reference)
    check_maps(metadata, reference)
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
