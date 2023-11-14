#!/usr/bin/env python3

import os
import pathlib
import logging

from common.io import read_metadata, write_metadata
from common.parser import ArgumentParser
from common.metadata import update_metadata


def _all_list_ids(metadata, label):
    ids = []
    for filename in metadata['list'][label]:
        ids.extend(metadata['list'][label][filename])
    return ids


def _remove_manual(args, metadata, manual_metadata):
    # file tree
    for root in manual_metadata['tree']:
        for node in manual_metadata['tree'][root]:
            logging.debug('tree node: {}/{}'.format(root, node))
            if node in metadata['tree'][root]:
                if args.verbose:
                    print('Removed node: {}/{}'.format(root, node))
                metadata['tree'][root].pop(node)
    # ID lists
    for label in manual_metadata['list']:
        for filename in manual_metadata['list'][label]:
            logging.debug('list label={} filename={}'.format(label, filename))
            if not filename in metadata['list'][label]:
                continue
            for name in manual_metadata['list'][label][filename]:
                if name in metadata['list'][label][filename]:
                    if args.verbose:
                        print('Removed ID: {}'.format(name))
                    metadata['list'][label][filename].remove(name)
    # ID maps
    for direction in manual_metadata['map']:
        for label in manual_metadata['map'][direction]:
            logging.debug('map direction={} label={}'.format(direction, label))
            for hop in manual_metadata['map'][direction][label]:
                if hop in metadata['map'][direction][label]:
                    if args.verbose:
                        tgt = metadata['map'][direction][label][hop]
                        print('Removed map: {} -> {}'.format(hop, tgt))
                    metadata['map'][direction][label].pop(hop)


def manual(args):
    metadata = read_metadata()
    logging.debug('metadata={}'.format(metadata))
    manual_metadata = read_metadata('data/manual')
    logging.debug('manual_metadata={}'.format(manual_metadata))

    print('Manually edited metadata:')
    for label in manual_metadata['list']:
        ids = _all_list_ids(manual_metadata, label)
        if ids:
            print('  {:9} {}'.format(
                '{} IDs:'.format(label.upper()), len(ids)))
    for direction in ['source', 'target']:
        count = 0
        for label in manual_metadata['map'][direction]:
            count += len(manual_metadata['map'][direction][label])
        print('  {} mappings: {}'.format(direction, count))

    if args.remove:
        print('')
        _remove_manual(args, metadata, manual_metadata)
    else:
        update_metadata(metadata, manual_metadata)
        logging.debug('updated metadata={}'.format(metadata))

    write_metadata(metadata, dry_run=args.dry_run)


if __name__ == '__main__':
    usage = '%(prog)s [options]'
    desc = 'Add/remove manually edited metadata (from data/manual/).'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('-r', '--remove', action='store_true', default=False,
            help='remove manually edited metadata instead of adding it')
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

    manual(args)
