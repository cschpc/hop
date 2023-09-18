#!/usr/bin/env python3

import os
import re
import logging

from common.parser import ArgumentParser
from common.io import read_metadata


def _find_header(id_lists, name):
    for header, ids in id_lists.items():
        if name in ids:
            return header
    return None


def _connections(metadata, source, target):
    logging.debug('_connections < source={}, target={}'.format(source, target))
    connect = {}
    for header, ids in metadata['list'][source].items():
        connect.setdefault(header, {})
        for name in ids:
            hop = metadata['map']['source'][source][name]
            tgt = metadata['map']['target'][target][hop]
            pair = _find_header(metadata['list'][target], tgt)
            if not pair:
                continue
            connect[header].setdefault(pair, [])
            connect[header][pair].append((name, tgt))
    logging.debug('_connections > {}'.format(connect))
    return connect


def _pretty_print(source, targets):
    print('  {:24} ∈ {}'.format(source, targets[0]))
    for tgt in targets[1:]:
        print('  {:24}   {}'.format('', tgt))


def overlap(args):
    metadata = read_metadata()
    connects = {
            'cuda': _connections(metadata, 'cuda', 'hip'),
            'hip': _connections(metadata, 'hip', 'cuda'),
            }

    # completely equivalent headers
    equals = []
    for header, pairs in connects['cuda'].items():
        logging.debug('pairs={}'.format(pairs))
        if not pairs:
            continue
        pair = list(pairs.keys())[0]
        if not pair:
            continue
        logging.debug('header={}  pair={}'.format(header, pair))
        pair_connects = list(connects['hip'][pair].keys())
        if (len(pairs) == 1 and len(pair_connects) == 1
                and pair_connects[0] == header):
            equals.append((header, pair))
    logging.debug('equals={}'.format(equals))
    if equals:
        print('EQUIVALENT')
        for cuda, hip in equals:
            connects['cuda'].pop(cuda)
            connects['hip'].pop(hip)
            print('  {:24} = {}'.format(cuda, hip))
        print('')

    for label in connects:
        print(label.upper())
        for source in connects[label]:
            if not connects[label][source]:
                print('  {:24}   ???'.format(source))
                continue
            targets = []
            for tgt, ids in connects[label][source].items():
                targets.append('{} ({})'.format(tgt, len(ids)))
            _pretty_print(source, targets)
        print('')


if __name__ == '__main__':
    usage = '%(prog)s [options] file.h {file2.h ...}'
    desc = 'Show overlap of IDs between source and target headers.'
    parser = ArgumentParser(usage=usage, description=desc)
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

    overlap(args)
