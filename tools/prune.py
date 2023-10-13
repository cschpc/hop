#!/usr/bin/env python3

import os
import pathlib
import logging

from common.io import read_metadata, write_metadata
from common.parser import ArgumentParser
from common.scrape import obsolete_ids
from common.metadata import known_list_ids, make_triplet, translate


def _map_keys(metadata, direction, label, name):
    keys = []
    for key, value in metadata['map'][direction][label].items():
        if value == name:
            keys.append(key)
    return keys


def _remove_id(metadata, label, name):
    logging.debug('_remove_id < label={}, name={}'.format(label, name))
    touch = False
    # remove from ID lists
    for filename, id_list in metadata['list'][label].items():
        if name in id_list:
            id_list.remove(name)
            touch = True
            logging.debug('_remove_id: list {}'.format(filename))
            break
    # remove from ID maps
    if label == 'hop':
        if metadata['map']['target']['hip'].pop(name, None):
            touch = True
            logging.debug('_remove_id: target map hip {}'.format(name))
        if metadata['map']['target']['cuda'].pop(name, None):
            touch = True
            logging.debug('_remove_id: target map cuda {}'.format(name))
        for src in ['hip', 'cuda']:
            for key in _map_keys(metadata, 'source', src, name):
                if metadata['map']['source'][src].pop(key, None):
                    touch = True
                    logging.debug(
                            '_remove_id: source map {} {}'.format(src, key))
    else:
        if metadata['map']['source'][label].pop(name, None):
            touch = True
            logging.debug('_remove_id: source map {} {}'.format(label, name))
        for key in _map_keys(metadata, 'target', label, name):
            if metadata['map']['target'][label].pop(key, None):
                touch = True
                logging.debug(
                        '_remove_id: target map {} {}'.format(label, key))
    if touch:
        return [name]
    return []


def _remove_obsolete(metadata, label, name):
    logging.debug('_remove_obsolete < label={} name={}'.format(label, name))

    # forward translation
    hop, hip, cuda = make_triplet(metadata, label, name)
    logging.debug('(hop, hip, cuda)=({}, {}, {})'.format(hop, hip, cuda))
    if label == 'hip':
        tgt = cuda
        other = 'cuda'
    else:
        tgt = hip
        other = 'hip'
    logging.debug('hop={}  tgt={}  other={}'.format(hop, tgt, other))

    # backward translation
    hop2, hip2, cuda2 = make_triplet(metadata, other, tgt)
    logging.debug('(hop2, hip2, cuda2)=({}, {}, {})'.format(hop2, hip2, cuda2))

    # count occurrences
    hip_values = list(metadata['map']['source']['hip'].values())
    cuda_values = list(metadata['map']['source']['cuda'].values())
    count_hop = {
            'hip': hip_values.count(hop),
            'cuda': cuda_values.count(hop),
            }
    count_hop2 = {
            'hip': hip_values.count(hop2),
            'cuda': cuda_values.count(hop2),
            }
    logging.debug('count_hop={}'.format(count_hop))
    logging.debug('count_hop2={}'.format(count_hop2))

    removed = []
    remove_id = lambda x, y, z: removed.extend(_remove_id(x, y, z))
    if (hip, cuda) != (hip2, cuda2):
        logging.debug('asymmetric translation')
        remove_id(metadata, label, name)
        # remove HOP ID only if not used anymore
        if hop != hop2 and count_hop[label] == 1 and count_hop[other] == 0:
            remove_id(metadata, 'hop', hop)
    else:
        logging.debug('symmetric translation')
        remove_id(metadata, label, name)
        # remove also translated IDs and their dependencies
        remove_id(metadata, other, tgt)
        remove_id(metadata, 'hop', hop)
        if count_hop[label] > 1:
            for x in _map_keys(metadata, 'source', label, name):
                remove_id(metadata, label, x)
        if count_hop[other] > 1:
            for x in _map_keys(metadata, 'source', other, tgt):
                remove_id(metadata, other, x)
        # remove HOP ID in backward translation only if not used anymore
        if hop != hop2 and count_hop2[label] == 0 and count_hop2[other] == 1:
            remove_id(metadata, 'hop', hop2)
    logging.debug('removed={}'.format(removed))
    return removed


def prune(args):
    metadata = read_metadata()
    known_ids = known_list_ids(metadata, flat=False)
    logging.debug('known_ids={}'.format(known_ids))

    path = os.path.join(os.path.expanduser(args.hipify), 'bin/hipify-perl')
    obsolete = obsolete_ids(path, args.cuda_version)
    logging.debug('obsolete={}'.format(obsolete))

    if args.verbose:
        print('Obsolete IDs:')
    removed = []
    for name in obsolete:
        logging.debug('obsolete ID {}'.format(name))
        label = translate.lang(name)
        if not label:
            continue
        removed.extend(_remove_obsolete(metadata, label, name))
    if removed and args.verbose:
        print('')
        print('Removed IDs:')
        for name in sorted(removed):
            print('  ' + name)

    print('')
    print('Obsolete IDs: {}'.format(len(obsolete)))
    print('Removed IDs:  {}'.format(len(removed)))

    write_metadata(metadata)


if __name__ == '__main__':
    usage = '%(prog)s [options]'
    desc = 'Prune obsolete IDs and mappings from metadata.'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('hipify',
            help='path to HIPIFY git repository / installation')
    parser.add_argument('--cuda-version', default=None,
            help='version of CUDA to use for obsolete IDs')
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

    prune(args)
