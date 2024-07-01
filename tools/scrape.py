#!/usr/bin/env python3

import os
import copy
import logging

from common.io import read_metadata, write_metadata
from common.parser import ArgumentParser
from common.scrape import scrape_header, scrape_hipify
from common.metadata import known_list_ids, translate


def update_maps(args, metadata, triplets, known_ids):
    if args.verbose:
        print('Update translation maps:')
    count = 0
    for hop, hip, cuda in triplets:
        # cuda -> hip translation
        _hop = hop
        if cuda in metadata['map']['source']['cuda']:
            _hop = metadata['map']['source']['cuda'][cuda]
        elif cuda in known_ids['cuda']:
            metadata['map']['source']['cuda'][cuda] = _hop
            count += 1
            if args.verbose:
                print('  New mapping: {} -> {}'.format(cuda, _hop))
        if hip in known_ids['hip']:
            if (_hop not in metadata['map']['target']['hip']
                    or translate.is_default_hip(_hop, hip)):
                metadata['map']['target']['hip'][_hop] = hip
                count += 1
                if args.verbose:
                    print('  New mapping: {} -> {}'.format(_hop, hip))
        # hip -> cuda translation
        _hop = hop
        if hip in metadata['map']['source']['hip']:
            _hop = metadata['map']['source']['hip'][hip]
        elif hip in known_ids['hip']:
            metadata['map']['source']['hip'][hip] = _hop
            count += 1
            if args.verbose:
                print('  New mapping: {} -> {}'.format(hip, _hop))
        if cuda in known_ids['cuda']:
            if (_hop not in metadata['map']['target']['cuda']
                    or translate.is_default_cuda(_hop, cuda)):
                metadata['map']['target']['cuda'][_hop] = cuda
                count += 1
                if args.verbose:
                    print('  New mapping: {} -> {}'.format(_hop, cuda))
    logging.debug('updated metadata={}'.format(metadata))
    if args.verbose:
        print('  New mapping chains:', count)
    return count


def _known_triplets(triplets, known_ids):
    known = []
    for hop, hip, cuda in triplets:
        if (hop in known_ids['hop']
                or hip in known_ids['hip']
                or cuda in known_ids['cuda']):
            known.append((hop, hip, cuda))
    return known


def scrape(args):
    metadata = read_metadata()
    known_ids = known_list_ids(metadata)

    path = os.path.join(os.path.expanduser(args.hipify),
                        'bin/hipify-perl')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    triplets = scrape_hipify(path,
                             cuda_version=args.cuda_version,
                             exclude=args.exclude,
                             exclude_group=args.exclude_group,
                             include_experimental=args.include_experimental,
                             verbose=args.verbose)
    logging.debug('triplets={}'.format(triplets))

    count = {
            'old': 0,
            'new': 0,
            'move': 0,
            }
    for path in args.files:
        basename = os.path.basename(path)
        if basename.endswith('.h') or basename.endswith('.hpp'):
            scrape_header(args, os.path.expanduser(path), metadata, known_ids,
                          triplets, count)
        else:
            print('Unable to scrape: {}'.format(path))
    triplets = _known_triplets(triplets, known_ids)
    count['map'] = update_maps(args, metadata, triplets, known_ids)
    print('')
    print('Old identifiers:    {}'.format(count['old']))
    print('New identifiers:    {}'.format(count['new']))
    print('Moved identifiers:  {}'.format(count['move']))
    print('New mapping chains: {}'.format(count['map']))

    write_metadata(metadata, force=args.force, dry_run=args.dry_run)


if __name__ == '__main__':
    usage = '%(prog)s [options] hipify file.h {file2.h ...}'
    desc = 'Scrape files to suggest new identifiers and mappings.'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('hipify',
            help='path to HIPIFY git repository / installation')
    parser.add_argument('files', nargs='+',
            help='header files to scrape')
    parser.add_argument('-r', '--no-preprocess',
            action='store_true', default=False,
            help='do not preprocess headers')
    parser.add_argument('-p', '--cpp-macros',
            action='store_true', default=False,
            help='preprocess for C++ with CUDA/HIP macros')
    parser.add_argument('-a', '--expand-macros',
            action='store_true', default=False,
            help='expand macros when preprocessing')
    parser.add_argument('-e', '--include-experimental',
            action='store_true', default=False,
            help='(hipify) include experimental substitutions')
    parser.add_argument('-x', '--exclude', action='append', default=[],
            help='(hipify) exclude identifiers with this prefix')
    parser.add_argument('-g', '--exclude-group', action='append', default=[],
            help='(hipify) exclude substitution group (library, ...)')
    parser.add_argument('-m', '--ignore-moved',
            action='store_true', default=False,
            help='ignore moved identifiers')
    parser.add_argument('--cuda-version', default=None,
            help='version of CUDA to use for obsolete IDs')
    parser.add_argument('--force', action='store_true', default=False,
            help='force overwriting of existing files')
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

    scrape(args)
