#!/usr/bin/env python3

import os
import copy
import logging

from common.io import read_tree, read_map, read_list, write_map, write_list
from common.parser import ArgumentParser
from common.map import translate
from common.scrape import scrape_hipify, scrape_header


def update_maps(args, id_maps, triplets, known_ids):
    if args.verbose:
        print('Update translation maps:')
    count = 0
    for hop, hip, cuda in triplets:
        # cuda -> hip translation
        _hop = hop
        if cuda in id_maps['source']['cuda']:
            _hop = id_maps['source']['cuda'][cuda]
        elif cuda in known_ids and not translate.is_default_cuda(_hop, cuda):
            id_maps['source']['cuda'][cuda] = _hop
            count += 1
            if args.verbose:
                print('  New mapping: {} -> {}'.format(cuda, _hop))
        if hip in known_ids and _hop not in id_maps['target']['hip']:
            if not translate.is_default_hip(_hop, hip):
                id_maps['target']['hip'][_hop] = hip
                count += 1
                if args.verbose:
                    print('  New mapping: {} -> {}'.format(_hop, hip))
        # hip -> cuda translation
        _hop = hop
        if hip in id_maps['source']['hip']:
            _hop = id_maps['source']['hip'][hip]
        elif hip in known_ids and not translate.is_default_hip(_hop, hip):
            id_maps['source']['hip'][hip] = _hop
            count += 1
            if args.verbose:
                print('  New mapping: {} -> {}'.format(hip, _hop))
        if cuda in known_ids and _hop not in id_maps['target']['cuda']:
            if not translate.is_default_cuda(_hop, cuda):
                id_maps['target']['cuda'][_hop] = cuda
                count += 1
                if args.verbose:
                    print('  New mapping: {} -> {}'.format(_hop, cuda))
    if args.verbose:
        print('  New mapping chains:', count)
    return count


def _all_identifiers(id_lists):
    ids = []
    for label in id_lists:
        for filename in id_lists[label]:
            ids.extend(id_lists[label][filename])
    return ids


def _known_triplets(triplets, known_ids):
    known = []
    for hop, hip, cuda in triplets:
        if hop in known_ids or hip in known_ids or cuda in known_ids:
            known.append((hop, hip, cuda))
    return known


def scrape(args):
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
    known_ids = _all_identifiers(id_lists)
    logging.debug('tree={}'.format(tree))
    logging.debug('id_maps={}'.format(id_maps))
    logging.debug('id_lists={}'.format(id_lists))

    orig_maps = copy.deepcopy(id_maps)
    orig_lists = copy.deepcopy(id_lists)

    path = os.path.join(os.path.expanduser(args.hipify),
                        'bin/hipify-perl')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    triplets = scrape_hipify(path, args.verbose, args.include_experimental,
                             args.exclude, args.exclude_group)
    logging.debug('triplets={}'.format(triplets))

    count = {
            'old': 0,
            'new': 0,
            'move': 0,
            }
    for path in args.files:
        basename = os.path.basename(path)
        if basename.endswith('.h') or basename.endswith('.hpp'):
            scrape_header(args, os.path.expanduser(path), tree, id_maps,
                          id_lists, known_ids, triplets, count)
        else:
            print('Unable to scrape: {}'.format(path))
    triplets = _known_triplets(triplets, known_ids)
    count['map'] = update_maps(args, id_maps, triplets, known_ids)
    print('')
    print('Old identifiers:    {}'.format(count['old']))
    print('New identifiers:    {}'.format(count['new']))
    print('Moved identifiers:  {}'.format(count['move']))
    print('New mapping chains: {}'.format(count['map']))

    logging.debug('id_maps={}'.format(id_maps))
    logging.debug('id_lists={}'.format(id_lists))
    todo = []
    if not id_maps['source'] == orig_maps['source']:
        todo.append('data/source.map')
    if not id_maps['target'] == orig_maps['target']:
        todo.append('data/target.map')
    if not id_lists['hop'] == orig_lists['hop']:
        todo.append('data/hop.list')
    if not id_lists['hip'] == orig_lists['hip']:
        todo.append('data/hip.list')
    if not id_lists['cuda'] == orig_lists['cuda']:
        todo.append('data/cuda.list')
    if not todo:
        return
    print('')
    print('Updated metadata:')
    print('  ' + '\n  '.join(todo))
    if args.force or input('Overwrite file(s)? [Y/n] ') in ['y', 'yes', '']:
        for filename in todo:
            base, ext = os.path.splitext(filename)
            label = os.path.basename(base)
            if ext == '.map':
                source = True if label == 'source' else False
                if not args.dry_run:
                    write_map(filename, id_maps[label], source, force=True)
            elif ext == '.list':
                if not args.dry_run:
                    write_list(filename, id_lists[label], force=True)
            else:
                raise ValueError('Unknown file type: {}'.format(filename))


if __name__ == '__main__':
    usage = '%(prog)s [options] hipify file.h {file2.h ...}'
    desc = 'Scrape files to suggest new identifiers and mappings.'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('hipify',
            help='path to HIPIFY git repository / installation')
    parser.add_argument('files', nargs='+',
            help='header files to scrape')
    parser.add_argument('-p', '--cpp-macros',
            action='store_true', default=False,
            help='preprocess for C++ with CUDA/HIP macros')
    parser.add_argument('-e', '--include-experimental',
            action='store_true', default=False,
            help='(hipify) include experimental substitutions')
    parser.add_argument('-x', '--exclude', action='append', default=[],
            help='(hipify) exclude identifiers with this prefix')
    parser.add_argument('-g', '--exclude-group', action='append', default=[],
            help='(hipify) exclude substitution group (library, ...)')
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
