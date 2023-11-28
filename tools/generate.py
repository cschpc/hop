#!/usr/bin/env python3

import os
import logging

from common.headers import make_headers
from common.io import (create_link, file_path, read_metadata, root_path,
                       write_header)
from common.parser import ArgumentParser


def _make_links(metadata):
    links = {}
    for label in metadata['tree']:
        if label == 'hop':
            path = file_path(label)
        else:
            path = file_path(os.path.join('hop', label))
        for node in metadata['tree'][label].values():
            logging.debug('node={}'.format(repr(node)))
            if not node.link:
                continue
            links.setdefault(path, [])
            links[path].append((node.name, node.link))
    return links


def generate(args):
    metadata = read_metadata()

    headers = make_headers(metadata)
    links = _make_links(metadata)
    if args.verbose:
        print('Working directory:')
        print('  {}'.format(root_path()))
        print('')
        print('Writing headers:')
        for path, content in headers.items():
            print('  {}'.format(path))
        print('')
        print('Creating links:')
        for path in links:
            _path = os.path.relpath(path, root_path())
            for src, tgt in links[path]:
                print('  {} -> {}'.format(os.path.join(_path, src),
                                          os.path.join(_path, tgt)))
        print('')
    for path, content in headers.items():
        logging.debug('path={}'.format(path))
        logging.debug('content={}'.format(content))
        if not args.dry_run:
            write_header(path, content, args.force)
    for path in links:
        for src, tgt in links[path]:
            if not args.dry_run:
                create_link(path, src, tgt, args.force)


if __name__ == '__main__':
    usage = '%(prog)s [options]'
    desc = 'Generate header files based on metadata (in ../data/).'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('-d', '--dry-run', action='store_true', default=False,
            help='run without modifying any files')
    parser.add_argument('-f', '--force', action='store_true', default=False,
            help='force overwriting of existing files')
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

    generate(args)
