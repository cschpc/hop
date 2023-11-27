#!/usr/bin/env python3

import logging

from common.headers import make_headers
from common.io import read_metadata, write_header
from common.parser import ArgumentParser


def generate(args):
    metadata = read_metadata()

    headers = make_headers(metadata)
    if args.verbose:
        print('Writing headers:')
        for path, content in headers.items():
            print('  {}'.format(path))
        print('')
    for path, content in headers.items():
        logging.debug('path={}'.format(path))
        logging.debug('content={}'.format(content))
        if not args.dry_run:
            write_header(path, content, args.force)


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
