#!/usr/bin/env python3

import os
import re
import logging

from common.parser import ArgumentParser
from common.io import header_name, header_root, lang


def _includes(path):
    regex_include = re.compile('^#include [<"]([^>"]*)[">]')
    includes = []
    for line in open(path):
        if regex_include.match(line):
            includes.append(regex_include.match(line).group(1))
    logging.debug('_includes < {} > {}'.format(path, includes))
    return includes


def _all_filenames(files):
    return [header_name(x) for x in files]


def _collect(root, filename, included, all_filenames):
    logging.debug('_collect < root={} filename={} included={}'.format(
        root, filename, included))
    dirname = os.path.dirname(filename)
    for include in _includes(os.path.join(root, filename)):
        logging.debug('_collect: include={}'.format(include))
        if os.path.exists(os.path.join(root, dirname, include)):
            include = os.path.join(dirname, include)
        elif not os.path.exists(os.path.join(root, include)):
            continue
        logging.debug('_collect: include={}'.format(include))
        if include in included:
            continue
        included.append(include)
        if include not in all_filenames:
            _collect(root, include, included, all_filenames)


def _single(root, filename, expanded, shallow=False, indent=0):
    logging.debug('_single < root={} filename={} expanded={}'.format(
        root, filename, expanded))
    prefix = ' ' * indent
    if indent:
        prefix += '+ '
    print(prefix + filename)
    dirname = os.path.dirname(filename)
    logging.debug('_single: dirname={}'.format(dirname))
    if indent and shallow:
        return
    for include in sorted(_includes(os.path.join(root, filename))):
        logging.debug('_single: include={}'.format(include))
        if os.path.exists(os.path.join(root, dirname, include)):
            include = os.path.join(dirname, include)
        elif not os.path.exists(os.path.join(root, include)):
            continue
        logging.debug('_single: include={}'.format(include))
        if not args.all and include in expanded:
            print(' ' * (indent + 2) + '* ' + include)
        else:
            expanded.append(include)
            _single(root, include, expanded, shallow, indent + 2)


def graph(args):
    expanded = []
    all_filenames = _all_filenames(args.files)
    logging.debug('all_filenames={}'.format(all_filenames))
    for path in args.files:
        print('')
        filename = header_name(path)
        root = header_root(path)
        if args.flatten:
            included = [filename]
            _collect(root, filename, included, all_filenames)
            included.remove(filename)
            print(filename)
            for include in sorted(included):
                print('+', include)
        else:
            _single(root, filename, expanded, args.shallow)
            expanded.append(filename)


if __name__ == '__main__':
    usage = '%(prog)s [options] file.h {file2.h ...}'
    desc = 'Show a graph of included header files.'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('files', nargs='+',
            help='header files to scrape')
    parser.add_argument('-a', '--all', action='store_true', default=False,
            help='full hierarchy repeating already expanded entries')
    parser.add_argument('-f', '--flatten', action='store_true', default=False,
            help='flatten hierarchy to a single level')
    parser.add_argument('-s', '--shallow', action='store_true', default=False,
            help='do not expand entries')
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

    graph(args)
