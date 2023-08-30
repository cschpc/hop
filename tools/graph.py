#!/usr/bin/env python3

import os
import re
import logging

from common.parser import ArgumentParser
from common.io import lang


def _includes(path):
    regex_include = re.compile('^#include [<"]([^>"]*)[">]')
    includes = []
    for line in open(path):
        if regex_include.match(line):
            includes.append(regex_include.match(line).group(1))
    return includes


def _tree_includes(tree, label, parent):
    if label != 'hop':
        label = 'source/' + label
    return tree[label].get(parent, [])


def _included_ids(path, tree, id_lists):
    label = lang(path).lower()
    filename = _filename(path)
    ids = id_lists[label].get(_filename(path), [])
    for include in _includes(path):
        logging.debug('{} includes {}'.format(filename, include))
        ids.extend(id_lists[label].get(include, []))
    for include in _tree_includes(tree, label, filename):
        logging.debug('{} tree includes {}'.format(filename, include))
        ids.extend(id_lists[label].get(include, []))
    return ids


def _filename(path):
    dirname, filename = os.path.split(path)
    if lang(path) == 'HIP':
        subname = os.path.basename(dirname)
        filename = os.path.join(subname, filename)
    return filename


def _all_filenames(files):
    return [_filename(x) for x in files]


def _root(path):
    dirname, filename = os.path.split(path)
    if lang(path) == 'HIP':
        dirname = os.path.dirname(dirname)
    return dirname



def _collect(root, filename, included, all_filenames):
    for include in _includes(os.path.join(root, filename)):
        if include not in included \
                and os.path.exists(os.path.join(root, include)):
            included.append(include)
            if include not in all_filenames:
                _collect(root, include, included, all_filenames)


def _single(root, filename, expanded, indent=0):
    prefix = ' ' * indent
    if indent:
        prefix += '+ '
    print(prefix + filename)
    for include in sorted(_includes(os.path.join(root, filename))):
        if os.path.exists(os.path.join(root, include)):
            if not args.all and include in expanded:
                print(' ' * (indent + 2) + '* ' + include)
            else:
                _single(root, include, expanded, indent + 2)
                expanded.append(include)


def graph(args):
    expanded = []
    all_filenames = _all_filenames(args.files)
    for path in args.files:
        print('')
        filename = _filename(path)
        root = _root(path)
        if args.flatten:
            included = [filename]
            _collect(root, filename, included, all_filenames)
            included.remove(filename)
            print(filename)
            for include in sorted(included):
                print('+', include)
        else:
            _single(root, filename, expanded)
            expanded.append(filename)


if __name__ == '__main__':
    usage = '%(prog)s [options] hipify file.h {file2.h ...}'
    desc = 'Show a graph of included header files.'
    parser = ArgumentParser(usage=usage, description=desc)
    parser.add_argument('files', nargs='+',
            help='header files to scrape')
    parser.add_argument('-a', '--all', action='store_true', default=False,
            help='full hierarchy repeating already expanded entries')
    parser.add_argument('-f', '--flatten', action='store_true', default=False,
            help='flatten hierarchy to a single level')
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
