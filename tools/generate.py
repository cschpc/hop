import argparse

from common.headers import make_headers
from common.io import read_tree, read_map, read_list, write_header


def generate(args):
    tree = read_tree('../data/file.tree')
    id_maps = {
            'source': read_map('../data/source.map', source=True),
            'target': read_map('../data/target.map'),
            }
    id_lists = {
            'hop': read_list('../data/hop.list'),
            'hip': read_list('../data/hip.list'),
            'cuda': read_list('../data/cuda.list'),
            }

    headers = make_headers(tree, id_maps, id_lists)
    for path, content in headers.items():
        if args.verbose:
            print('Writing header: {}'.format(path))
        if not args.dry_run:
            write_header(path, content, args.force)


if __name__ == '__main__':
    usage = '%(prog)s [options]'
    desc = 'Generate header files based on metadata (in ../data/).'
    parser = argparse.ArgumentParser(usage=usage, description=desc)
    parser.add_argument('-d', '--dry-run', action='store_true', default=False,
            help='run without modifying any files')
    parser.add_argument('-f', '--force', action='store_true', default=False,
            help='force overwriting of existing files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
            help='display additional information while running')

    args = parser.parse_args()

    generate(args)
