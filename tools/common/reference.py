import logging

from common.scrape import scrape_hipify


def reference_map(path):
    triplets = scrape_hipify(path)
    reference = {
            'hip': {},
            'cuda': {},
            }
    for hop, hip, cuda in triplets:
        reference['cuda'][cuda] = hip
        reference['hip'].setdefault(hip, [])
        reference['hip'][hip].append(cuda)
    logging.debug('reference={}'.format(reference))
    return reference
