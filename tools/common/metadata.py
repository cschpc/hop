import re
import logging
import functools
import collections


def log(f):
    def _format(f, args, kwargs):
        if f.__name__ != f.__qualname__:
            args = args[1:]
        _args = [repr(x) for x in args] + \
                ['{}={}'.format(x, repr(y)) for x,y in kwargs.items()]
        return ', '.join(_args)
    @functools.wraps(f)
    def logger(*args, **kwargs):
        logging.debug('{} < {}'.format(f.__qualname__,
                                       _format(f, args, kwargs)))
        out = f(*args, **kwargs)
        logging.debug('{} > {}'.format(f.__qualname__, repr(out)))
        return out
    return logger


def known_list_ids(metadata):
    ids = {}
    for label in metadata['list']:
        ids.setdefault(label, [])
        for filename in metadata['list'][label]:
            ids[label].extend(metadata['list'][label][filename])
    return ids


def known_map_ids(metadata):
    ids = []
    for direction in metadata['map']:
        for label in metadata['map'][direction]:
            ids.extend(metadata['map'][direction][label].keys())
            ids.extend(metadata['map'][direction][label].values())
    return ids


class Map(collections.UserDict):
    """Custom dictionary for identifier maps

    Tries to guess correct translation for non-defined identifiers.
    """

    def __init__(self, *args, **kwargs):
        self.label = kwargs.pop('label', '')
        self.source = kwargs.pop('source', False)
        self.translator = Translator()
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key not in self.data:
            return self._guess(key)
        return self.data[key]

    @log
    def _guess(self, key):
        # guess identifier, if key seems otherwise ok
        if not self.translator.match(key):
            self._error(key)
        if self.source:
            return self.translator.to_hop(key)
        else:
            return self.translator.translate(key, self.label)

    def _error(self, key):
        kind = 'source' if self.source else 'target'
        raise ValueError(
                "Unable to guess translation for identifier "
                "'{}' in {} {} map".format(key, self.label.upper(), kind))


class Translator:
    regex_lower = re.compile('^(.*?)(gpu|hip|cuda|cu)')
    regex_camel = re.compile('^(.*?)(Gpu|Hip|Cuda|Cu)')
    regex_upper = re.compile('^(.*?)(GPU|HIP|CUDA|CU)')
    regex_default_lower = re.compile('^()(gpu|hip|cuda)')
    regex_default_camel = re.compile('^()(Gpu|Hip|Cuda)')
    regex_default_upper = re.compile('^()(GPU|HIP|CUDA)')
    regex_lib_lower = re.compile('^()(gpu|hip|cuda|cu)(blas|fft|rand|sparse)')
    regex_lib_upper = re.compile('^()(GPU|HIP|cuda|cu)(BLAS|FFT|RAND|SPARSE)')
    regex_rtc_lower = re.compile('^()(hip|nv)(rtc)')
    regex_rtc_upper = re.compile('^()(HIP|NV)(RTC)')

    def _translate(self, name, target, default=False):
        if default:
            _regex_lower = self.regex_default_lower
            _regex_camel = self.regex_default_camel
            _regex_upper = self.regex_default_upper
        else:
            _regex_lower = self.regex_lower
            _regex_camel = self.regex_camel
            _regex_upper = self.regex_upper
        if self.regex_rtc_lower.match(name):
            return self.regex_rtc_lower.sub(r'\1{}\3'.format(target), name)
        if self.regex_rtc_upper.match(name):
            return self.regex_rtc_upper.sub(
                    r'\1{}\3'.format(target.upper()), name)
        if _regex_lower.match(name):
            return _regex_lower.sub(r'\1' + target, name)
        if _regex_camel.match(name):
            return _regex_camel.sub(r'\1' + target.capitalize(), name)
        if _regex_upper.match(name):
            return _regex_upper.sub(r'\1' + target.upper(), name)
        return name

    def translate(self, name, target, default=False):
        if target == 'cuda' and self.is_lib(name):
            return self._translate(name, 'cu', default=default)
        return self._translate(name, target, default=default)

    def to_hop(self, name, default=False):
        return self.translate(name, 'gpu', default=default)

    def to_hip(self, name, default=False):
        return self.translate(name, 'hip', default=default)

    def to_cuda(self, name, default=False):
        return self.translate(name, 'cuda', default=default)

    def default(self, name, target):
        return self.translate(name, target, True)

    def is_default_hip(self, hop, hip):
        return hip == self.default(hop, 'hip')

    def is_default_cuda(self, hop, cuda):
        return cuda == self.default(hop, 'cuda')

    def is_lib(self, name):
        return (self.regex_lib_lower.match(name)
                or self.regex_lib_upper.match(name))

    def match(self, name, default=False):
        if default:
            _regex_lower = self.regex_default_lower
            _regex_camel = self.regex_default_camel
            _regex_upper = self.regex_default_upper
        else:
            _regex_lower = self.regex_lower
            _regex_camel = self.regex_camel
            _regex_upper = self.regex_upper
        return (_regex_lower.match(name)
                or _regex_camel.match(name)
                or _regex_upper.match(name)
                or self.regex_rtc_lower.match(name)
                or self.regex_rtc_upper.match(name))


translate = Translator()


class Node(list):

    def __init__(self, include=[], name='', link=None):
        self.name = str(name)
        self.link = link
        super().__init__(include)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Node({}, name={}, link={})".format(list(self),
                                                   repr(self.name),
                                                   repr(self.link))


class Include(str):
    """Filename of a header file to be included"""

    def __repr__(self):
        return "Include('{}')".format(self)


class Embed(str):
    """Filename of a header file to be embedded"""

    def __repr__(self):
        return "Embed('{}')".format(self)


class Special(str):
    """Filename of a header file to be handled only by a custom template

    No IDs will be embedded nor include statements added to the content.
    Enables one to implement special treatment in a custom template, but
    any IDs contained will still be considered included.
    """

    def __repr__(self):
        return "Special('{}')".format(self)
