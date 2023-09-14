import re
import collections


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

    def _guess(self, key):
         # guess identifier, if key seems otherwise ok
        if not self.translator.match(key):
            self._error(key)
        if self.source:
            return self.translator.default(key, 'gpu')
        else:
            return self.translator.default(key, self.label)

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

    def _translate(self, name, target, default=False):
        if default:
            _regex_lower = self.regex_default_lower
            _regex_camel = self.regex_default_camel
            _regex_upper = self.regex_default_upper
        else:
            _regex_lower = self.regex_lower
            _regex_camel = self.regex_camel
            _regex_upper = self.regex_upper
        if _regex_lower.match(name):
            return _regex_lower.sub(r'\1' + target, name)
        if _regex_camel.match(name):
            return _regex_camel.sub(r'\1' + target.capitalize(), name)
        if _regex_upper.match(name):
            return _regex_upper.sub(r'\1' + target.upper(), name)
        return name

    def translate(self, name, target):
        return self._translate(name, target)

    def to_hop(self, name, default=False):
        return self.translate(name, 'gpu')

    def to_hip(self, name, default=False):
        return self.translate(name, 'hip')

    def to_cuda(self, name, default=False):
        return self.translate(name, 'cuda')

    def default(self, name, target):
        return self._translate(name, target, True)

    def is_default_hip(self, hop, hip):
        return hip == self.default(hop, 'hip')

    def is_default_cuda(self, hop, cuda):
        return cuda == self.default(hop, 'cuda')

    def match(self, name, default=False):
        if default:
            _regex_lower = self.regex_default_lower
            _regex_camel = self.regex_default_camel
            _regex_upper = self.regex_default_upper
        else:
            _regex_lower = self.regex_lower
            _regex_camel = self.regex_camel
            _regex_upper = self.regex_upper
        return (_regex_lower.match(name) or _regex_camel.match(name)
                or _regex_upper.match(name))


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