import re


class Map(dict):
    """Custom dictionary for identifier maps

    Tries to guess correct translation for non-defined identifiers.
    """

    def __init__(self, *args, **kwargs):
        self.label = kwargs.pop('label', '')
        self.source = kwargs.pop('source', False)
        self._prep_regex()
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key not in self:
            return self._guess(key)
        return self[key]

    def _prep_regex(self):
        # match the correct identifier prefix
        if self.label == 'cuda':
            self._regex_src = re.compile('^(cuda|cu)')
        else:
            self._regex_src = re.compile('^hip')
        self._regex_tgt = re.compile('^gpu')

    def _guess(self, key):
         # guess identifier, if key seems otherwise ok
        if self.source:
            if not self._regex_src.match(key):
                self._error(key)
            return self._regex_src.sub('gpu', key)
        else:
            if not self._regex_tgt.match(key):
                self._error(key)
            return self._regex_tgt.sub(self.label, key)

    def _error(self, key):
        kind = 'source' if self.source else 'target'
        raise ValueError(
                "Unable to guess translation for identifier "
                "'{}' in {} {} map".format(key, self.label.upper(), kind))
