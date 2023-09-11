class Node(list):
    def __init__(self, name, include=[], link=None):
        super().__init__(include)
        self.name = str(name)
        self.link = link
    def __str__(self):
        return self.name
    def __repr__(self):
        return 'Node({}, {}, {})'.format(self.name, list(self), self.link)
