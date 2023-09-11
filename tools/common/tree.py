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
