from collections import UserList


class UniqueList(UserList):
    """Custom list that contains only unique items

    Silently ignores any duplicates while appending/extending.
    """

    def __init__(self, init=[]):
        super().__init__()
        self.extend(init)

    def append(self, x):
        if x not in self:
            self.data.append(x)

    def extend(self, data):
        for x in data:
            self.append(x)
