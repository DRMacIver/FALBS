class UnionFind(object):
    def __init__(self):
        self.table = {}

    def find(self, value):
        try:
            if self.table[value] == value:
                return value
        except KeyError:
            self.table[value] = value
            return value

        trail = []
        while value != self.table[value]:
            trail.append(value)
            value = self.table[value]
        for t in trail:
            self.table[t] = value
        return value

    def merge(self, left, right):
        left = self.find(left)
        right = self.find(right)
        self.table[right] = left

    def __repr__(self):
        classes = {}
        for k in self.table:
            trail = [k]
            v = k
            while self.table[v] != v:
                v = self.table[v]
                trail.append(v)
            classes.setdefault(v, set()).update(trail)
        return "UnionFind(%r)" % (
            sorted(
                classes.values(),
                key=lambda x: (len(x), sorted(map(repr, x)))))
