class VoseAliasSampler(object):
    """Samples values from a weighted distribution using Vose's algorithm for
    the Alias Method.

    See http://www.keithschwarz.com/darts-dice-coins/ for details.

    """

    def __init__(self, weights, options):
        assert any(weights)
        assert all(w >= 0 for w in weights)

        n = len(weights)

        self.__options = options

        total = sum(weights)

        weights = tuple(float(w) / total for w in weights)

        self._alias = [None] * len(weights)
        self._probabilities = [None] * len(weights)

        self._size = total

        small = []
        large = []

        ps = [w * n for w in weights]

        for i, p in enumerate(ps):
            if p < 1:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            l = small.pop()
            g = large.pop()
            assert ps[g] >= 1 >= ps[l]
            self._probabilities[l] = ps[l]
            self._alias[l] = g
            ps[g] = (ps[l] + ps[g]) - 1
            if ps[g] < 1:
                small.append(g)
            else:
                large.append(g)
        for q in [small, large]:
            while q:
                g = q.pop()
                self._probabilities[g] = 1.0
                self._alias[g] = g

        assert None not in self._alias
        assert None not in self._probabilities

    def sample(self, random):
        i = random.randint(0, len(self._probabilities) - 1)
        p = self._probabilities[i]

        if p >= 1:
            toss = True
        elif p <= 0:
            toss = False
        else:
            toss = random.random() <= p

        if toss:
            return self.__options[i]
        else:
            return self.__options[self._alias[i]]

    def __repr__(self):
        return 'Sampler(%r, %r)' % (
            list(zip(
                range(len(self._probabilities)),
                self._probabilities, self._alias)), self.__options)
