import sympy
from sympy.matrices.sparse import SparseMatrix
from aliasmethod import VoseAliasSampler
from random import Random
from regex import build_dfa


def compute_generating_functions(accepting, transitions):
    """Uses sympy to calculate the counting generating function for all states
    in a DFA simultaneously.

    The idea is that for any DFA the the generating function of the language
    matched starting from any state is a linear function can be expressed as
    sum(generating_function(transitions[i][c]) for c in ALPHABET), with an
    additional +1 if i is an accepting state.

    This gives us a linear system of equations (over the field of rational
    functions in one variable) which we can just ask sympy to solve for us.

    After that we now know the generating functions for every state in the DFA.
    """

    assert len(accepting) == len(transitions)

    n = len(accepting)

    z = sympy.Symbol('z', real=True)

    weights = {}
    for i in range(n):
        weights[(i, i)] = 1
        for _, j in transitions[i].items():
            key = (i, j)
            weights[key] = weights.get(key, 0) - z
    matrix = SparseMatrix(
        n, n, weights
    )

    vector = sympy.Matrix(n, 1, list(map(int, accepting)))

    return z, matrix.LUsolve(vector)


class ParamTooLarge(ValueError):
    pass


class Simulator(object):
    """Automatically converts any regular expression into a family of
    Boltzmann samplers."""

    def __init__(self, regex, seed=None):
        self.__random = Random(seed)
        self.__dfa = build_dfa(regex)
        symbol, gfs = (
            compute_generating_functions(*self.__dfa))

        # Converting the generating functions to lambdas significantly speeds
        # up drawing. I'm not totally sure why? I suspect it might be partly
        # not going through arbitrary precision decimals, so this probably
        # comes at some cost to numeric stability.
        self.__generating_functions = [
            sympy.lambdify(symbol, g) for g in gfs
        ]

        f = gfs[0]
        size = symbol * sympy.diff(f) / f
        self.expected_size = sympy.lambdify(
            symbol, size
        )

        self.__symbol = symbol
        self.__expected_size = size

    def draw(self, parameter, max_size=None):
        """Repeatedly draw from the Boltzmann sampler of the specified parameter
        for this language and yield the results. Where max_size is specified,
        will draw from the conditional distribution of the Boltzmann sampler
        where the length is <= max_size.
        """

        assert 0 <= parameter <= 1

        state_weights = [
            g(parameter)
            for g in self.__generating_functions
        ]

        # The generating function is a sum of positive terms, so if we got a
        # negative answer then that must be a sign that those terms failed to
        # converge. Note that we are not guaranteed that the sum converges even
        # if all of these happen to be >= 0.
        if any(w <= 0 for w in state_weights):
            raise ParamTooLarge("Parameter %r too large" % (parameter,))

        samplers = []
        for terminal, transitions in zip(*self.__dfa):
            options = []
            weights = []
            if terminal:
                options.append(None)
                weights.append(1)
            for i, state in sorted(transitions.items()):
                options.append(i)
                weights.append(parameter * state_weights[state])
            assert len(options) == len(weights)
            samplers.append(VoseAliasSampler(weights, options))

        while True:
            result = bytearray()
            state = 0
            while True:
                choice = samplers[state].sample(self.__random)
                if choice is None:
                    yield bytes(result)
                    break
                result.append(choice)
                state = self.__dfa[1][state][choice]

                if max_size is not None and len(result) > max_size:
                    break
