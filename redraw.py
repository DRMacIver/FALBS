import functools
from collections import deque
import sympy
from sympy.matrices.sparse import SparseMatrix
from random import Random


ALL_CACHES = []


def clear_caches():
    for s in ALL_CACHES:
        s.clear()


def cached(function):
    cache = {}
    ALL_CACHES.append(cache)

    @functools.wraps(function)
    def accept(*args):
        try:
            return cache[args]
        except KeyError:
            pass
        result = function(*args)
        cache[args] = result
        if isinstance(result, Regex):
            result.matches_empty
        return result
    return accept

ALPHABET = range(256)


class Regex(object):
    _cached_matches_empty = None

    @property
    def matches_empty(self):
        if self._cached_matches_empty is None:
            self._cached_matches_empty = self._calculate_matches_empty()
        return self._cached_matches_empty

    def __or__(self, other):
        return union(self, other)

    def __and__(self, other):
        return intersection(self, other)

    def __add__(self, other):
        return concatenate(self, other)

    def __invert__(self):
        return negate(self)

    def __sub__(self, other):
        return subtract(self, other)

    def __lt__(self, other):
        # provide a consistent < ordering across all Regex. Only really useful
        # for normalizing some expressions.
        if not isinstance(other, Regex):
            return NotImplemented
        if type(self) != type(other):
            if isinstance(other, BinaryOp) and not isinstance(self, BinaryOp):
                return True
            if isinstance(self, BinaryOp) and not isinstance(other, BinaryOp):
                return False
            return type(self).__name__ < type(other).__name__
        return self._do_le(other)


class Special(Regex):
    def __init__(self, name, matches_empty):
        self.name = name
        self._cached_matches_empty = matches_empty

    def __repr__(self):
        return self.name

    def _do_le(self, other):
        return self.name < other.name

Epsilon = Special("Epsilon", True)
Omega = Special("Omega", True)
Empty = Special("Empty", False)
Dot = Special("Dot", False)


class Character(Regex):
    _cached_matches_empty = False

    def __init__(self, c):
        self.character = c

    def __repr__(self):
        return "char(%r)" % (self.character,)

    def _do_le(self, other):
        return self.character < other.character


@cached
def char(c):
    if isinstance(c, bytes):
        assert len(c) == 1
        c = c[0]
    assert isinstance(c, int)
    return Character(c)


class UnaryOp(Regex):
    def __init__(self, child):
        self.child = child

    def _do_le(self, other):
        return self.child < other.child


class Star(UnaryOp):
    def __repr__(self):
        return "star(%r)" % (self.child,)

    _cached_matches_empty = True


@cached
def star(child):
    if child in (Epsilon, Empty):
        return Epsilon
    if child is Omega:
        return child
    if isinstance(child, Star):
        return child
    if isinstance(child, Plus):
        return star(child.child)
    return Star(child)


class Plus(UnaryOp):
    def __repr__(self):
        return "plus(%r)" % (self.child,)

    def _calculate_matches_empty(self):
        return self.child.matches_empty


@cached
def plus(regex):
    if regex is Empty:
        return Empty
    if regex in (Epsilon, Omega):
        return regex
    if isinstance(regex, (Plus, Star)):
        return regex
    return Plus(regex)


class BinaryOp(Regex):
    def __init__(self, left, right):
        assert left is not Empty
        assert right is not Empty
        self.left = left
        self.right = right

    def _do_le(self, other):
        if self.left is other.left:
            return self.right < other.right
        else:
            return self.left < other.left


class Union(BinaryOp):
    def __repr__(self):
        return "(%r | %r)" % (self.left, self.right)

    def _calculate_matches_empty(self):
        return self.left.matches_empty or self.right.matches_empty


@cached
def union(left, right):
    if left is right:
        return left
    if left is Empty:
        return right
    if right is Empty:
        return left
    if left is Omega or right is Omega:
        return Omega
    if right.matches_empty:
        if left.matches_empty:
            return union(left, nonempty(right))
        else:
            return union(Epsilon, union(left, nonempty(right)))
    if isinstance(left, Union):
        return union(left.left, union(left.right, right))
    elif right < left and not isinstance(right, Union):
        assert not (left < right)
        return union(right, left)
    if isinstance(right, Plus) and left is Epsilon:
        return star(right.child)
    return Union(left, right)


class Intersection(BinaryOp):
    def __repr__(self):
        return "(%r & %r)" % (self.left, self.right)

    def _calculate_matches_empty(self):
        return self.left.matches_empty and self.right.matches_empty


@cached
def intersection(left, right):
    if left is right:
        return left
    if left is Empty or right is Empty:
        return Empty
    if left is Omega:
        return right
    if right is Omega:
        return left
    if left.matches_empty ^ right.matches_empty:
        return intersection(nonempty(left), nonempty(right))
    if isinstance(left, Intersection):
        return intersection(left.left, intersection(left.right, right))
    elif right < left and not isinstance(right, Intersection):
        return intersection(right, left)
    return Intersection(left, right)


class Subtraction(BinaryOp):
    def __repr__(self):
        if self.left is Omega:
            return "~%r" % (self.right,)
        return "(%r - %r)" % (self.left, self.right)

    def _calculate_matches_empty(self):
        return self.left.matches_empty and not self.right.matches_empty

@cached
def subtract(left, right):
    if left is Empty or right is Empty:
        return left
    if right is Epsilon:
        return nonempty(left)
    if left is Epsilon:
        if right.matches_empty:
            return Empty
        else:
            return left
    if right is Omega:
        return Empty
    if (
        isinstance(right, Subtraction) and
        left is Omega and right.left is Omega
    ):
        return right.right
    if isinstance(left, Subtraction):
        return subtract(left.left, union(left.right, right))
    if right.matches_empty:
        return nonempty(left) - nonempty(right)
    return Subtraction(left, right)
    

def negate(regex):
    return subtract(Omega, regex)


class Concatenation(BinaryOp):
    def __repr__(self):
        return "(%r + %r)" % (self.left, self.right)

    def _calculate_matches_empty(self):
        return self.left.matches_empty and self.right.matches_empty


@cached
def concatenate(left, right):
    if left is Empty or right is Empty:
        return Empty
    if left is Epsilon:
        return right
    if right is Epsilon:
        return left
    if left is Omega and right.matches_empty:
        return left
    if isinstance(right, Star) and right.child is left:
        return Plus(left)
    if isinstance(left, Star) and left.child is right:
        return Plus(right)
    if isinstance(left, Concatenation):
        return concatenate(left.left, concatenate(left.right, right))
    return Concatenation(left, right)


class NonEmpty(UnaryOp):
    def __repr__(self):
        return "nonempty(%r)" % (self.child,)

    def _calculate_matches_empty(self):
        return False


@cached
def nonempty(regex):
    if not regex.matches_empty:
        return regex
    if regex in (Epsilon, Empty):
        return Empty
    if isinstance(regex, Union):
        return nonempty(regex.left) | nonempty(regex.right)
    if isinstance(regex, Intersection):
        return nonempty(regex.left) & nonempty(regex.right)
    if isinstance(regex, (Star, Plus)):
        return plus(nonempty(regex.child))
    return NonEmpty(regex)

@cached
def derivative(regex, c):
    if regex is Omega:
        return Omega
    if regex in (Epsilon, Empty):
        return Empty
    if regex is Dot:
        return Epsilon 
    if isinstance(regex, Character):
        if regex.character == c:
            return Epsilon
        else:
            return Empty
    if isinstance(regex, NonEmpty):
        return derivative(regex.child, c)
    if isinstance(regex, Subtraction):
        return derivative(regex.left, c) - derivative(regex.right, c)
    if isinstance(regex, Star):
        return derivative(regex.child, c) + regex
    if isinstance(regex, Plus):
        return derivative(regex.child, c) + star(regex.child)
    if isinstance(regex, Union):
        return derivative(regex.left, c) | derivative(regex.right, c)
    if isinstance(regex, Intersection):
        return derivative(regex.left, c) & derivative(regex.right, c)
    if isinstance(regex, Concatenation):
        base = derivative(regex.left, c) + regex.right
        if regex.left.matches_empty:
            base |= derivative(regex.right, c)
        return base
    assert False, (regex, type(regex))

@cached
def has_matches(regex):
    return not equivalent(regex, Empty)

@cached
def valid_starts(regex):
    if regex in (Omega, Dot):
        return frozenset(ALPHABET)
    if regex in (Epsilon, Empty):
        return frozenset()
    if isinstance(regex, Character):
        return frozenset((regex.character,))
    if isinstance(regex, NonEmpty):
        return valid_starts(regex.child)
    if isinstance(regex, (Plus, Star)):
        return valid_starts(regex.child)
    if isinstance(regex, Union):
        return valid_starts(regex.left) | valid_starts(regex.right)
    if isinstance(regex, Concatenation):
        result = valid_starts(regex.left)
        if regex.left.matches_empty:
            result |= valid_starts(regex.right)
        return result
    elif isinstance(regex, Subtraction):
        base_valid_starts = valid_starts(regex.left)
    else:
        assert isinstance(regex, Intersection), regex
        base_valid_starts = valid_starts(regex.left) & valid_starts(
            regex.right)
    return frozenset(
        c for c in base_valid_starts if has_matches(derivative(regex, c))
    )

@cached
def symmetric_difference(left, right):
    return intersection(
        union(left, right),
        negate(intersection(left, right)),
    )


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


def matches(regex, string):
    for s in string:
        regex = derivative(regex, s)
    return regex.matches_empty


def equivalent(left, right):
    if left is right:
        return True
    if left.matches_empty != right.matches_empty:
        return False
    merges = UnionFind()
    merges.merge(left, right)

    stack = [(left, right)]
    while stack:
        p, q = stack.pop()
        for a in ALPHABET:
            pa = merges.find(derivative(p, a))
            qa = merges.find(derivative(q, a))
            if qa != pa:
                if pa.matches_empty != qa.matches_empty:
                    return False
                merges.merge(pa, qa)
                stack.append((pa, qa))
    return True


@cached
def lexmin(language):
    queue = deque()

    queue.append((language, ()))

    seen = set()

    while len(queue) > 0:
        x, s = queue.popleft()
        for a in ALPHABET:
            d = derivative(x, a)
            t = (a, s)
            if d.matches_empty:
                result = bytearray()
                while t:
                    a, t = t
                    result.append(a)
                result.reverse()
                return bytes(result)
            elif d in seen:
                continue
            seen.add(d)
            queue.append((d, t))


class RegexKey(object):
    def __init__(self, regex):
        if equivalent(regex, Epsilon):
            self.__hash = 0
        elif equivalent(regex, Empty):
            self.__hash = 1 
        else:
            self.__hash = hash((
                regex.matches_empty,
                lexmin(nonempty(regex)),
            ))
        self.regex = regex

    def __eq__(self, other):
        if not isinstance(other, RegexKey):
            return NotImplemented
        return equivalent(self.regex, other.regex)

    def __ne__(self, other):
        if not isinstance(other, RegexKey):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self):
        return self.__hash


def build_dfa(regex):
    regex_to_states = {}
    keys_to_states = {}
    states = []
    transitions = []
    
    def state_for(regex):
        try:
            return regex_to_states[regex]
        except KeyError:
            pass
        key = RegexKey(regex)
        try:
            result = keys_to_states[key]
            regex_to_states[regex] = result
            return result
        except KeyError:
            pass
        i = len(states)
        states.append(regex)
        keys_to_states[key] = i
        regex_to_states[regex] = i
        return i

    state_for(regex)
    assert len(states) == 1
    while len(transitions) < len(states):
        i = len(transitions)
        re = states[i]
        transitions.append({
            c: state_for(derivative(re, c))
            for c in valid_starts(re)
        })

    return [s.matches_empty for s in states], transitions


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


class VoseAliasSampler(object):
    """Samples values from a weighted distribution using Vose's algorithm for
    the Alias Method.

    See http://www.keithschwarz.com/darts-dice-coins/ for details.

    """

    def __init__(self, weights, options=None):
        assert any(weights)
        assert all(w >= 0 for w in weights)

        n = len(weights)
        if options is None:
            options = range(n)

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


class Simulator(Regex):
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

    def parameter_for_size(self, size):
        return sympy.solve(self.__expected_size - size, self.__symbol)

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
        if any(w < 0 for w in state_weights):
            raise ValueError("Parameter %r too large" % (parameter,)) 

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
