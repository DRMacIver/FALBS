"""Module implementing extended regular expressions and some operations on
them."""


import functools
from pyrsistent import PSet, pset
from unionfind import UnionFind
from functools import reduce
import operator

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
        if isinstance(result, Regex):
            # Super cheap way of normalizing a lot of useless expressions:
            # Check if there are any valid characters it could possibly start
            # with. If not, it has no non-null matches so it'seither Epsilon
            # or Empty, and we can tell which.by checking if it's nullable.
            # This won't cut out all possible dead ends, but it does a pretty
            # good job of reducing them.
            # Is this a hack? It's probably a hack.
            if not result.plausible_starts:
                if result.nullable:
                    result = Epsilon
                else:
                    result = Empty
        cache[args] = result
        return result
    return accept


class Regex(object):
    def __init__(self, nullable, plausible_starts):
        self.nullable = nullable
        self.plausible_starts = plausible_starts


class Special(Regex):
    def __init__(self, name, nullable, plausible_starts):
        Regex.__init__(self, nullable, plausible_starts)
        self.name = name

    def __repr__(self):
        return self.name


Epsilon = Special("Epsilon", True, pset())
Empty = Special("Empty", False, pset())


class Characters(Regex):
    def __init__(self, c):
        Regex.__init__(self, False, c),

    @property
    def characters(self):
        return self.plausible_starts

    def __repr__(self):
        return "char(%r)" % (bytes(list(self.characters)),)


@cached
def char(c):
    if isinstance(c, bytes):
        return char(pset(c))
    if isinstance(c, int):
        return char(pset((c,)))
    return Characters(c)


class Star(Regex):
    def __init__(self, child):
        Regex.__init__(self, True, child.plausible_starts)
        self.child = child

    def __repr__(self):
        return "star(%r)" % (self.child,)


@cached
def star(child):
    if child in (Epsilon, Empty):
        return Epsilon
    if isinstance(child, Star):
        return child
    return Star(child)


def nonempty(regex):
    return subtract(regex, Epsilon)


class Union(Regex):
    def __init__(self, children):
        assert len(children) > 1
        assert isinstance(children, PSet)
        assert Empty not in children
        Regex.__init__(
            self, any(c.nullable for c in children),
            reduce(operator.or_, (c.plausible_starts for c in children))
        )
        self.children = children

    def __repr__(self):
        parts = sorted(map(repr, self.children), key=lambda x: (len(x), x))
        return "union(%s)" % (', '.join(parts),)


@cached
def _union_from_set(children):
    return Union(children)


@cached
def union(*args):
    characters = pset()
    children = pset().evolver()
    bulk = []
    for a in args:
        if isinstance(a, Union):
            bulk.append(a.children)
        elif isinstance(a, Characters):
            characters |= a.characters
        elif a is Empty:
            pass
        else:
            children.add(a)
    if characters:
        children.add(char(characters))
    children = children.persistent()
    for b in bulk:
        children |= b
    if len(children) == 0:
        return Empty
    if len(children) == 1:
        return list(children)[0]
    return _union_from_set(children)


class Intersection(Regex):
    def __init__(self, children):
        assert len(children) > 1
        assert isinstance(children, PSet)
        assert Empty not in children
        Regex.__init__(
            self, all(c.nullable for c in children),
            reduce(operator.and_, (c.plausible_starts for c in children))
        )
        self.children = children

    def __repr__(self):
        parts = sorted(map(repr, self.children), key=lambda x: (len(x), x))
        return "intersection(%s)" % (', '.join(parts),)


@cached
def _intersection_from_set(children):
    result = Intersection(children)
    if Epsilon in children:
        if result.nullable:
            return Epsilon
        else:
            return Empty
    return result


@cached
def intersection(*args):
    children = pset().evolver()
    bulk = []
    for a in args:
        if isinstance(a, Intersection):
            bulk.append(a.children)
        elif a is Empty:
            return Empty
        else:
            children.add(a)
    children = children.persistent()
    for b in bulk:
        children |= b
    if len(children) == 0:
        return Empty
    if len(children) == 1:
        return list(children)[0]
    return _intersection_from_set(children)


class Concatenation(Regex):
    def __init__(self, left, right):
        plausible_starts = left.plausible_starts
        if left.nullable:
            plausible_starts |= right.plausible_starts
        Regex.__init__(
            self, left.nullable and right.nullable, plausible_starts)
        self.left = left
        self.right = right

    def __repr__(self):
        return 'concatenate(%r, %r)' % (self.left, self.right)


@cached
def concatenate(*args):
    if not args:
        return Epsilon
    if len(args) == 1:
        return args[0]
    if len(args) > 2:
        result = Epsilon
        for c in reversed(args):
            result = concatenate(c, result)
        return result

    left, right = args

    if left is Empty or right is Empty:
        return Empty
    if left is Epsilon:
        return right
    if right is Epsilon:
        return left
    if isinstance(left, Concatenation):
        return concatenate(left.left, concatenate(left.right, right))
    return Concatenation(left, right)


class Subtraction(Regex):
    def __init__(self, left, right):
        Regex.__init__(
            self, left.nullable and not right.nullable,
            left.plausible_starts)
        self.left = left
        self.right = right

    def __repr__(self):
        return "subtract(%r, %r)" % (self.left, self.right)


@cached
def subtract(left, right):
    if left is right:
        return Empty
    if right is Empty or left is Empty:
        return left
    if right is Epsilon and not left.nullable:
        return left
    if left is Epsilon:
        if right.nullable:
            return Empty
        else:
            return left
    if isinstance(left, Subtraction):
        return subtract(left.left, union(left.right, right))
    if isinstance(right, Subtraction):
        return union(
            left & right.right,
            subtract(left, right.left),
        )
    return Subtraction(left, right)


@cached
def derivative(regex, c):
    if regex in (Empty, Epsilon):
        return Empty
    if c not in regex.plausible_starts:
        return Empty
    if isinstance(regex, Characters):
        if c in regex.characters:
            return Epsilon
        else:
            return Empty
    if isinstance(regex, Union):
        return union(*[derivative(r, c) for r in regex.children])
    if isinstance(regex, Intersection):
        return intersection(*[derivative(r, c) for r in regex.children])
    if isinstance(regex, Star):
        return concatenate(derivative(regex.child, c), regex)
    if isinstance(regex, Subtraction):
        return subtract(derivative(regex.left, c), derivative(regex.right, c))
    if isinstance(regex, Concatenation):
        result = concatenate(derivative(regex.left, c), regex.right)
        if regex.left.nullable:
            result = union(result, derivative(regex.right, c))
        return result
    assert False, (type(regex), regex)


def equivalent(left, right):
    if left is right:
        return True
    if left.nullable != right.nullable:
        return False
    merges = UnionFind()
    merges.merge(left, right)

    stack = [(left, right)]
    while stack:
        p, q = stack.pop()
        for a in p.plausible_starts | q.plausible_starts:
            pa = merges.find(derivative(p, a))
            qa = merges.find(derivative(q, a))
            if qa != pa:
                if pa.nullable != qa.nullable:
                    return False
                merges.merge(pa, qa)
                stack.append((pa, qa))
    return True


@cached
def has_matches(regex):
    return not equivalent(regex, Empty)


@cached
def valid_starts(regex):
    if regex in (Epsilon, Empty):
        return regex.plausible_starts
    if isinstance(regex, Characters):
        return regex.characters
    if isinstance(regex, Star):
        return valid_starts(regex.child)
    if isinstance(regex, Union):
        return reduce(operator.or_, (valid_starts(c) for c in regex.children))
    return pset(
        c for c in regex.plausible_starts if has_matches(derivative(regex, c))
    )


def set_union(sets):
    if not sets:
        return pset()
    return reduce(operator.or_, sets)


def join_classes(classes):
    classes = tuple(classes)

    if not classes:
        return pset([pset()])
    if len(classes) == 1:
        return classes[0]
    alphabets = [set_union(c) for c in classes]
    whole_alphabet = set_union(alphabets)
    if len(whole_alphabet) == 0:
        return pset([pset()])
    if len(whole_alphabet) == 1:
        return pset((whole_alphabet,))
    adjusted_classes = []
    for a, c in zip(alphabets, classes):
        if not whole_alphabet.issubset(a):
            c = c | pset((whole_alphabet - a,))
        adjusted_classes.append(c)

    classes = {whole_alphabet}
    for cls in adjusted_classes:
        classes = {
           c & d for c in cls for d in classes
        }
    classes.discard(pset())
    return pset(classes)


@cached
def character_classes(regex):
    if regex in (Empty, Epsilon):
        return pset()
    if isinstance(regex, Characters):
        return pset([regex.characters])
    if isinstance(regex, Star):
        return character_classes(regex.child)
    if isinstance(regex, (Union, Intersection)):
        return join_classes(map(character_classes, regex.children))
    if isinstance(regex, Concatenation):
        if regex.left.nullable:
            return join_classes((
                character_classes(regex.left),
                character_classes(regex.right)))
        else:
            return character_classes(regex.left)
    if isinstance(regex, Subtraction):
        return join_classes((
            character_classes(regex.left),
            character_classes(regex.right)))
    assert False, (type(regex), regex)


def matches(regex, string):
    for c in string:
        regex = derivative(regex, c)
    return regex.nullable


def build_dfa(regex):
    regex_to_states = {}
    states = []
    transitions = []

    def state_for(regex):
        try:
            return regex_to_states[regex]
        except KeyError:
            pass
        for i, r in enumerate(states):
            if equivalent(r, regex):
                regex_to_states[regex] = i
                return i
        i = len(states)
        states.append(regex)
        regex_to_states[regex] = i
        return i

    state_for(regex)
    assert len(states) == 1
    while len(transitions) < len(states):
        i = len(transitions)
        re = states[i]

        table = {}

        for cs in character_classes(re):
            c, *rest = cs
            target = derivative(re, c)
            if has_matches(target):
                i = state_for(target)
                for c in cs:
                    table[c] = i
        transitions.append(table)

    return [s.nullable for s in states], transitions


class LanguageCounter(object):
    def __init__(self, terminal, transitions):
        self.__terminal = terminal
        self.__transitions = transitions
        self.__cache = {}

    def __count_from(self, state, length):
        if length < 0:
            return 0
        if length == 0:
            return int(self.__terminal[state])
        key = (state, length)
        try:
            return self.__cache[key]
        except KeyError:
            pass
        result = 0
        for i in self.__transitions[state].values():
            result += self.__count_from(i, length - 1)
        self.__cache[key] = result
        return result

    def count(self, size):
        return self.__count_from(0, size)
