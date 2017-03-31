from helpers import regex
import regex as rd
from hypothesis import given, assume, strategies as st
from pyrsistent import pset


@given(regex())
def test_can_build_a_dfa(re):
    assume(rd.has_matches(re))
    rd.build_dfa(re)


def test_char_classes_1():
    assert rd.character_classes(
        rd.concatenate(rd.union(rd.char(0), rd.char(1)), rd.char(2))
    ) == pset([pset([0, 1])])


@given(regex())
def test_characters_in_same_class_produce_equivalent_expressions(re):
    assume(rd.has_matches(re))
    classes = rd.character_classes(re)
    assume(any(len(cs) > 1 for cs in classes))
    for cs in classes:
        if len(cs) > 1:
            derivs = [rd.derivative(re, c) for c in cs]
            for a in derivs:
                for b in derivs:
                    assert rd.equivalent(a, b)


@given(regex())
def test_decompilation(re):
    assume(rd.has_matches(re))
    dfa = rd.build_dfa(re)
    rewritten = rd.decompile_dfa(*dfa)
    assert rd.equivalent(re, rewritten)


def symdiff(x, y):
    return rd.union(rd.subtract(x, y), rd.subtract(y, x))


@given(regex(), regex())
def test_lexmin_of_symmetric_difference_is_refutation(x, y):
    assume(not rd.equivalent(x, y))
    w = rd.lexmin(symdiff(x, y))
    assert w is not None
    assert w == rd.witness_difference(x, y)


@given(regex(), st.data())
def test_lexmin_of_mutated_regex_is_refutation(x, data):
    assume(rd.has_matches(x))

    accepting, transitions = rd.build_dfa(x)

    j = data.draw(st.integers(0, len(accepting) - 1))

    assume(transitions[j])
    c = data.draw(st.sampled_from(sorted(transitions[j])))
    transitions[j][c] = data.draw(st.integers(0, len(accepting) - 1))

    y = rd.decompile_dfa(accepting, transitions)

    assume(rd.has_matches(y))
    assume(not rd.equivalent(x, y))

    w = rd.lexmin(symdiff(x, y))
    assert w is not None
    assert w == rd.witness_difference(x, y)
