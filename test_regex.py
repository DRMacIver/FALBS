from helpers import regex
import regex as rd
from hypothesis import given, assume
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
