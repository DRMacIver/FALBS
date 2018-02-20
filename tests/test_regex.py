from tests.helpers import regex
import falbs.regex as rd
from hypothesis import given, assume, strategies as st, example
from pyrsistent import pset
import pytest


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


@example(rd.star(rd.char(b'0')))
@example(rd.subtract(rd.star(rd.char(b'0')), rd.char(b'0')))
@given(regex())
def test_infinite_regex_have_more_than_one_solution(reg):
    assume(rd.is_infinite(reg))
    x = rd.subtract(reg, rd.literal(rd.lexmin(reg)))
    assert rd.has_matches(x)


@example(rd.concatenate(rd.star(rd.char(b'\0')), rd.char(b'\1')))
@example(rd.union(rd.char(b'\0'), rd.star(rd.literal(b'\0\0'))))
@example(rd.star(rd.char(0)))
@given(regex())
def test_decompilation(re):
    assume(rd.has_matches(re))
    dfa = rd.build_dfa(re)
    rewritten = rd.decompile_dfa(*dfa)
    assert rd.equivalent(re, rewritten)


def symdiff(x, y):
    return rd.union(rd.subtract(x, y), rd.subtract(y, x))


@example(
    rd.union(
        rd.char(b'\x00'), rd.subtract(rd.star(rd.char(b'\x01')), rd.Epsilon)),
    rd.intersection(rd.char(b'\x00'), rd.star(rd.char(b'\x00')))
)
@example(x=rd.literal(b'01'), y=rd.literal(b'11'))
@given(regex(), regex())
def test_lexmin_of_symmetric_difference_is_refutation(x, y):
    assume(not rd.equivalent(x, y))
    w = rd.lexmin(symdiff(x, y))
    assert w is not None
    assert w == rd.witness_difference(x, y)


@example(rd.union(rd.char(b'\0'), rd.star(rd.char(b'\x00'))))
@given(regex())
def test_no_refutation_for_decompilation(re):
    dec = rd.decompile_dfa(*rd.build_dfa(re))
    assert rd.witness_difference(dec, re) is None


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


@example(rd.union(rd.char(b'\x00'), rd.star(rd.char(b'\x00'))), 0, 1)
@example(rd.star(rd.char(b'\x00')), 1, 1)
@given(regex(), st.integers(0, 10), st.integers(0, 10))
def test_count_below_bound_is_the_same(re, m, n):
    assume(rd.has_matches(re))
    m, n = sorted((m, n))

    count1 = rd.LanguageCounter(*rd.build_dfa(re)).count(m)
    count2 = rd.LanguageCounter(*rd.build_dfa(rd.bounded(re, n))).count(m)
    assert count1 == count2


def test_clearing_caches_resets_identity():
    c1 = rd.char(0)
    c2 = rd.char(0)
    rd.clear_caches()
    c3 = rd.char(0)
    assert c1 is c2 is not c3


@pytest.mark.parametrize(
    'c',
    [rd.Empty, rd.Epsilon, rd.char(0), rd.bounded(rd.star(rd.char(0)), 1)]
)
def test_bounded_does_not_wrap_obviously_bounded(c):
    assert rd.bounded(c, 1) is c
    assert rd.bounded(rd.Empty, 1) is rd.Empty


def test_basic_impossible_bounds_are_empty():
    assert rd.bounded(rd.char(0), -1) is rd.Empty
    assert rd.bounded(rd.char(0), 0) is rd.Empty


def test_bounds_are_not_nested():
    x = rd.bounded(rd.star(rd.char(0)), 7)
    y = rd.bounded(x, 5)

    assert x.bound == 7
    assert y.bound == 5
    assert isinstance(y.child, rd.Star)


def test_bounds_propagate_through_unions():
    assert isinstance(
        rd.bounded(rd.union(rd.star(rd.char(0)), rd.star(rd.char(1))), 1),
        rd.Union
    )


def test_bounds_propagate_through_intersections():
    x = rd.star(rd.char(b'\0\1'))
    y = rd.star(rd.char(b'\1\2'))
    assert isinstance(
        rd.bounded(rd.intersection(x, y), 3),
        rd.Intersection
    )


def test_bounds_propagate_through_subtraction():
    x = rd.star(rd.char(b'\0\1'))
    y = rd.literal(b'\0\0\0\1')
    z = rd.subtract(x, y)
    b = rd.bounded(z, 10)
    assert isinstance(b, rd.Subtraction)
    assert isinstance(b.left, rd.Bounded)


@example(rd.concatenate(rd.char(b'\x00\x01'), rd.char(b'\x00')), 2)
@given(regex(), st.integers(0, 10))
def test_bounded_min_matches_bounds(re, n):
    bd = rd.bounded(re, n)
    assume(rd.has_matches(bd))
    assert len(rd.lexmin(bd)) <= n


@example(rd.char(0))
@given(regex())
def test_non_empty_is_identity_on_non_nullable(re):
    assume(not re.nullable)
    assume(rd.has_matches(re))
    assert rd.nonempty(re) is re


def test_star_collapses_trivial_children():
    assert rd.star(rd.Empty) is rd.Epsilon
    assert rd.star(rd.Epsilon) is rd.Epsilon


def test_star_collapses_stars():
    x = rd.star(rd.char(0))
    assert rd.star(x) is x


def test_flattens_unions():
    x = rd.star(rd.char(0))
    y = rd.star(rd.char(1))
    z = rd.star(rd.char(2))
    assert rd.union(x, rd.union(y, z)) is rd.union(rd.union(x, z), y)


def test_flattens_intersections():
    x = rd.star(rd.char(b'01'))
    y = rd.star(rd.char(b'02'))
    z = rd.star(rd.char(b'03'))
    assert rd.intersection(x, rd.intersection(y, z)) is \
        rd.intersection(rd.intersection(x, z), y)


def test_removes_empty_from_unions():
    c = rd.char(0)
    assert c is rd.union(rd.Empty, c)


def test_union_of_empty_is_empty():
    assert rd.union(rd.Empty, rd.Empty) is rd.Empty


def test_epsilon_prunes_down_intersections():
    assert rd.intersection(rd.Epsilon, rd.star(rd.char(0))) is rd.Epsilon
    assert rd.intersection(rd.Epsilon, rd.char(0)) is rd.Empty


def test_empty_kills_intersections():
    assert rd.intersection(rd.Empty, rd.Epsilon) is rd.Empty


def test_self_intersection_is_identity():
    x = rd.char(0)
    assert rd.intersection(x, x, x) is x


def test_valid_starts_of_nullable_cat():
    x = rd.concatenate(rd.star(rd.char(0)), rd.char(1))
    assert rd.valid_starts(x) == pset([0, 1])


def test_empty_concatenation_is_epsilon():
    assert rd.concatenate() is rd.Epsilon


def test_single_concatenation_is_self():
    assert rd.concatenate(rd.char(0)) is rd.char(0)


def test_rebalances_concatenation():
    x = rd.char(0)
    y = rd.star(rd.char(1))
    z = rd.char(2)
    assert rd.concatenate(x, rd.concatenate(y, z)) is \
        rd.concatenate(rd.concatenate(x, y), z)


def test_self_subtraction_is_empty():
    x = rd.char(0)
    assert rd.subtract(x, x) is rd.Empty


def test_empty_subtraction_is_identity():
    x = rd.char(0)
    assert rd.subtract(x, rd.Empty) is x


def test_subtraction_from_empty_is_empty():
    x = rd.char(0)
    assert rd.subtract(rd.Empty, x) is rd.Empty


def test_subtraction_from_epsilon_checks_nullability():
    assert rd.subtract(rd.Epsilon, rd.char(0)) is rd.Epsilon
    assert rd.subtract(rd.Epsilon, rd.star(rd.char(0))) is rd.Empty


def test_merges_multiple_subtracts():
    x = rd.star(rd.char(b'012'))
    y = rd.star(rd.char(b'0'))
    z = rd.star(rd.char(b'1'))

    t = rd.subtract(rd.subtract(x, y), z)
    assert t is rd.subtract(x, rd.union(y, z))

    t = rd.subtract(x, rd.subtract(y, z))
    assert t.nullable
    assert isinstance(t, rd.Union)


def test_derivatives_of_unions():
    assert rd.derivative(
        rd.union(rd.star(rd.char(0)), rd.star(rd.char(1))), 0
    ) is rd.star(rd.char(0))


def test_derivatives_of_intersection():
    x = rd.star(rd.char(b'\0\1'))
    y = rd.star(rd.literal(b'\0\1'))
    z = rd.intersection(x, y)
    d1 = rd.derivative(z, 0)
    d2 = rd.derivative(d1, 1)
    assert d2 is z


def test_valid_starts_of_subtraction():
    x = rd.star(rd.char(b'\0\1'))
    y = rd.char(b'\1')
    z = rd.subtract(x, y)
    assert rd.valid_starts(z) == pset([0, 1])


def test_difference_of_same_is_none():
    x = rd.char(0)
    assert rd.witness_difference(x, x) is None


def test_difference_of_epsilon_and_non_nullable_is_epsilon():
    assert rd.witness_difference(rd.char(0), rd.Epsilon) is b''


def test_witness_difference_of_literals_is_smaller_of_two():
    assert rd.witness_difference(rd.literal(b'00'), rd.literal(b'01')) == b'00'


def test_lexmin_of_star_is_empty():
    assert rd.lexmin(rd.star(rd.char(b'0'))) is b''


def test_empty_is_not_infinite():
    assert not rd.is_infinite(rd.Empty)


def test_basic_finite_are_not_infinite():
    assert not rd.is_infinite(rd.Epsilon)
    assert not rd.is_infinite(rd.char(0))


def test_union_of_infinite_and_finite_is_infinite():
    assert rd.is_infinite(rd.union(rd.char(1), rd.star(rd.char(0))))


def test_can_walk_graph_for_infintiy():
    assert rd.is_infinite(rd.intersection(
        rd.star(rd.char(b'01')), rd.star(rd.char(b'12'))
    ))


def test_bounded_is_not_infinite():
    assert not rd.is_infinite(rd.bounded(rd.star(rd.char(0)), 10 ** 6))


def to_basic(re):
    return rd.decompile_dfa(*rd.build_dfa(re))


def test_complex_graphs_may_be_finite():
    x = to_basic(rd.bounded(
        rd.union(rd.star(rd.char(0)), rd.star(rd.char(1))), 20))

    assert not rd.is_infinite(x)


def test_non_empty_star_dfa():
    accepting, _ = rd.build_dfa(rd.nonempty(rd.star(rd.char(0))))
    assert accepting == [False, True]


def test_two_phase_dfa():
    re = rd.concatenate(rd.star(rd.char(0)), rd.star(rd.char(1)))
    accepting, transitions = rd.build_dfa(re)
    assert accepting == [True, True]
    assert transitions == [{0: 0, 1: 1}, {1: 1}]


def test_lexmin_of_empty_is_none():
    assert rd.lexmin(rd.Empty) is None


def test_trival_dfa_from_intersection():
    assert rd.build_dfa(
        rd.intersection(rd.char(b'\x00'), rd.char(b'\x00\x01'))) == (
        [False, True], [{0: 1}, {}]
    )
