import hypothesis.strategies as st
import regex as rd
from falbs import Simulator, compute_generating_functions, ParamTooLarge
from hypothesis import given, assume, note, example, reject
from helpers import regex
from sympy import series
import pytest


@given(regex())
def test_can_compute_a_generating_function(re):
    compute_generating_functions(*rd.build_dfa(re))


@example(rd.literal(b'\0\0'), 0, None, 0.01)
@example(rd.star(rd.char(b'\0')), 0, None, 0.01)
@example(rd.star(rd.char(b'\0')), 0, 1, 0.01)
@example(rd.star(rd.char(b'\0')), 0, 0, 0.5)
@given(
    regex(state_bound=10), st.integers(), st.none() | st.integers(0, 10),
    st.floats(0, 0.5),
)
def test_can_simulate_accurately(regex, seed, max_size, param):
    assume(param > 0)
    assume(rd.has_matches(regex))
    if max_size is not None:
        assume(rd.has_matches(rd.bounded(regex, max_size)))
    sim = Simulator(regex, seed)
    d = sim.draw(param, max_size=max_size)
    try:
        next(d)
    except ParamTooLarge:
        assert param > 0
        reject()
    for _ in range(10):
        x = next(d)
        assert rd.matches(regex, x)
        if max_size is not None:
            assert len(x) <= max_size


@given(regex(), st.integers(0, 10))
def test_power_series_counts_language(re, i):
    dfa = rd.build_dfa(re)
    z, gfs = compute_generating_functions(*dfa)
    f = gfs[0]
    note(f)
    power_series = series(f, n=i + 1)
    counter = rd.LanguageCounter(*dfa)
    assert counter.count(i) == power_series.coeff(z, i)


def test_raises_if_parameter_too_large():
    reg = rd.star(rd.char(b'\0\1\2\3'))
    sim = Simulator(reg, 0)
    with pytest.raises(ValueError):
        next(sim.draw(0.5))
