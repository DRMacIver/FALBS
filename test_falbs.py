import hypothesis.strategies as st
import regex as rd
from falbs import Simulator, compute_generating_functions
from hypothesis import given, assume, note
from helpers import regex
from sympy import series


@given(regex())
def test_can_compute_a_generating_function(re):
    compute_generating_functions(*rd.build_dfa(re))


@given(regex(), st.integers())
def test_can_simulate_accurately(regex, seed):
    assume(rd.has_matches(regex))
    sim = Simulator(regex, seed)
    d = sim.draw(0.001)
    for _ in range(10):
        assert rd.matches(regex, next(d))


@given(regex(), st.integers(0, 10))
def test_power_series_counts_language(re, i):
    dfa = rd.build_dfa(re)
    z, gfs = compute_generating_functions(*dfa)
    f = gfs[0]
    note(f)
    power_series = series(f, n=i + 1)
    counter = rd.LanguageCounter(*dfa)
    assert counter.count(i) == power_series.coeff(z, i)
