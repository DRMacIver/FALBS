import hypothesis.strategies as st
import regex as rd
from falbs import Simulator, compute_generating_functions
from hypothesis import given, assume
from helpers import regex
from sympy import fps
from sympy.series.formal import FormalPowerSeries


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


@given(regex())
def test_formal_power_series_counts_language(re):
    dfa = rd.build_dfa(re)
    z, gfs = compute_generating_functions(*dfa)
    # This is a stupid hack to reduce the complexity of the expressions to
    # something sympy can reliably produce a formal power series for.
    assume(len(repr(gfs[0])) <= 20)
    power_series = fps(gfs[0])

    # See https://github.com/sympy/sympy/issues/12310
    assume(isinstance(power_series, FormalPowerSeries))
    counter = rd.LanguageCounter(*dfa)

    for i in range(10):
        assert counter.count(i) == power_series[i].subs(z, 1)
