import hypothesis.strategies as st
import regex as rd
from falbs import Simulator, compute_generating_functions
from hypothesis import given, assume
from helpers import regex


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
