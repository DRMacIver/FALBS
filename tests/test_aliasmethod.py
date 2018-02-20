from falbs.aliasmethod import VoseAliasSampler
from hypothesis import given, assume, strategies as st, example
from random import Random


@example([1], Random(0))
@example([0.0, 1], Random(1))
@example([1, 1, 0], Random(0))
@given(st.lists(st.floats(0, 1), min_size=1), st.randoms())
def test_can_build_a_sampler(ls, rnd):
    assume(sum(ls) > 0)
    sampler = VoseAliasSampler(ls, range(len(ls)))
    i = sampler.sample(rnd)
    assert ls[i] > 0
