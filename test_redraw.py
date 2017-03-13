import hypothesis.strategies as st
import cleanroom as rd
from hypothesis import given, assume


@st.composite
def regex(draw):
    bases = draw(st.lists(
        st.builds(rd.char, st.integers(0, 255),), min_size=1, average_size=99))

    while len(bases) > 1:
        n, op = draw(
            st.sampled_from((
                (1, rd.star), (1, rd.nonempty),
                (2, rd.union), (2, rd.intersection), (2, rd.concatenate),
                (2, rd.subtract),
            )),
        )
        if n > len(bases):
            continue
        args = [bases.pop() for _ in range(n)]
        bases.append(op(*args))
    result = bases[0]
    assume(result not in (rd.Empty, rd.Epsilon))
    assume(not isinstance(result, rd.Character))
    return result


@given(regex())
def test_can_build_a_dfa(re):
    assume(rd.has_matches(re))
    rd.build_dfa(re)


@given(regex())
def test_can_compute_a_generating_function(re):
    rd.compute_generating_functions(*rd.build_dfa(re))


@given(regex(), st.integers())
def test_can_simulate_accurately(regex, seed):
    assume(rd.has_matches(regex))
    sim = rd.Simulator(regex, seed)
    d = sim.draw(0.001)
    for _ in range(100):
        assert rd.matches(regex, next(d))
