import redraw as rd
from hypothesis import strategies as st
from hypothesis import given, assume


def n_times(x, n):
    t = rd.Epsilon
    for _ in range(n):
        t = rd.concatenate(x, t)
    return t


def test_stuff():
    r = rd.char(0) | rd.star(~rd.char(0))
    sim = rd.Simulator(r, 0)
    stream = sim.draw(0.001)
    for _ in range(10):
        s = next(stream)
        assert (
            s == b'\0' or
            s == b'' or
            0 not in s
        )
    


def test_can_intersect_stuff():
    ten = n_times(rd.char(0) | rd.char(1), 10)
    no_triples = ~(rd.Omega + rd.char(0) + rd.char(0) + rd.char(0) + rd.Omega)

    assert no_triples < ten
    assert not (ten < no_triples)

    x = ten & no_triples
    assert rd.lexmin(x) == bytes([
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0
    ])


def test_can_simulate_a_short_string():
    r = rd.Simulator(rd.char(0), 0)
    for p in range(1, 10):
        p *= 0.1
        assert next(r.draw(p)) == b'\0'


@st.composite
def regex(draw):
    bases = draw(st.lists(
        st.one_of(
            st.builds(rd.char, st.integers(0, 255)),
            st.sampled_from((rd.Empty, rd.Omega, rd.Dot, rd.Epsilon)),
    ), min_size=1))

    while len(bases) > 1:
        n, op = draw(
            st.sampled_from((
                (1, rd.negate), (1, rd.star), (1, rd.nonempty),
                (2, rd.union), (2, rd.intersection), (2, rd.concatenate),
                (2, rd.subtract),
            )),
        )
        if n > len(bases):
            continue
        args = [bases.pop() for _ in range(n)]
        bases.append(op(*args))
    return bases[0]


@given(regex())
def test_can_build_a_dfa(re):
    rd.build_dfa(re)


@given(regex(), st.integers(), st.floats(0, 0.99 / 256))
def test_matches_what_it_generates(regex, seed, param):
    assume(param > 0)
    rd.clear_caches()
    assume(rd.has_matches(regex))
    simulator = rd.Simulator(regex, seed=seed)
    d = simulator.draw(param)
    for _ in range(10):
        s = next(d)
        assert rd.matches(regex, s)


@given(regex(), regex(), st.integers(), st.floats(0, 0.99 / 256))
def test_values_from_union_match_either(r1, r2, seed, param):
    assume(param > 0)
    rd.clear_caches()
    assume(rd.has_matches(r1))
    assume(rd.has_matches(r2))
    simulator = rd.Simulator(r1 | r2, seed=seed)
    d = simulator.draw(param)
    for _ in range(10):
        s = next(d)
        assert rd.matches(r1, s) | rd.matches(r2, s)


@given(regex(), regex(), st.integers(), st.floats(0, 0.99 / 256))
def test_values_from_intersection_match_both(r1, r2, seed, param):
    assume(param > 0)
    rd.clear_caches()
    assume(rd.has_matches(r1))
    assume(rd.has_matches(r2))
    simulator = rd.Simulator(r1 & r2, seed=seed)
    d = simulator.draw(param)
    for _ in range(10):
        s = next(d)
        assert rd.matches(r1, s) & rd.matches(r2, s)
