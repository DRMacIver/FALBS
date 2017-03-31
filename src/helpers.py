import regex as rd
import hypothesis.strategies as st
from hypothesis import assume


@st.composite
def regex(draw, state_bound=None):
    bases = draw(st.lists(
        st.builds(rd.char, st.integers(0, 255),), min_size=1, average_size=20))

    while len(bases) > 1:
        n, op = draw(
            st.sampled_from((
                (1, rd.star), (1, rd.nonempty),
                (2, rd.union), (2, rd.intersection), (2, rd.concatenate),
                (2, rd.subtract), (1, lambda r: rd.bounded(r, 1)),
                (1, lambda r: rd.bounded(r, 3)),
                (1, lambda r: rd.bounded(r, 10)),
            )),
        )
        if n > len(bases):
            continue
        args = [bases.pop() for _ in range(n)]
        bases.append(op(*args))
    result = bases[0]
    assume(result not in (rd.Empty, rd.Epsilon))
    assume(not isinstance(result, rd.Characters))

    if state_bound is not None:
        seen = {result}
        threshold = {result}
        while threshold:
            threshold = {
                rd.derivative(u, c) for u in threshold
                for c in rd.valid_starts(u)
            }
            threshold -= seen
            seen.update(threshold)
            assume(len(seen) <= state_bound)
    return result
