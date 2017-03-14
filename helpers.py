import regex as rd
import hypothesis.strategies as st
from hypothesis import assume


@st.composite
def regex(draw):
    bases = draw(st.lists(
        st.builds(rd.char, st.integers(0, 255),), min_size=1, average_size=20))

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
    assume(not isinstance(result, rd.Characters))
    return result

