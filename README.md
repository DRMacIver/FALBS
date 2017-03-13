# Fully Automated Luxury Boltzmann Samplers for Regular Languages

Some experiments around generating arbitrary strings from a regular
language, such that all strings of the same length are produced with equal
probability.

Advance warning: If you don't know what a Boltzmann sampler is then you're
going to have a bad time reading past this point. IOU one proper blog post
about this.

This repo contains demo code I wrote to explore a concept: Can we automatically
derive a closed form for a [Boltzmann Sampler](http://algo.inria.fr/flajolet/Publications/DuFlLoSc04.pdf)
for a regular language.

Turns out we can! If you have a deterministic finite automaton you can just
randomly walk it, weighting your walk by the counting generating functions for
the languages matched starting from each state.

We can then calculate those functions using symbolic linear algebra: The
generating counting functions for each state in a deterministic finite
automaton are related by a bunch of affine equations, so we can just pass them
to a solver that understands linear algebra over the field of rational
functions. Conveniently, sympy provides us with one.

So this code implements roughly the following algorithm:

1. Use regular expression derivatives to compile an extended regular expression
   to a fully minimized deterministic finite automaton.
2. Based on that automaton, use sympy to figure out the counting generating
   functions for each state in the DFA.
3. Given those generating functions and a Boltzmann parameter, build weighted
   discrete samplers using Vose's algorithm for the alias method for each
   state.
4. Randomly walk the graph using those samplers until we sample a Stop event,
   emitting characters as we walk.

It's not very efficient (it's at least O(n^2), and might be O(n^3) depending on
how sympy handles LU solving on sparse matrices, where n is the number of
states in the compiled DFA, which may in turn be exponential in the size of the
regular expression), but as long as the DFA is small then this works pretty
well.

This code is very far from production ready and if you wanted to actually use
something like this you should look at something like [revex](https://github.com/lucaswiman/revex)
instead. This code exists mostly to explore a concept and learn a bit more
about how various conceptual tools fit together. You might find it edifying to
read it, and reading code usually requires running it, but gods forbid you
actually want to run it for non-educational purposes.
