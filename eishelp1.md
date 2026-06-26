# Internal Format Used In

## [The On-Line Encyclopedia of Integer Sequences](https://oeis.org/Seis.html)

This file describes the internal format used in [The On-Line Encyclopedia of Integer Sequences](https://oeis.org/Seis.html)

[For a description of the **standard (or beautified) format** used in the web pages, click [here](https://oeis.org/eishelp2.html).]

Each sequence is described by about 10 lines, each line beginning

```
%x Aabcdef
```

where x is a letter (I, S, T, N, etc.) and abcdef is the 6-digit identification number (or catalogue number) of the sequence. Each sequence gets a unique A-number.

Here are two artificial examples, to illustrate the format used in the table (the abbreviations are explained below):

## A simple example:

```
%I A007299
%S A007299 1,1,1,5,3,60,487
%N A007299 Hadamard matrices of order 4n.
%D A007299 M. Jones, The Catalan numbers, Amer. Math. Monthly, Vol. 256 (1939), pp. 1444-1578.
%K A007299 nonn,easy,more
%F A007299 a(n) = n^4 + 3*n.
%O A007299 1,4
%A A007299 Jane Smith (jsmith(AT)math.www.edu)
```

## A more complicated example:

```
%I A000112 M1495 N0588
%S A000112 1,1,2,5,16,63,318,2045,16999,183231,2567284,46749427,1104891746,
%T A000112 33823827452
%N A000112 Partially ordered sets ("posets") with n elements.
%C A000112 A comment explaining the definition would go here.
%D A000112 A. Jones, Title of paper, Amer. Math. Monthly, vol. 21, pp. 100-120, 1991.
%D A000112 A. Jones, Further results on Euler's problem, preprint, 2002.
%H A000112 P. J. Cameron, <a href="http://www.math.uwaterloo.ca/JIS/index.html#P00.1.5">Sequences realized by oligomorphic permutation groups</a>, J. Integ. Seqs. Vol. 3 (2000), #00.1.5.
%H A000112 David Rusin, <a href="http://www.math.niu.edu/~rusin/known-math/94/finite.top">Finite Topologies</a>
%H A000112 E. W. Weisstein, <a href="http://mathworld.wolfram.com/HarshadNumber.html">Harshad Numbers</a>
%O A000112 0,3
%F A000112 a(n) = n^4 + 3*n.
%K A000112 nonn,hard,core
%p A000112 (n^2+n+3)*(n+29);
%Y A000112 Cf. A000798 (labeled topologies).
%Y A000112 Sequence in context: A022494 A079566 A059685 this_sequence A003149 A027046 A000522
%Y A000112 Adjacent sequences: A000109 A000110 A000111 this_sequence A000113 A000114 A000115
%A A000112 N. J. A. Sloane
```

## Further Examples and Style Guide:

Use "Back" to return to this page.

- **A simple formula:** [A051925](https://oeis.org/A051925). Note that the running variable is normally referred to as n.
- **A simple recurrence:** [A046699](https://oeis.org/A046699). Note that the nth term is usually referred to as a(n).
- **Numbers with some particular property:** [A051915](https://oeis.org/A051915). Here the typical term is usually referred to as n.
- **Primes of a certain form:** [A045637](https://oeis.org/A045637).
- **An example with many comments, references and links:** [A000108](https://oeis.org/A000108).
- **A partition function:** [A000837](https://oeis.org/A000837). Note that the words "The number of" are normally omitted if clear from the context.
- **A sequence with signs:** [A000594](https://oeis.org/A000594).
- **A continued fraction:** [A002852](https://oeis.org/A002852). It is customary to give the decimal expansion as a separate sequence, cross-referenced in a %Y line, and to give the beginning of the decimal expansion in a %e line.
- **Decimal expansion of an important constant (as a sequence of single digits):** [A001620](https://oeis.org/A001620). It is customary to also give the true decimal expansion in a %e line. If possible also give the continued fraction for the number as a separate sequence, cross-referenced in a %Y line, and to give the beginning of the decimal expansion in a %e line.
- **Theta series of a lattice:** [A004009](https://oeis.org/A004009). Note that if every other term is zero, then it is customary to omit these zeros.
- **Weight distribution of a code:** [A010463](https://oeis.org/A010463). Note that if a(2k+1) is always zero, or a(4k+i) is always zero for i != 0, etc) is zero, then it is customary to omit these zeros.
- **An example with a complicated Maple program:** [A000022](https://oeis.org/A000022).
- **A triangle of numbers read by rows:** [A008277](https://oeis.org/A008277). Note that the typical entry in the array is usually denoted by T(n,k) or T(n,m).
- **A square array of numbers read by antidiagonals:** [A003987](https://oeis.org/A003987). Note that the typical entry in the array is usually denoted by T(n,k) or T(n,m).
- **A sequence of fractions:** Normally produces a pair of sequences, one giving the numerators (possibly with signs), and the other the denominators. They have keyword "frac" and are cross-referenced to each other in %Y lines. For example: [A000367](https://oeis.org/A000367) and [A002445](https://oeis.org/A002445).

## For even more examples,

enter an arbitrary A-number (e.g. A005132) at <https://oeis.org/> and click "Submit". Use "Back" to return to this page.

---

# Abbreviations Used

There is a [summary](#summary-all-the-possible-lines) at the end of this file.

## %I = Identification line: <span style="color:red">Required!</span>

- Annnnnn = absolute catalogue number of sequence
- Example:

```
%I A012345
```

- When you submit a new sequence, it will automatically be assigned an A-number. You can also reserve a block of A-numbers, which is helpful if you are planning to submit a group of related sequences, so that you can cross-reference them.
- Mnnnn = number (if any) in **"The Encyclopedia of Integer Sequences"** by N.J.A. Sloane and S. Plouffe, Academic Press, San Diego, CA, 1995.
- Nnnnn = number (if any) in "Handbook of Integer Sequences", by N. J. A. Sloane, Academic Press, NY, 1973.

## %S, %T and %U lines

- The %S, %T and %U lines give the beginning of the sequence.
- **The %S line (at least) is required.**
- If possible, give enough terms to fill 3 lines on the screen. The numbers should be separated by commas, with no spaces or tabs. Don't give more than 3 lines. Label the 3 lines %S, %T and %U.
- At least 4 terms are required.
- The terms must be integers.
- If the terms are fractions, enter the numerators and denominators as separate sequences, use the keyword "frac", and put links in the %Y lines to link the two sequences together.
- The sequence should be well-defined and of general interest.
- Example (the Catalan numbers):

```
%S A000108 1,1,2,5,14,42,132,429,1430,4862,16796,58786,208012,742900,
%T A000108 2674440,9694845,35357670,129644790,477638700,1767263190,
%U A000108 6564120420,24466267020,91482563640,343059613650,1289904147324
```

## %N = Name of sequence: <span style="color:red">Required!</span>

- The %N line gives a brief description or definition of the sequence. It must fit on one line (but the line can be fairly long).
- In the description, a(n) usually denotes the n-th term of the sequence, and n is a typical subscript.
- Only one %N line may appear.
- Here are 3 separate examples:

```
%N A000108 Catalan numbers: a(n) = C(2n,n)/(n+1) = (2n)!/(n!(n+1)!).
%N A000594 Ramanujan's tau function (or tau numbers).
%N A010085 Weight distribution of Hamming code of length 15 and minimal distance 3.
```

## %D = Detailed references.

- Put each reference on a separate line.
- There may be several such lines.
- Here are 3 separate examples:

```
%D A010109 I. G. Enting, A, J. Guttmann and I. Jensen, Low-Temperature Series Expansions
   for the Spin-1 Ising Model, J. Phys. A. 27 (1994) 6987-7006.

%D A000925 A. Das and A. C. Melissinos, Quantum Mechanics: A Modern Introduction, Gordon
   and Breach, 1986, p. 47.

%D A022818 W. C. Yang (wcyang(AT)cco.caltech.edu), Derivatives of self-compositions of
   functions, preprint, 2002.
```

## %H = Links related to this sequence

- Put each link on a separate line.
- There may be several such lines.
- The lines **must** have the following form:

```
%H A036432 S. Colton, <a href="JIS/index.html#P99.1.2">
   Refactorable Numbers - A Machine Invention,</a> J. Integer Sequences, Vol. 2, 1999, #2.

%H A001371 F. Ruskey, <a href=http://www.theory.cs.uvic.ca/~cos/inf/neck/NecklaceInfo.html">Counting Necklaces</a>

%H A027414 N. J. A. Sloane, <a href="transforms.html">Transforms</a>
```

(Except that you should put the information on one long line, whereas the lines above were broken to make them fit better on the screen).

In other words, please use this format:

**`%H A012345 Author, <a href="http://www.etc.etc/file">Title</a>`**

## %F = Formula (if not included in %N line)

- a(n) usually denotes the n-th term of the sequence, and n is a typical subscript.
- The ordinary generating function (G.f.) or exponential generating function (E.g.f.) is usually denoted by A(x).
- There may be several such lines.
- Here are 3 separate examples:

```
%F A008346 G.f.: 1/(1-2*x^2-x^3).
%F A014551 a(n+1) = 2 * a(n) - (-1)^n * 3.
%N A030033 a(n+1)= Sum a(k)a(n-k), k = 0 ... [2n/3].
```

- Note that the %O line (see below) gives the initial value of n.

## %Y = Cross-references to other sequences

- Examples:

```
%Y A003485 Cf. A003484.
%Y A007295 Cf. A006546, A007104, A007203.
%Y A005282 a(n) = A025583(n)^2+1.
```

- **Sequence in context.** This line show the three sequences immediately before and after the sequence in the lexicographic listing. Example:

```
%Y A000112 Sequence in context: A022494 A079566 A059685 this_sequence A003149
                 A027046 A000522
```

- **Adjacent sequences.** This line show the three sequences whose A-numbers are immediately before and after the A-number of the sequence. Example:

```
%Y A000112 Adjacent sequences: A000109 A000110 A000111 this_sequence A000113
              A000114 A000115
```

## %A = Author, submitter or other Authority: <span style="color:red">Required!</span>

- Give your name and email address.
- Example:

```
%A A023600 Clark Kimberling (ck6(AT)evansville.edu)
```

## %O = Offset a, b : <span style="color:red">Required!</span>

- a is subscript of first term
- b gives position of first entry greater than or equal to 2 in magnitude (or 1 if no such entry exists)
- Examples: The Fibonacci numbers F(0), F(1), F(2), ... begin

```
%S A000045 0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,
```

and the 4th term is the first that is greater than 1, so here a = 0 and b = 4, and the %O line is:

```
%O A000045 0,4
```

- On the other hand, in this sequence:

```
%S A010051 0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0
%N A010051 Characteristic function of primes: 1 if n is prime else 0.
```

no term exceeds 1, so b takes its default value of 1. n starts at 0, so a = 0, and the %O line is:

```
%O A010051 0,1
```

## %p, %t, %o = Computer program to produce the sequence

- %p = Maple
- %t = Mathematica
- %o = other computer language
- There may be several such lines, and the lines may be long.
- Examples:

```
%p A010051 f:=i->if isprime(i) then 1 else 0; fi; [seq(f(i),i=0..100)];
%p A008334 for i from 1 to 100 do if isprime(i) then print(nops(factorset(i-1))); fi; od;
%t A011773 Table[If[n==1,1,LCM@@Map[ (#1[1]]-1)*#1[1]]^(#1[2]]-1)&, FactorInteger[n]]],{n,1,70}]
%o A002837 (PARI) v=[];for(n=0,60,if(isprime(n^2+n+41),v=concat(v,n),));v
%o A006006 (MAGMA) R := ReedMullerCode(2,7); print(WeightEnumerator(R));
```

## %E = Extensions and Errors

- Notes about sequences that have been significantly extended, etc.
- Also significant errors in the [book](https://oeis.org/book.html) or in the source.
- Examples:

```
%E A007097 15th term corrected by loria.fr!Paul.Zimmermann (Paul Zimmermann).
%E A010334 There is a typo at the n=6 term in the printed version of the paper.
```

## %e = examples

- Expanded information or examples to illustrate the initial terms of the sequence.
- If the sequence is the coefficients of a power series, the %e line can be used to show the beginning of the series.
- If the sequence is formed by reading the rows of an array, the %e line can show the beginning of the array (see the keyword "tabl" below.)
- Examples:

```
%e A002654 4=2^2, so a(4)=1; 5=1^2+2^2=2^2+1^2, so a(5)=2.
%e A027824 1+3600*q^3+101250*q^4+...
%e A007318 {1}; {1,1}; {1,2,1}; {1,3,3,1}; {1,4,6,4,1}; ...
```

## %K = Keywords: <span style="color:red">Required!</span>

- At the very least, indicate if the terms are all nonnegative ("nonn") or if there are negative numbers ("sign").
- **base**: dependent on base used for sequence
- **bref**: sequence is too short to do any analysis with
- **cofr**: a continued fraction expansion of a number
- **cons**: a decimal expansion of a number
- **core**: an important sequence
- **dead**: an erroneous sequence
- **dumb**: an unimportant sequence
- **dupe**: duplicate of another sequence
- **easy**: it is very easy to produce terms of sequence
- **eigen**: an **eigensequence**: a fixed sequence for some transformation - see the files [transforms](https://oeis.org/transforms.html) and [transforms (2)](https://oeis.org/transforms2.html) for further information.
- **fini**: a finite sequence
- **frac**: numerators or denominators of sequence of rationals
- **full**: the full sequence is given
- **hard**: next term not known, may be hard to find. Would someone please extend this sequence?
- **less**: reluctantly accepted
- **more**: more terms are needed! would someone please extend this sequence?
- **mult**: Multiplicative: a(mn)=a(m)a(n) if g.c.d.(m,n)=1
- **new**: New (added within last two weeks, roughly)
- **nice**: an exceptionally nice sequence
- **nonn**: a sequence of nonnegative numbers
- **obsc**: obscure, better description needed
- **sign**: sequence contains negative numbers
- **tabf**: An irregular (or funny-shaped) array of numbers made into a sequence by reading it row by row
- **tabl**: typically a triangle of numbers, such as Pascal's triangle, made into a sequence by reading it row by row.
- **uned**: Not edited. Incoming sequences are normally edited to check that:
  - the sequence is worth including
  - the definition is sensible
  - the sequence is not already in the database
  - the English is correct
  - the different parts of the entry all have the correct prefixes: cross-references are in %Y lines, formulae in %F lines, etc.
  - any %H lines are correctly formatted (this is easy to get wrong)
  - etc.

  The keyword "uned" indicates that this sequence was not edited, usually because of time pressure. Perhaps someone could edit this sequence and email the result.
- **unkn**: little is known; an unsolved problem; anyone who can find a formula or recurrence is urged to let the editors know.
- **walk**: counts walks (or self-avoiding paths)
- **word**: depends on words for the sequence in some language
- Examples:

```
%K A029403 nonn
%K A002654 core,easy,nonn
%K A024022 sign
```

## %C = Comments

- Use this if you have a comment which does not fit into any of the other categories. Often used to give a more precise definition of the sequence, or to explain an unfamiliar word.
- Examples:

```
%C A002324 The hexagonal lattice is the familiar 2-dim. lattice in which each
point has 6 neighbors. This is sometimes called the triangular lattice.
%C A039997 a(n) counts substrings of digits of n which denote primes.
%C A046810 An anagram of a k-digit number is one of the k! permutations of the
digits that does not begin with 0.
```

---

## SUMMARY: all the possible lines:

```
%I A000001 Identification line (required)
%S A000001 First line of sequence (required)
%T A000001 2nd line of sequence.
%U A000001 3rd line of sequence.
%N A000001 Name (required)
%D A000001 Detailed reference line.
%D A000001 Detailed references (2).
%H A000001 Link to other site.
%H A000001 Link to other site (2).
%F A000001 Formula.
%F A000001 Formula (2).
%Y A000001 Cross-references to other sequences.
%A A000001 Author (required)
%O A000001 Offset (required)
%E A000001 Extensions, errors, etc.
%e A000001 examples to illustrate initial terms.
%p A000001 Maple program.
%t A000001 Mathematica program.
%o A000001 Program in another language.
%K A000001 Keywords (required)
%C A000001 Comments.
```

---

*Source: <https://oeis.org/eishelp1.html> — Maintained by The OEIS Foundation Inc.*
