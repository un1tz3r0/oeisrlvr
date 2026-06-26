# Explanation of Terms Used in OEIS Sequences

## [The On-Line Encyclopedia of Integer Sequences](https://oeis.org/)

The following (imaginary) example shows all the different types of lines that may appear in a reply from the **On-Line Encyclopedia of Integer Sequences**.

[For a description of the **Internal Format** used in the database, click [here](https://oeis.org/eishelp1.html).]

Click on the heading to get more information.

```
ID Number: A004001 (Formerly M0276 and N0101))
Data:      1,-1,-1,0,0,1,1,2,0,0,-1,-1,-2,-2,-2,-4,-1,-2,0,0,3,4,6,6,8,8,10,10,9,9,7,
           5,2,0,-7,-10,-18,-22,-29,-32,-41,-43,-49,-50,-54,-53,-54,-50,-46,-38,-30,
           -18,-6,8,25,43,62,82,108,129,155
Name:      Bell or exponential numbers: ways of placing n labeled balls into n indistinguishable boxes.
Comments:  On first day, each gossip has his own tidbit. On each successive day, disjoint pairs of
           gossips may share tidbits (over the phone). After a(n) days, all gossips have all
           tidbits.
References R. L. Graham, D. E. Knuth and O. Patashnik, Concrete Mathematics. Addison-Wesley, Reading, MA, 2nd Ed., 1998, p. 329.
           C. L. Mallows, Conway's challenge sequence, Amer. Math. Monthly, 98 (1991), 5-20.
Links:     Douglas E. Iannucci and Donna Mills-Taylor, On Generalizing the Connell Sequence, J. Integer Sequences, Vol. 2, 1999, #99.1.7.
Formula:   a(n) = (1/4)*n^2*(n^2+3).
Example:   a(24) = 4 because we can form 2, 4, 24 and 42.
Maple:     a:=proc(n) option remember; if n<=2 then 1 else a(a(n-1))+a(n-a(n-1)); fi; end;
Math'ca:   dtn[L_]:=Fold[2#1+#2&,0,L]; f[n_]:=dtn[Reverse[1-IntegerDigits[n,2]]];
           Table[f[n],{n,0,100}]
Program:   (PARI) direuler(p=2,101,1/(1-(kronecker(5,p)*(X-X^2))-X))
See also:  Cf. A005380
Keywords:  sign,nice,easy
Offset:    5,8
Author(s): Antti Karttunen, Dec 05 2001
Extension: Extended by Alois P. Heinz, Mar 10, 2010.
```

## Explanation of the Different Lines

### ID Number

- The A-number (for example [A000031](https://oeis.org/A000031)) is the absolute catalogue number of the sequence. It consists of A followed by 6 digits.
- Some sequences also have a 4-digit M-number, such as M1459, which is the number they carried in **"The Encyclopedia of Integer Sequences"** by N.J.A. Sloane and S. Plouffe, Academic Press, San Diego, CA, 1995.
- Some older sequences also have a 4-digit N-number, such as N0577, which is the number they carried in the "Handbook of Integer Sequences", by N. J. A. Sloane, Academic Press, NY, 1973.

### Data

- These lines give the beginning of the sequence.
  - For example: `0,1,1,2,3,5,8,13,21,34,55,89,144,...`
- Ideally the entry gives enough terms to fill about three lines on the screen.
- The terms must be integers.
- If the terms are fractions, then the numerators and denominators appear as separate sequences, labeled with the [Keyword](#keywords) "frac", and with links connecting the two sequences.
- Only sequences that are well-defined and of general interest are included.

### Name

- The "Name" line gives a brief description or definition of the sequence.
  - For example: `The even numbers.`
- In the description, a(n) usually denotes the n-th term of the sequence, and n is a typical subscript.
  - For example: `a(n) = a(n-1) + a(n-3)`.
- In some cases however a letter such as k or m is used to denote a typical term in the sequence.
  - For example: `List of numbers k such that k and k+1 have the same number of divisors`.

### Comments

- Additional remarks about the sequence that do not fit into any of the other lines (additional contexts where the sequence occurs, for instance).

### References

- References where information about the sequence can be found.
- Whenever possible the reference gives full bibliographical information:
  - For an article in a journal: author(s), title of article, name of journal, volume, issue number if relevant, year, starting and ending page numbers, etc.
  - For a book: author(s), title, publisher, place, year, edition, page numbers where sequence appears, etc.
  - For an article in a book: author(s), title of article, page numbers, editors' names, title of book, publisher, place, year, etc.

### Links

- Links related to this sequence
- Preferred format:

  `J. B. Smith, < a href = " http : // www.this.that.com/etc/etc.html ">Title< /a >`

  - spaces have been inserted to make it visible, but of course you should not insert any spaces.

  In other words, the format is

  **`Author, <a href="http://www.etc.etc/file">Title</a>`**
- Web page addresses can change very quickly, so if you find a link that is broken, please add a comment to that link — you might say, for example, [broken link? - ~~~~] (the four tildes will be transformed into your signature).

### Formula

- These lines give formulae, recurrences, generating functions, etc. for the sequence.
- a(n) usually denotes the n-th term of the sequence, and n is a typical subscript.
- Note that the first offset (see [Offset](#offset)) then gives the value of n corresponding to the first term shown.
  - An example of an explicit formula: `a(n) = n^2 + n + 1.`
  - An example of a recurrence: `a(n+1) = 2 * a(n) - (-1)^n * 3.`
- The ordinary generating function (G.f.) for a sequence a(0), a(1), a(2), ... is the formal power series

  ```
          A(x) = a(0) + a(1)*x + a(2)*x^2 + a(3)*x^3 + ...
  ```

  - An example of an ordinary generating function: `G.f.: A(x) = 1/(1-x)^4.`
  - Usually one can think of an ordinary generating function as a Taylor series, and extract the nth coefficient by differentiating A(x) n times, setting x = 0, and dividing by n!. Computer algebra languages such as Maple make this easy - one simply says (for example) series(A,x,100).
- The exponential generating function (E.g.f.) for a sequence a(0), a(1), a(2), ... is the formal power series

  ```
                  a(0)   a(1)*x   a(2)*x^2   a(3)*x^3   a(4)*x^4
          A(x) =  ---- + ------ + -------- + -------- + -------- + ...
                   1       1         2          6          24
  ```

  where the numbers in the denominators are the factorial numbers n! = 1*2*3*4*...*n, Sequence [A000142](https://oeis.org/A000142).

  - An example of an exponential generating function: `E.g.f.: A(x) = exp(exp(x)-1).`

### Example

- These lines give expanded information or examples to illustrate the initial terms of the sequence.
  - For instance: `4=2^2, so a(4)=1;   5=1^2+2^2=2^2+1^2, so a(5)=2.`
- If the sequence is formed from the coefficients of a power series, this line can be used to show the beginning of the series.
  - For instance: `1+3600*q^3+101250*q^4+...`
- If the sequence is formed from the decimal expansion or continued fraction expansion of a real number, this line may show the actual decimal expansion.
  - For instance: `3.141592653589793238462643383279502884...`
- If the sequence is formed by reading the rows of an [array](#arrays), this line may show the beginning of the array (see the [Keywords](#keywords) "tabl" and "tabf" below.)
  - For instance: `{1}; {1,1}; {1,2,1}; {1,3,3,1}; {1,4,6,4,1}; ...`

### Maple

- These lines give Maple code to produce the sequence. Examples:
  - `f:=i->if isprime(i) then 1 else 0; fi; [seq(f(i),i=0..100)];`
  - `for i from 1 to 100 do if isprime(i) then print(nops(factorset(i-1))); fi; od;`

### Math'ca

- These lines give Mathematica code to produce the sequence. For example:
  - `Table[ If[ n==1,1,LCM@@Map[ (#1[ [ 1 ] ]-1)*#1[ [ 1 ] ]^(#1[ [ 2 ] ]-1)&, FactorInteger[ n ] ] ],{n,1,70} ]`

### Program

- These lines give a program in some other language that will produce the sequence. Examples:
  - `(PARI) v=[];for(n=0,60,if(isprime(n^2+n+41),v=concat(v,n),));v`
  - `(Magma) R := ReedMullerCode(2,7); print(WeightEnumerator(R));`
  - `(SageMath) CuspForms( Gamma1(1), 12, prec=100).0;`

### See also

- These lines gives cross-references to related sequences. Examples:
  - `Cf. A006546, A007104, A007203.`
  - `a(n) = A025582(n)^2+1.`

- **Sequence in context.** This line show the three sequences immediately before and after the sequence in the lexicographic listing. Example:
  - `Sequence in context: A036656 A000055 A006787 this_sequence A036648 A047750 A072187`

- **Adjacent sequences.** This line show the three sequences whose A-numbers are immediately before and after the A-number of the sequence. Example:
  - `Adjacent sequences: A000989 A000990 A000991 this_sequence A000993 A000994 A000995`

### Keywords

These lines give keywords describing the sequence. At present the following keywords are in use.

- **base**: Sequence is dependent on base used
- **bref**: Sequence is too short to do any analysis with
- **changed**: A sequence that was changed in the last two or three weeks (this keyword is set automatically)
- **cofr**: A continued fraction expansion of a number
- **cons**: A decimal expansion of a number
- **core**: An important sequence
- **dead**: An erroneous or duplicated sequence (the table contains a number of incorrect sequences that have appeared in the literature, with pointers to the correct versions)
- **dumb**: An unimportant sequence
- **dupe**: Duplicate of another sequence
- **easy**: It is easy to produce terms of this sequence
- **eigen**: An **eigensequence**: a fixed sequence for some transformation - see the files [transforms](https://oeis.org/transforms.html) and [transforms (2)](https://oeis.org/transforms2.html) for further information.
- **fini**: A finite sequence
- **frac**: Numerators or denominators of sequence of rational numbers
- **full**: The full sequence is given, either in the DATA section or in the b-file (implies the sequence is finite and has keyword "fini")
- **hard**: Next term is not known and may be hard to find. Would someone please extend this sequence?
- **hear**: A sequence worth listening to.
- **less**: This is a less interesting sequence and is less likely to be the one you were looking for.
- **look**: A sequence with an interesting graph.
- **more**: More terms are needed! Would someone please extend this sequence? We need enough terms to fill about three lines on the screen.
- **mult**: Multiplicative: a(mn)=a(m)a(n) if g.c.d.(m,n)=1
- **new**: New (added or modified within last two weeks, roughly; this keyword is set automatically)
- **nice**: An exceptionally nice sequence
- **nonn**: A sequence of nonnegative numbers (more precisely, all the displayed terms are nonnegative; it is not excluded that later terms in the sequence become negative)
- **obsc**: Obscure, better description needed
- **probation**: Included on a provisional basis, but may be deleted later at the discretion of the editor.
- **sign**: Sequence contains negative numbers
- **tabf**: An irregular or funny-shaped [triangle](#arrays) of numbers (one in which the n-th row does not contain n terms) made into a sequence by reading it by rows; or a table with a fixed number of columns that are read by rows — a list of pairs, triples, quadruples, etc. Any 2- or 3-D sequence that does not warrant the keyword "tabl". See [A028297](https://oeis.org/A028297) and [A027113](https://oeis.org/A027113) for examples.
- **tabl**: A regular [triangle](#arrays) of numbers (one in which the n-th row contains n terms) made into a sequence by reading it by rows; or an infinite [square](#arrays) array T(n,k), n >= 0, k >= 0, say, made into a sequence by reading it by antidiagonals either upwards or downwards. See [A007318](https://oeis.org/A007318) and [A003987](https://oeis.org/A003987) for examples.
- **uned**: Not edited. The editors normally check all incoming sequences to make sure that:
  - the sequence is worth including
  - the definition is sensible
  - the sequence is not already in the database
  - the English is correct
  - the different parts of the entry all have the correct prefixes: cross-references are in %Y lines, formulae in %F lines, etc.
  - any %H lines are correctly formatted (this is easy to get wrong)
  - etc.

  The keyword "uned" indicates that this sequence needs editing. If you can help by editing the entry, please do so!
- **unkn**: Little is known; an unsolved problem; anyone who can find a formula or recurrence is urged to add it to the entry.
- **walk**: Counts walks (or self-avoiding paths)
- **word**: Depends on words for the sequence in some language

For further information about the keywords, see [here](https://oeis.org/wiki/User:Charles_R_Greathouse_IV/Keywords).

### Offset

- The offset line contains two numbers.
- The first offset usually gives the subscript of the first term in the sequence, or is 1 if the sequence is a list.
  - For example: the [Fibonacci numbers](https://oeis.org/A000045) [A000045](https://oeis.org/A000045), F(0), F(1), F(2), ..., begin `0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89,...` and the subscript of the initial term is 0, so the first offset is **0**
  - On the other hand, the sequence of [Evil numbers](https://oeis.org/A001969) (numbers with an even number of 1's in their binary expansion): `0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, ...,` is a list, so the first offset is **1**
- However, if the sequence gives the decimal expansion of a constant, the offset is the number of digits before the decimal point.
  - For example, the speed of light is `299792458 (m/sec),` giving the sequence `2,9,9,7,9,2,4,5,8,` and so the first offset is **9**.
  - On the other hand, the decimal expansion of 1/13 is `.07692307692307692...`, givng the sequence `0, 7, 6, 9, 2, 3, 0, 7, 6, 9, 2, 3, 0, 7, 6, 9, 2,` and now the first offset is **0**.
- The second offset, that is, the second number in the Offset line, is automatically set by the system, and is used to determine where this sequence lies in the lexicographic order.

  The normal user need not worry about the second offset, but for those who are interested, here are the details.

  The second offset says which term (counting from the left, and labeling the first term with 1), first exceeds 1 in absolute value. It is set to 1 if all the terms are 0 or +-1.

  The reason for this goes back to the very beginning of the database.

  All the sequences in the database are (virtually) arranged lexicographically, but to do that they have to be lined up, and they are lined up so the first field they are sorted on is the first term >= 2 in magnitude.

  You cannot compare the Fibonacci numbers and the primes unless you line them up properly:

  ```
  0 1 1 2 3 5 8 13 ...
  2 3 5 7 11 13 ...
  ```

  You mark the first term > 1 in magnitude with asterisks (this position is the second offset):

  ```
  0 1 1 *2* 3 5 8 13 21 ...  (second offset is 4)
        *2* 3 5 7 11 13 ...  (second offset is 1)
  ```

  and now you can compare them. They are out of order! The correct order is

  ```
  2 3 5 7 11 13 ...
  0 1 1 2 3 5 8 13 ...
  ```

  The primes come before the Fibonacci numbers.

  You can see this arrangement if you look at the line in any entry that says "Sequence in context" That shows you the three sequences before and after the one you are looking at, in the lexicographic order.

  However, if all terms are less than 2 in magnitude, you can't sort them like this. So those get sorted on the first field — these sequences have second offset equal to 1.

### Author

- This line gives the name of the person or persons who contributed the sequence. For example: `Clark Kimberling`. This name will usually be an active link to the user page of the submitter on the OEIS wiki.

### Extension

- **E** stands for **Extensions**, **Errors** or **Edited**. These lines contain information about sequences that have been significantly extended, errors that have been corrected, or entries in the database that have been edited by someone.
- The errors might be in an earlier version of the entry in the database or in the published literature.
- Examples:
  - `Corrected and extended by Henry Bottomley, Jan 01 2002`
  - `The sixth term is incorrect in the book by Smith and Jones.`
  - `Edited by Michel Marcus, Jan 02 2019`

### Arrays

- The database also contains a number of sequences based on triangular or square arrays, such as **Pascal's Triangle**:

  ```
              1
            1   1
          1   2   1
        1   3   3   1
      1   4   6   4   1
    1   5  10  10   5   1
    ...  ...  ...  ...  ...  ...
  ```

  When read by rows this produces the sequence `1, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 4, 6, 4, 1, ...,` Sequence [A007318](https://oeis.org/A007318).
- Square arrays are usually read by anti-diagonals. For example, the **Nim-addition table**:

  ```
  0 1 2 3 4 5
  1 0 3 2 5 4
  2 3 0 1 6 7
  3 2 1 0 7 6
  4 5 6 7 0 1
  . . . . . .
  ```

  when read by anti-diagonals produces the sequence `0, 1, 1, 2, 0, 2, 3, 3, 3, 3, 4, 2, 0, 2, 4, ...,` Sequence [A003987](https://oeis.org/A003987).
- The typical term in these arrays is usually denoted by T(n,k) (or sometimes A(n,k)).
- The [Example](#example) lines for these sequences usually show the beginning of the two-dimensional array.
- These sequences are usually indicated by the keywords **tabl** or **tabf**.
- Some ordinary (one-dimensional) sequences also have the keyword **tabl**, indicating that they can also be regarded as arrays.

---

*Source: <https://oeis.org/eishelp2.html> — Maintained by The OEIS Foundation Inc.*
