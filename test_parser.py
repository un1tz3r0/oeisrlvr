"""Test harness: inspect one parse, then run coverage over the whole dataset."""

import glob
import json
import sys
import time
import traceback
from collections import Counter

from oeis_parser import parse_file

GLOB = "oeisdata/seq/A???/A??????.seq"


def show(path: str) -> None:
    seq = parse_file(path)
    print(json.dumps(seq.to_dict(), indent=2, ensure_ascii=False, default=str))


def coverage(limit: int | None) -> None:
    paths = sorted(glob.glob(GLOB))
    if limit:
        paths = paths[:limit]
    n = len(paths)
    print(f"parsing {n} files...")

    ok = 0
    errors: list[tuple[str, str]] = []
    field_present = Counter()
    prog_langs = Counter()
    kw = Counter()
    no_data = no_name = no_kw = no_offset = no_author = 0
    attr_on = Counter()
    t0 = time.time()

    for i, p in enumerate(paths):
        try:
            s = parse_file(p)
        except Exception as e:  # noqa: BLE001
            errors.append((p, f"{type(e).__name__}: {e}"))
            continue
        ok += 1
        if s.data:
            field_present["data"] += 1
        else:
            no_data += 1
        if s.name:
            field_present["name"] += 1
        else:
            no_name += 1
        if s.keywords:
            field_present["keywords"] += 1
        else:
            no_kw += 1
        if s.offset:
            field_present["offset"] += 1
        else:
            no_offset += 1
        if s.author:
            field_present["author"] += 1
        else:
            no_author += 1
        for k in s.keywords:
            kw[k] += 1
        for prog in s.programs:
            prog_langs[prog.language] += 1
        attr_on["comments"] += sum(1 for c in s.comments if c.attribution)
        attr_on["formulas"] += sum(1 for f in s.formulas if f.attribution)
        attr_on["programs"] += sum(
            1 for pr in s.programs + s.maple + s.mathematica if pr.attribution
        )
        if i and i % 50000 == 0:
            print(f"  ...{i}/{n}")

    dt = time.time() - t0
    print(f"\nparsed OK: {ok}/{n}  errors: {len(errors)}  ({dt:.1f}s, {n/dt:.0f}/s)")
    print(f"missing required-ish fields: data={no_data} name={no_name} "
          f"keywords={no_kw} offset={no_offset} author={no_author}")
    print(f"attributions separated: {dict(attr_on)}")
    print(f"top program langs: {prog_langs.most_common(12)}")
    print(f"unexpected keywords: "
          f"{[(k, c) for k, c in kw.most_common() if k not in KNOWN_KW]}")
    if errors:
        print(f"\nfirst {min(15, len(errors))} errors:")
        for p, e in errors[:15]:
            print(f"  {p}: {e}")


KNOWN_KW = {
    "base", "bref", "changed", "cofr", "cons", "core", "dead", "dumb", "dupe",
    "easy", "eigen", "fini", "frac", "full", "hard", "hear", "less", "look",
    "more", "mult", "new", "nice", "nonn", "obsc", "probation", "sign", "tabf",
    "tabl", "uned", "unkn", "walk", "word", "allocated", "recycled", "smpc",
}


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "show":
        show(sys.argv[2])
    else:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
        coverage(limit)
