"""Parser for the OEIS internal sequence file format (``A??????.seq``).

The format is documented at https://oeis.org/eishelp1.html (internal format)
and https://oeis.org/eishelp2.html (beautified format). This module turns one
``.seq`` file into a structured :class:`Sequence`, separating field content from
attributions, parsing links, splitting multi-language programs, and extracting
cross-references and ``_Name_`` author signatures.

Heuristic boundaries (documented, not silent):
  * ``%o`` programs are split reliably on a leading ``(Language)`` tag.
  * ``%t``/``%p`` have no language tag; snippets are split on a trailing
    ``(* _Author_, date *)`` credit or a standalone separator comment. This is
    best-effort -- see :func:`_split_tagless_programs`.
  * Trailing attribution stripping requires either a known comment/dash opener
    or a trailing date, to avoid eating code that merely contains underscores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# --- shared regexes ---------------------------------------------------------

# A line of the internal format: ``%X Annnnnn <content>``. Content may be empty.
_LINE_RE = re.compile(r"^%(?P<code>.)\s+(?P<anum>A\d{6})(?: (?P<content>.*))?$")

# An OEIS sequence reference appearing in free text.
_SEQREF_RE = re.compile(r"\bA\d{6}\b")

# An OEIS-linked author signature: _First M. Last_. The lookarounds keep it from
# matching underscore math subscripts (u_1, u_{n-1}): the opening underscore must
# not follow an identifier char, the body must start with a capital, and the
# closing underscore must not precede one.
_SIGNATURE_RE = re.compile(r"(?<![A-Za-z0-9])_([A-Z][^_]*?)_(?![A-Za-z0-9])")

_MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
_DATE = rf"(?:{_MONTHS})[a-z]*\.?\s+\d{{1,2}},?\s+\d{{4}}"

# Trailing attribution credit. We anchor at end of string and require either a
# recognized opener (dash or a language comment marker) OR a trailing date, so
# that we do not strip code/identifiers that merely contain underscores.
_ATTR_RE = re.compile(
    rf"""
    \s*
    (?P<opener> -\ |\\\\\ |//\ |\#\ |--\ |/\*\ |\(\*\ |;\ )   # comment/dash opener
    (?P<attr>
        (?:by\ |from\ |after\ )?
        _[^_]+_                # at least one signature
        [^*]*?                 # rest of credit (authors, dates, "after _X_")
    )
    \s*
    (?:\*\)|\*/)?              # optional comment closer
    \s*$
    """,
    re.VERBOSE,
)

# A %o language tag at the very start of a program line: ``(PARI)``, ``(Magma)``.
_LANG_TAG_RE = re.compile(r"^\((?P<lang>[^)]+)\)\s?(?P<rest>.*)$")

# A %H link: ``Author, <a href="URL">Title</a> suffix``.
_LINK_RE = re.compile(
    r'^(?:(?P<author>.*?),\s*)?<a href="?(?P<url>[^">]+)"?>'
    r"(?P<title>.*?)</a>\s*(?P<suffix>.*)$",
    re.DOTALL,
)


# --- data model -------------------------------------------------------------


@dataclass
class AnnotatedText:
    """A field item: body text with attribution and references separated out."""

    text: str
    attribution: Optional[str] = None
    signatures: list[str] = field(default_factory=list)
    seq_refs: list[str] = field(default_factory=list)


@dataclass
class Link:
    """A parsed ``%H`` link line."""

    author: Optional[str]
    url: str
    title: str
    suffix: Optional[str]
    seq_refs: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class Program:
    """A single program in one language, with its credit separated out."""

    language: str          # 'Maple', 'Mathematica', or the %o (Lang) tag
    code: str
    attribution: Optional[str] = None


@dataclass
class Identification:
    anum: str
    mnum: Optional[str] = None       # number in the 1995 Encyclopedia
    nnum: Optional[str] = None       # number in the 1973 Handbook
    revision: Optional[int] = None   # OEIS edit revision (#NNN)
    timestamp: Optional[str] = None  # last-modified timestamp


@dataclass
class Offset:
    first: int                       # subscript of the first term
    second: Optional[int] = None     # position of first |term| >= 2


@dataclass
class CrossRefs:
    items: list[AnnotatedText] = field(default_factory=list)
    in_context: list[str] = field(default_factory=list)   # lexicographic neighbors
    adjacent: list[str] = field(default_factory=list)      # A-number neighbors


@dataclass
class Sequence:
    anum: str
    identification: Optional[Identification] = None
    data: list[int] = field(default_factory=list)
    name: Optional[str] = None
    comments: list[AnnotatedText] = field(default_factory=list)
    references: list[AnnotatedText] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    formulas: list[AnnotatedText] = field(default_factory=list)
    examples: list[AnnotatedText] = field(default_factory=list)
    maple: list[Program] = field(default_factory=list)
    mathematica: list[Program] = field(default_factory=list)
    programs: list[Program] = field(default_factory=list)
    crossrefs: CrossRefs = field(default_factory=CrossRefs)
    keywords: list[str] = field(default_factory=list)
    offset: Optional[Offset] = None
    author: Optional[AnnotatedText] = None
    extensions: list[AnnotatedText] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- helpers ----------------------------------------------------------------


def _signatures(text: str) -> list[str]:
    return [m.group(1).strip() for m in _SIGNATURE_RE.finditer(text)]


def _seq_refs(text: str) -> list[str]:
    # dict.fromkeys: dedupe while preserving order
    return list(dict.fromkeys(_SEQREF_RE.findall(text)))


def _strip_attribution(text: str) -> tuple[str, Optional[str]]:
    """Split a trailing author/date credit off ``text``.

    Returns ``(body, attribution)`` where ``attribution`` is ``None`` if no
    trailing credit was found.
    """
    m = _ATTR_RE.search(text)
    if not m:
        return text.strip(), None
    attr = m.group("attr").strip()
    # Require a date or a signature for it to count as a credit (the opener
    # alone, e.g. a stray " - ", is not enough).
    if "_" not in attr and not re.search(_DATE, attr):
        return text.strip(), None
    body = text[: m.start()].rstrip()
    return body.strip(), attr


def _annotate(text: str) -> AnnotatedText:
    body, attr = _strip_attribution(text)
    return AnnotatedText(
        text=body,
        attribution=attr,
        signatures=_signatures(text),
        seq_refs=_seq_refs(body),
    )


def _group_items(lines: list[str]) -> list[str]:
    """Join physical lines into logical items.

    Each line is its own item, except a ``... (Start)`` line starts a block that
    continues (joined by newlines) until a line containing ``(End)``.
    """
    items: list[str] = []
    buf: list[str] | None = None
    for line in lines:
        if buf is not None:
            if "(End)" in line:
                buf.append(re.sub(r"\s*\(End\)\s*$", "", line))
                items.append("\n".join(s for s in buf if s))
                buf = None
            else:
                buf.append(line)
        elif re.search(r"\(Start\)\s*$", line):
            buf = [re.sub(r"\s*\(Start\)\s*$", "", line)]
        else:
            items.append(line)
    if buf is not None:  # unterminated (Start) -- keep what we have
        items.append("\n".join(s for s in buf if s))
    return items


# --- program splitting ------------------------------------------------------


def _is_separator(line: str) -> bool:
    """A standalone comment line that separates snippets (e.g. ``(* Alternative: *)``)."""
    s = line.strip()
    return bool(re.fullmatch(r"\(\*.*\*\)", s)) and "_" not in s


def _split_programs(lines: list[str], default_lang: str, tagged: bool) -> list[Program]:
    """Split program lines into individual :class:`Program` objects.

    ``%o`` (``tagged=True``) starts a new program at each leading ``(Language)``
    tag. In every case a line carrying a trailing ``_Author_`` credit ends the
    current snippet, and a standalone separator comment starts a new one -- this
    also separates two same-language programs that share one tag block.
    """
    programs: list[Program] = []
    cur_lang = default_lang
    buf: list[str] = []

    def flush():
        if not buf:
            return
        code, attr = _strip_attribution("\n".join(buf).rstrip())
        if code:
            programs.append(Program(language=cur_lang, code=code, attribution=attr))

    for line in lines:
        if tagged:
            m = _LANG_TAG_RE.match(line)
            if m:
                flush()
                buf = []
                cur_lang = m.group("lang").strip()
                rest = m.group("rest")
                if rest:
                    buf.append(rest)
                    if _strip_attribution(rest)[1]:  # one-line tagged program
                        flush()
                        buf = []
                continue
        if _is_separator(line):  # pure comment separator; carries no code
            flush()
            buf = []
            continue
        buf.append(line)
        if _strip_attribution(line)[1]:  # credit terminates this snippet
            flush()
            buf = []
    flush()
    return programs


# --- per-field parsing ------------------------------------------------------


def _parse_identification(content: str, anum: str) -> Identification:
    ident = Identification(anum=anum)
    tokens = content.split()
    rest: list[str] = []
    for tok in tokens:
        if re.fullmatch(r"M\d+", tok):
            ident.mnum = tok
        elif re.fullmatch(r"N\d+", tok):
            ident.nnum = tok
        elif re.fullmatch(r"#\d+", tok):
            ident.revision = int(tok[1:])
        else:
            rest.append(tok)
    if rest:
        ident.timestamp = " ".join(rest)
    return ident


def _parse_data(lines: list[str]) -> list[int]:
    joined = "".join(lines)  # S/T/U continuation lines concatenate directly
    return [int(t) for t in joined.split(",") if t.strip()]


def _parse_offset(content: str) -> Offset:
    parts = [p.strip() for p in content.split(",") if p.strip()]
    first = int(parts[0])
    second = int(parts[1]) if len(parts) > 1 else None
    return Offset(first=first, second=second)


def _parse_link(content: str) -> Link:
    m = _LINK_RE.match(content)
    if not m:
        return Link(author=None, url="", title=content, suffix=None, raw=content)
    author = (m.group("author") or "").strip() or None
    suffix = (m.group("suffix") or "").strip() or None
    return Link(
        author=author,
        url=m.group("url").strip(),
        title=m.group("title").strip(),
        suffix=suffix,
        seq_refs=_seq_refs(content),
        raw=content,
    )


def _parse_crossrefs(lines: list[str]) -> CrossRefs:
    xr = CrossRefs()
    for item in _group_items(lines):
        ctx = re.match(r"Sequence in context:\s*(.*)$", item, re.DOTALL)
        adj = re.match(r"Adjacent sequences:\s*(.*)$", item, re.DOTALL)
        if ctx:
            xr.in_context = [t for t in _SEQREF_RE.findall(ctx.group(1))]
        elif adj:
            xr.adjacent = [t for t in _SEQREF_RE.findall(adj.group(1))]
        else:
            xr.items.append(_annotate(item))
    return xr


# --- top-level --------------------------------------------------------------


def parse_text(text: str) -> Sequence:
    """Parse the contents of one ``.seq`` file into a :class:`Sequence`."""
    fields: dict[str, list[str]] = {}
    anum: Optional[str] = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = _LINE_RE.match(raw)
        if not m:
            continue  # tolerate stray lines
        anum = anum or m.group("anum")
        fields.setdefault(m.group("code"), []).append(m.group("content") or "")

    if anum is None:
        raise ValueError("no valid %-lines found; not an OEIS internal-format file")

    seq = Sequence(anum=anum)

    if "I" in fields:
        seq.identification = _parse_identification(fields["I"][0], anum)

    data_lines = fields.get("S", []) + fields.get("T", []) + fields.get("U", [])
    if data_lines:
        seq.data = _parse_data(data_lines)

    if "N" in fields:
        seq.name = fields["N"][0]

    seq.comments = [_annotate(it) for it in _group_items(fields.get("C", []))]
    seq.references = [_annotate(it) for it in _group_items(fields.get("D", []))]
    seq.links = [_parse_link(c) for c in fields.get("H", [])]
    seq.formulas = [_annotate(it) for it in _group_items(fields.get("F", []))]
    seq.examples = [_annotate(it) for it in _group_items(fields.get("e", []))]
    seq.extensions = [_annotate(it) for it in _group_items(fields.get("E", []))]

    seq.maple = _split_programs(fields.get("p", []), "Maple", tagged=False)
    seq.mathematica = _split_programs(fields.get("t", []), "Mathematica", tagged=False)
    seq.programs = _split_programs(fields.get("o", []), "", tagged=True)

    if "Y" in fields:
        seq.crossrefs = _parse_crossrefs(fields["Y"])

    if "K" in fields:
        seq.keywords = [k.strip() for k in fields["K"][0].split(",") if k.strip()]

    if "O" in fields:
        seq.offset = _parse_offset(fields["O"][0])

    if "A" in fields:
        # Multiple %A lines occur; first is the primary author.
        seq.author = _annotate(fields["A"][0])

    return seq


def parse_file(path: str) -> Sequence:
    with open(path, encoding="utf-8") as fh:
        return parse_text(fh.read())
