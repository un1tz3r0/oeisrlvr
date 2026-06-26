#!/usr/bin/env python3
"""Generate cover art for the OEIS RLVR repository.

Produces cover_art.png (1280×640) depicting the core idea:
integer sequences → RLVR training loop → synthesised Python functions.
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch

# ── Canvas ──────────────────────────────────────────────────────────────────
W, H  = 1280, 640
BG    = '#050a14'

fig = plt.figure(figsize=(W / 100, H / 100), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')

# ── Background: faint scattered sequence numbers ─────────────────────────────
rng = np.random.default_rng(2025)
bg_pool = [
    0,1,1,2,3,5,8,13,21,34,55,89,144,233,
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,
    1,1,2,5,14,42,132,429,1430,4862,
    1,2,6,24,120,720,5040,40320,
    1,4,9,16,25,36,49,64,81,100,121,
    0,1,3,6,10,15,21,28,36,45,55,
]
for _ in range(400):
    x = rng.uniform(0, W)
    y = rng.uniform(0, H)
    n = rng.choice(bg_pool)
    a = rng.uniform(0.015, 0.07)
    fs = rng.uniform(7, 18)
    ax.text(x, y, str(n), color='#3366aa', alpha=a, fontsize=fs,
            fontfamily='monospace', ha='center', va='center', zorder=1)

# ── Helpers ──────────────────────────────────────────────────────────────────
def gtext(x, y, s, color='white', fontsize=12, glow=None, glow_w=6, zorder=8, **kw):
    kw.setdefault('fontfamily', 'monospace')
    kw.setdefault('ha', 'left')
    kw.setdefault('va', 'center')
    t = ax.text(x, y, s, color=color, fontsize=fontsize, zorder=zorder, **kw)
    if glow:
        t.set_path_effects([
            pe.withStroke(linewidth=glow_w, foreground=glow),
            pe.Normal(),
        ])
    return t

# ── Left panel: famous OEIS sequences ────────────────────────────────────────
SEQS = [
    ('A000045', 'Fibonacci',  [0,1,1,2,3,5,8,13,21,34],     '#00e5ff'),
    ('A000040', 'Primes',     [2,3,5,7,11,13,17,19,23,29],  '#69ff47'),
    ('A000108', 'Catalan',    [1,1,2,5,14,42,132,429],       '#ff6ec7'),
    ('A000142', 'Factorial',  [1,2,6,24,120,720,5040],       '#ffd700'),
    ('A000290', 'Squares',     [0,1,4,9,16,25,36,49,64,81],  '#ff9500'),
]

LX     = 28
Y_TOP  = 522
Y_STEP = 96

for i, (anum, name, terms, col) in enumerate(SEQS):
    yb = Y_TOP - i * Y_STEP
    ax.text(LX, yb + 18, f'{anum}  {name}',
            color=col, fontsize=8.5, alpha=0.72,
            fontfamily='monospace', ha='left', va='center', zorder=3)
    ts = ', '.join(str(t) for t in terms) + ', …'
    t = ax.text(LX, yb - 8, ts, color=col, fontsize=11.5,
                fontfamily='monospace', ha='left', va='center', zorder=3)
    t.set_path_effects([
        pe.withStroke(linewidth=3, foreground='#001122'),
        pe.Normal(),
    ])

# Thin arrows converging leftward into center zone
CX, CY = 640, 310
for i, (_, _, _, col) in enumerate(SEQS):
    yb = Y_TOP - i * Y_STEP - 8
    ax.annotate(
        '', xy=(448, CY + (yb - CY) * 0.18), xytext=(410, yb),
        arrowprops=dict(arrowstyle='->', color=col, alpha=0.20,
                        lw=1.3, connectionstyle='arc3,rad=0.04'),
        zorder=2,
    )

# Soft left-to-center fade mask (stacked semi-transparent rects)
for xi in range(370, 500):
    a = 0.28 * ((xi - 370) / 130) ** 2
    ax.add_patch(Rectangle((xi, 0), 1, H, color=BG, alpha=a, zorder=4))

# ── Center: glow core ────────────────────────────────────────────────────────
for r, a in [(180, 0.03), (135, 0.06), (95, 0.11), (65, 0.19), (42, 0.35), (24, 0.65)]:
    ax.add_patch(Circle((CX, CY), r, color='#0088cc', alpha=a, zorder=5))
ax.add_patch(Circle((CX, CY), 20, color='#aaeeff', alpha=0.92, zorder=6))
ax.add_patch(Circle((CX, CY), 12, color='white',   alpha=1.00, zorder=7))

# f(n) label inside core
ax.text(CX, CY + 9,  'f',   color='#003355', fontsize=18, ha='center', va='center',
        fontstyle='italic', fontweight='bold', zorder=8)
ax.text(CX, CY - 11, '(n)', color='#003355', fontsize=10, ha='center', va='center',
        fontfamily='monospace', zorder=8)

# GRPO reward bars below core
bar_vals   = [0.18, 0.32, 0.52, 0.74, 0.94]
bar_colors = ['#0d2040', '#1a3a6a', '#1a5090', '#2277cc', '#00aaee']
bar_w, bar_gap = 28, 9
bar_max_h  = 90
bar_x0     = CX - (len(bar_vals) * (bar_w + bar_gap) - bar_gap) // 2
bar_bottom = CY - 178

for j, (v, bc) in enumerate(zip(bar_vals, bar_colors)):
    bx = bar_x0 + j * (bar_w + bar_gap)
    bh = v * bar_max_h
    ax.add_patch(Rectangle((bx, bar_bottom), bar_w, bh, color=bc, alpha=0.9, zorder=5))

ax.text(CX, bar_bottom - 18, 'GRPO reward', color='#3a5f7f',
        fontsize=9, fontfamily='monospace', ha='center', va='center', zorder=5)

ax.text(CX, CY + 75, 'RLVR', color='#0088cc', fontsize=13,
        fontfamily='monospace', fontweight='bold', ha='center', va='center',
        alpha=0.75, zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground='#000d1f'), pe.Normal()])

# Arrows from core rightward into code panel
for ty in [530, 455, 380, 305, 230]:
    ax.annotate(
        '', xy=(818, ty), xytext=(CX + 48, CY + (ty - CY) * 0.12),
        arrowprops=dict(arrowstyle='->', color='#00aadd', alpha=0.18,
                        lw=1.2, connectionstyle='arc3,rad=-0.04'),
        zorder=2,
    )

# Soft right-to-center fade mask
for xi in range(780, 840):
    a = 0.28 * ((840 - xi) / 60) ** 2
    ax.add_patch(Rectangle((xi, 0), 1, H, color=BG, alpha=a, zorder=4))

# ── Right panel: synthesised Python code ─────────────────────────────────────
RX = 848

code_bg = FancyBboxPatch(
    (RX - 14, 118), 424, 458,
    boxstyle='round,pad=6',
    facecolor='#080f1c', edgecolor='#1a3a6a',
    linewidth=1.5, alpha=0.93, zorder=6,
)
ax.add_patch(code_bg)

# Syntax colours (VSCode Dark+ inspired)
SK = '#c792ea'   # keyword
SN = '#82aaff'   # name
SP = '#89ddff'   # punctuation
SL = '#f78c6c'   # literal
SC = '#546e7a'   # comment
SO = '#c3e88d'   # operator
SW = '#cdd3de'   # plain
SG = '#4caf50'   # green pass

def code_row(x, y, parts, zorder=9):
    cx = x
    for text, color in parts:
        ax.text(cx, y, text, color=color, fontsize=10.5,
                fontfamily='monospace', ha='left', va='center', zorder=zorder,
                path_effects=[pe.withStroke(linewidth=1, foreground='#080f1c'),
                              pe.Normal()])
        cx += len(text) * 8.85   # monospace char width: ~0.6 × 14.6px height at 10.5pt/100dpi

INDENT  = 20
INDENT2 = 40
FS = 10.5

rows = [
    (RX,         543, [('def ', SK), ('a', SN), ('(n):', SP)]),
    (RX+INDENT,  498, [('a', SN), (', ', SW), ('b', SN), (' = ', SP), ('0', SL), (', ', SW), ('1', SL)]),
    (RX+INDENT,  453, [('for', SK), (' _ ', SW), ('in', SK), (' range', SN), ('(n):', SP)]),
    (RX+INDENT2, 408, [('a', SN), (', ', SW), ('b', SN), (' = ', SP), ('b', SN), (', ', SW), ('a', SN), ('+', SO), ('b', SN)]),
    (RX+INDENT,  363, [('return', SK), (' a', SN)]),
    (RX,         310, [('# ✓ compiles & no imports', SC)]),
    (RX,         278, [('# ✓ a(10) = ', SG), ('55', SG), ('  ← held-out', SC)]),
    (RX,         248, [('# ✓ a(11) = ', SG), ('89', SG)]),
    (RX,         218, [('# ✓ a(12) = ', SG), ('144', SG)]),
    (RX,         175, [('reward:', SC), ('  1.00  ✓✓✓', SG)]),
    (RX,         148, [('# sequence_matches = True', SG)]),
]
for (x, y, parts) in rows:
    code_row(x, y, parts)

# ── Title ────────────────────────────────────────────────────────────────────
title = ax.text(CX, H - 40, 'OEIS  RLVR',
                color='white', fontsize=44, fontweight='bold',
                fontfamily='monospace', ha='center', va='center', zorder=12)
title.set_path_effects([
    pe.withStroke(linewidth=14, foreground='#0055aa'),
    pe.withStroke(linewidth=6,  foreground='#0099cc'),
    pe.Normal(),
])

ax.text(CX, H - 82, 'Reinforcement Learning for Integer Sequence Synthesis',
        color='#4d7a9b', fontsize=11.5, fontfamily='monospace',
        ha='center', va='center', zorder=12)

# Thin top border line
ax.plot([0, W], [H - 102, H - 102], color='#0d2a44', lw=1, zorder=3)

# ── Export ───────────────────────────────────────────────────────────────────
plt.savefig('cover_art.png', dpi=100, facecolor=BG,
            bbox_inches=None, pad_inches=0)
print('Saved cover_art.png')
plt.close()
