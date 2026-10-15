"""Clean 'ray coordinates' inset for the architecture figure (replaces the generated raster).
Run from the repository root: python paper/figures/make_ray_inset.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thumbs', 'ray_inset.png')
TIP, DIRC, TARGET, INK = '#d2581f', '#6a35a8', '#12805a', '#191919'

fig = plt.figure(figsize=(7.85, 4.0), dpi=120)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 7.85)
ax.set_ylim(0, 4.0)
ax.axis('off')
for x in np.arange(0.25, 7.85, 0.5):
    ax.plot([x, x], [0.55, 3.95], color='#dcdcdc', lw=1, zorder=0)
for y in np.arange(0.55, 3.96, 0.5):
    ax.plot([0.25, 7.6], [y, y], color='#dcdcdc', lw=1, zorder=0)

p = np.array([0.9, 1.25])
u = np.array([0.94, 0.34]); u /= np.linalg.norm(u)
q = np.array([4.6, 3.35])
foot = p + np.dot(q - p, u) * u
ax.annotate('', xy=p + 6.5 * u, xytext=p, arrowprops=dict(arrowstyle='-|>', color=DIRC, lw=3.2, mutation_scale=26))
ax.text(*(p + 6.55 * u + np.array([0.0, 0.22])), 'θ', color=DIRC, fontsize=24, fontweight='bold', ha='center')
ax.plot(*zip(q, foot), color=TARGET, lw=2.6, ls=(0, (4, 2)))
n = np.array([-u[1], u[0]])
ax.text(*((q + foot) / 2 + 0.28 * u), '|s|', color=TARGET, fontsize=20, fontweight='bold', ha='left', va='center')
off = -0.32 * n
ax.annotate('', xy=foot + off, xytext=p + off, arrowprops=dict(arrowstyle='<->', color=TARGET, lw=2.0))
ax.text(*((p + foot) / 2 + off - 0.3 * n), 't', color=TARGET, fontsize=20, fontweight='bold', ha='center', va='center')
ax.plot(*p, 'o', ms=17, color=TIP, mec='white', mew=2.5, zorder=5)
ax.text(p[0] - 0.12, p[1] + 0.35, 'p', color=TIP, fontsize=22, fontweight='bold', ha='center')
ax.add_patch(plt.Rectangle(q - 0.17, 0.34, 0.34, color=TARGET, zorder=5))
ax.text(q[0], q[1] + 0.32, 'q', color=TARGET, fontsize=22, fontweight='bold', ha='center')
ax.text(3.92, 0.18, 'per cell q:  Δx, Δy,  t,  |s|,  cos θ,  sin θ', color=INK, fontsize=17, ha='center', va='center')
fig.savefig(OUT, dpi=120, facecolor='white')
print('wrote', OUT)
