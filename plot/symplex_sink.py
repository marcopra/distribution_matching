"""
simplex_sink.py
---------------
Three-panel 3D figure showing how the norm of vertex s3 changes
the point on the simplex closest to the origin:

    Simplex = conv{s1, s2, s3},  s1=[1,0,0],  s2=[0,1,0]

  s3 = [0,0, 0]  →  closest point is s3 itself (at the origin)
  s3 = [0,0, 1]  →  closest point is the centroid  [1/3, 1/3, 1/3]
  s3 = [0,0,10]  →  closest point near midpoint of s1–s2 edge

The simplex is drawn as a wireframe triangle (no fill).
The star ★ marks the closest point to the origin; a dotted line
connects the origin to that point.

Dependencies: matplotlib, numpy
Usage:       python simplex_sink.py
Output:      simplex_sink.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'serif'})

# ── Colour palette ────────────────────────────────────────────────────────
INK  = '#1A1A2E'
ACC  = '#3D6B8C'
RED  = '#C0392B'
GRN  = '#2E8C5A'
ORG  = '#E67E22'
STAR = '#8E44AD'   # purple star for closest point

# ── Fixed vertices ────────────────────────────────────────────────────────
s1 = np.array([1., 0.,  0.])
s2 = np.array([0., 1.,  0.])

# ── Panels config ─────────────────────────────────────────────────────────
s3_list = [
    np.array([0., 0.,  0.]),
    np.array([0., 0.,  1.]),
    np.array([0., 0., 3.]),
]
panel_titles = [
    r'$s_3 = (0,\,0,\,0)$',
    r'$s_3 = (0,\,0,\,1)$',
    r'$s_3 = (0,\,0,\,3)$',
]
minimiser_labels = [
    r'Closest $\to$ $s_3$' '\n(= origin)',
    r'Closest $\to$ centroid',
    r'Closest $\to$ mid($s_1$,$s_2$)',
]

# ── True closest point to origin on triangle conv{s1,s2,s3} ───────────────
def closest_on_simplex(s1, s2, s3):
    """Return (x, y, z, d1, d2, d3) where d1*s1 + d2*s2 + d3*s3 is the
    exact closest point to the origin on conv{s1,s2,s3}."""
    p = np.zeros(3)
    a, b, c = s1, s2, s3

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a[0], a[1], a[2], 1.0, 0.0, 0.0

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b[0], b[1], b[2], 0.0, 1.0, 0.0

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        q = a + v * ab
        return q[0], q[1], q[2], 1.0 - v, v, 0.0

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c[0], c[1], c[2], 0.0, 0.0, 1.0

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        q = a + w * ac
        return q[0], q[1], q[2], 1.0 - w, 0.0, w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        u = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = b + u * (c - b)
        return q[0], q[1], q[2], 0.0, 1.0 - u, u

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w
    q = u * a + v * b + w * c
    return q[0], q[1], q[2], u, v, w

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 6.5))
X_LIM = (0.0, 1.15)
Y_LIM = (0.0, 1.15)
Z_LIMS = [
    (0.0, 1.15),
    (0.0, 1.15),
    (0.0, 3.5),
]

for panel, (s3, title, mlbl) in enumerate(
        zip(s3_list, panel_titles, minimiser_labels)):

    vertices = np.array([s1, s2, s3])

    ax = fig.add_subplot(1, 3, panel+1, projection='3d')

    # ── Simplex face (translucent, no gradient) ───────────────────────────
    tri = [list(map(tuple, vertices))]
    poly = Poly3DCollection(tri, alpha=0.10,
                            facecolor=ACC, edgecolor='none')
    ax.add_collection3d(poly)

    # ── Simplex edges (wireframe only) ────────────────────────────────────
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        ax.plot3D(*zip(vertices[i], vertices[j]),
                  color=ACC, lw=2.2, zorder=5)

    # ── Origin ────────────────────────────────────────────────────────────
    ax.scatter(0, 0, 0, s=90, color='#333333', marker='o',
               zorder=6, depthshade=False)
    ax.text(0.03, 0.03, 0.05, r'$\mathbf{0}$',
            fontsize=11, color='#333333', fontweight='bold', zorder=9)

    # ── Closest point to origin on simplex ───────────────────────────────
    xm, ym, zm, d1m, d2m, d3m = closest_on_simplex(s1, s2, s3)

    ax.scatter(xm, ym, zm, s=320, color=STAR, zorder=10,
               marker='*', depthshade=False,
               edgecolors='white', linewidths=0.6)

    # Dotted line from origin to closest point
    ax.plot3D([0, xm], [0, ym], [0, zm],
              color=STAR, lw=2.0, linestyle=':', zorder=4, alpha=0.95)

    # Label for closest point
    z_off = 0.12 if s3[2] < 2 else 0.65
    ax.text(xm + 0.04, ym - 0.15, zm + z_off,
            mlbl, fontsize=9, color=STAR, fontweight='bold',
            zorder=12, ha='center',
            bbox=dict(facecolor='white', edgecolor=STAR,
                      alpha=0.88, pad=1.5, lw=0.8))

    # ── Vertex labels ─────────────────────────────────────────────────────
    v_labels  = [r'$s_1$', r'$s_2$', r'$s_3$']
    v_colors  = [RED,      GRN,      ORG      ]
    # offsets tuned per panel to avoid overlap
    v_off = [
        np.array([-0.18, -0.06,  0.00]),
        np.array([ 0.04,  0.12,  0.00]),
        np.array([ 0.04, -0.06,  0.10 if s3[2] > 0 else 0.06]),
    ]

    for vi, (lbl, col, off) in enumerate(zip(v_labels, v_colors, v_off)):
        ax.scatter(*vertices[vi], s=80, color=col, zorder=7, depthshade=False)
        ax.text(vertices[vi][0] + off[0],
                vertices[vi][1] + off[1],
                vertices[vi][2] + off[2],
                lbl, fontsize=12, color=col, fontweight='bold', zorder=9)

    # ── Axes & limits ─────────────────────────────────────────────────────
    z_lim = Z_LIMS[panel]
    box_aspect = (X_LIM[1] - X_LIM[0], Y_LIM[1] - Y_LIM[0], z_lim[1] - z_lim[0])
    ax.set_title(title, fontsize=13, color=INK, fontweight='bold', pad=10)
    ax.set_xlabel(r'$d_\pi(s_1)$', fontsize=10, labelpad=4)
    ax.set_ylabel(r'$d_\pi(s_2)$', fontsize=10, labelpad=4)
    ax.set_zlabel(r'$d_\pi(s_3)$', fontsize=10, labelpad=4)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect(box_aspect)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=24, azim=48)
    ax.set_facecolor('#F6F8FC')
    ax.grid(True, alpha=0.20)

fig.suptitle(
    r'Closest point on $\mathrm{conv}\{s_1,s_2,s_3\}$ to the origin '
    r'($\bigstar$) as $\|s_3\|$ varies',
    fontsize=12, color=INK, y=1.01)

plt.tight_layout()
fig.patch.set_facecolor('white')
plt.savefig('simplex_sink.png', dpi=220, bbox_inches='tight',
            pad_inches=0.08, facecolor='white')
print("Saved → simplex_sink.png")