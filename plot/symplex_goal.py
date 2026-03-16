"""
simplex_goal.py
---------------
Plots the probability simplex Δ(S) for n=3 states in 3D,
showing that the minimiser of ‖d^π‖² is the centroid u=[1/3,1/3,1/3],
i.e. the point on the simplex closest to the simplex center.

Dependencies: matplotlib, numpy, plotly
Usage:       python simplex_goal.py
Output:      simplex_goal.png, simplex_goal.html
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Colour palette ────────────────────────────────────────────────────────
INK = '#1A1A2E'
ACC = '#3D6B8C'
ACLt = '#EAF2F8'
RED = '#C0392B'
GRN = '#2E8C5A'
PUR = '#7C3AED'
ORG = '#E67E22'

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 8.5))
ax = fig.add_subplot(111, projection='3d')

plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'serif'})

# Simplex vertices (one-hot vectors)
v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
centroid = v.mean(axis=0)
origin = np.array([0., 0., 0.])

# ── Simplex face ──────────────────────────────────────────────────────────
face = Poly3DCollection([v], alpha=0.15, facecolor=ACLt,
                        edgecolor=ACC, linewidth=1.8, zorder=1)
ax.add_collection3d(face)

for i, j in [(0, 1), (1, 2), (0, 2)]:
    ax.plot3D(*zip(v[i], v[j]), color=ACC, lw=2.0, zorder=3)

# ── Origin and center line ────────────────────────────────────────────────
ax.scatter(*origin, s=80, color='#888888', zorder=6, depthshade=False)
ax.plot3D(*zip(origin, centroid), color='#AAAAAA', lw=1.8,
          linestyle=':', zorder=2)

# ── Vertices ──────────────────────────────────────────────────────────────
v_colors = [RED, GRN, PUR]
v_offsets = [(-0.20, -0.08, 0.00),
             (0.06, 0.10, 0.00),
             (0.00, -0.06, 0.12)]

for i, (col, off) in enumerate(zip(v_colors, v_offsets)):
    ax.scatter(*v[i], s=150, color=col, zorder=8, depthshade=False)
    ax.text(v[i][0] + off[0], v[i][1] + off[1], v[i][2] + off[2],
            f'$s_{i+1}$: $({int(v[i][0])},{int(v[i][1])},{int(v[i][2])})$',
            fontsize=10.5, color=col, fontweight='bold', zorder=10)

# ── Star at centroid (closest to simplex center) ─────────────────────────
ax.scatter(*centroid, s=300, color=ORG, zorder=12, marker='*',
           depthshade=False, edgecolors='none', linewidths=0.0)
ax.text2D(0.52, 0.60,
          r'$u=[1/3,\,1/3,\,1/3]$',
          transform=ax.transAxes, fontsize=11,
          color=ORG, fontweight='bold',
          bbox=dict(facecolor='white', edgecolor=ORG,
                    alpha=0.92, pad=2.5, lw=0.9))
ax.text2D(0.49, 0.54,
          r'$\bigstar$ = closest point to simplex center',
          transform=ax.transAxes, fontsize=10,
          color=ORG, style='italic')

# ── Sample policy point d_{pi_0} ─────────────────────────────────────────
pi_pt = np.array([0.58, 0.32, 0.10])
ax.scatter(*pi_pt, s=130, color=INK, zorder=9, depthshade=False, marker='D')
ax.text(pi_pt[0] + 0.06, pi_pt[1] + 0.15, pi_pt[2] + 0.10,
        r'$d_{\pi_0}$', fontsize=12, color=INK, fontweight='bold', zorder=10)

# ── Optimisation trajectory d_{pi_0} → star ──────────────────────────────
t = np.linspace(0, 1.00, 30)
traj = np.outer(1 - t, pi_pt) + np.outer(t, centroid)
ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2],
          color=PUR, lw=2.2, linestyle='--', zorder=5, alpha=0.85)
ax.text2D(0.12, 0.70, r'$\min_\pi\|d^\pi\|^2$',
          transform=ax.transAxes, fontsize=10,
          color=PUR, style='italic',
          bbox=dict(facecolor='white', edgecolor=PUR,
                    alpha=0.82, pad=1.5, lw=0.6))

# ── Axes & layout ─────────────────────────────────────────────────────────
ax.set_xlabel(r'$d^\pi(s_1)$', fontsize=11, color=RED, labelpad=10)
ax.set_ylabel(r'$d^\pi(s_2)$', fontsize=11, color=GRN, labelpad=10)
ax.set_zlabel(r'$d^\pi(s_3)$', fontsize=11, color=PUR, labelpad=10)
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
ax.set_zlim(0, 1.1)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.set_zticks([0, 0.5, 1])
ax.tick_params(labelsize=9)
ax.view_init(elev=28, azim=52)
ax.grid(True, alpha=0.25)
ax.set_facecolor('#F6F8FC')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('simplex_goal.png', dpi=220, bbox_inches='tight',
            pad_inches=0.12, facecolor='white')
print("Saved → simplex_goal.png")

# # ── Interactive HTML version (Plotly) ─────────────────────────────────────
# fig_html = go.Figure()

# fig_html.add_trace(go.Mesh3d(
#     x=v[:, 0], y=v[:, 1], z=v[:, 2],
#     i=[0], j=[1], k=[2],
#     color=ACLt,
#     opacity=0.18,
#     showscale=False,
#     hoverinfo='skip',
#     name='simplex'
# ))

# for i, j in [(0, 1), (1, 2), (0, 2)]:
#     fig_html.add_trace(go.Scatter3d(
#         x=[v[i, 0], v[j, 0]],
#         y=[v[i, 1], v[j, 1]],
#         z=[v[i, 2], v[j, 2]],
#         mode='lines',
#         line=dict(color=ACC, width=6),
#         hoverinfo='skip',
#         showlegend=False
#     ))

# fig_html.add_trace(go.Scatter3d(
#     x=[origin[0]], y=[origin[1]], z=[origin[2]],
#     mode='markers',
#     marker=dict(size=5, color='#888888'),
#     name='origin'
# ))

# fig_html.add_trace(go.Scatter3d(
#     x=[origin[0], centroid[0]],
#     y=[origin[1], centroid[1]],
#     z=[origin[2], centroid[2]],
#     mode='lines',
#     line=dict(color='#AAAAAA', width=4, dash='dot'),
#     hoverinfo='skip',
#     showlegend=False
# ))

# vertex_names = ['s1', 's2', 's3']
# vertex_colors = [RED, GRN, PUR]
# for idx in range(3):
#     fig_html.add_trace(go.Scatter3d(
#         x=[v[idx, 0]], y=[v[idx, 1]], z=[v[idx, 2]],
#         mode='markers+text',
#         marker=dict(size=6, color=vertex_colors[idx]),
#         text=[vertex_names[idx]],
#         textposition='top center',
#         name=vertex_names[idx]
#     ))

# fig_html.add_trace(go.Scatter3d(
#     x=[pi_pt[0]], y=[pi_pt[1]], z=[pi_pt[2]],
#     mode='markers+text',
#     marker=dict(size=6, color=INK, symbol='diamond'),
#     text=['d_{pi0}'],
#     textposition='middle right',
#     name='d_pi0'
# ))

# fig_html.add_trace(go.Scatter3d(
#     x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
#     mode='lines',
#     line=dict(color=PUR, width=6, dash='dash'),
#     name='min_pi ||d^pi||^2'
# ))

# fig_html.add_trace(go.Scatter3d(
#     x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
#     mode='markers+text',
#     marker=dict(size=8, color=ORG, symbol='diamond'),
#     text=['★  u = [1/3,1/3,1/3]'],
#     textposition='top center',
#     name='closest to simplex center'
# ))

# fig_html.update_layout(
#     title='Simplex goal: star is the point closest to simplex center',
#     scene=dict(
#         xaxis=dict(title='d^pi(s1)', range=[0, 1.1]),
#         yaxis=dict(title='d^pi(s2)', range=[0, 1.1]),
#         zaxis=dict(title='d^pi(s3)', range=[0, 1.1]),
#         aspectmode='cube'
#     ),
#     margin=dict(l=0, r=0, t=40, b=0)
# )

# fig_html.write_html('simplex_goal.html', include_plotlyjs='cdn')
print("Saved → simplex_goal.html")
