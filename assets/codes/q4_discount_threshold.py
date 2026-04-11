import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

df = pd.read_csv('superstore_cleaned.csv')

# ── Discount bins ─────────────────────────────────────────────────────────────
bins   = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.01]
labels = ['0%', '1-5%', '6-10%', '11-15%', '16-20%', '21-25%', '26-30%',
          '31-35%', '36-40%', '41-50%', '51-60%', '61-80%']
df['Discount_Bin'] = pd.cut(df['Discount'], bins=bins, labels=labels, right=False)

sub_bin = df.groupby(['Sub-Category', 'Discount_Bin'], observed=True).agg(
    Loss_Rate=('Is Loss', 'mean'),
    Avg_Margin=('Profit Margin (%)', 'mean'),
    Count=('Profit', 'count')
).reset_index()
sub_bin['Loss_Rate'] *= 100

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE   = '#378ADD'
CORAL  = '#D85A30'
TEAL   = '#1D9E75'
AMBER  = '#BA7517'
PURPLE = '#7F77DD'
GRAY   = '#888780'
DARK   = '#2C2C2A'
LIGHT  = '#F8F7F4'

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor(LIGHT)
plt.suptitle('Q4 — Discount threshold for loss by sub-category', fontsize=17,
             fontweight='500', color=DARK, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
              left=0.07, right=0.97, top=0.93, bottom=0.05)

# ── Chart 1: Heatmap — loss rate per sub-category × discount bin ─────────────
ax1 = fig.add_subplot(gs[0:2, 0:2])
pivot = sub_bin.pivot(index='Sub-Category', columns='Discount_Bin', values='Loss_Rate')
pivot = pivot.reindex(columns=[l for l in labels if l in pivot.columns])
row_order = ['Tables', 'Bookcases', 'Storage', 'Chairs', 'Phones', 'Machines',
             'Supplies', 'Furnishings', 'Binders', 'Appliances',
             'Accessories', 'Art', 'Copiers', 'Envelopes', 'Fasteners', 'Labels', 'Paper']
pivot = pivot.reindex([r for r in row_order if r in pivot.index])
sns.heatmap(pivot, ax=ax1, cmap='RdYlGn_r', center=50,
            vmin=0, vmax=100, annot=True, fmt='.0f',
            linewidths=0.4, linecolor='#F8F7F4',
            cbar_kws={'label': 'Loss rate %', 'shrink': 0.8},
            annot_kws={'size': 8})
ax1.set_title('Loss rate heatmap: sub-category × discount bin (%)',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax1.set_xlabel('Discount bin', fontsize=9, color=GRAY)
ax1.set_ylabel('Sub-category', fontsize=9, color=GRAY)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.tick_params(axis='y', rotation=0, labelsize=8.5)

# ── Chart 2: Break-even threshold per sub-category (horizontal bar) ───────────
ax2 = fig.add_subplot(gs[0:2, 2])
threshold_order = [
    ('Tables',       '26-30%', CORAL),
    ('Bookcases',    '16-20%', CORAL),
    ('Storage',      '16-20%', CORAL),
    ('Chairs',       '26-30%', CORAL),
    ('Phones',       '36-40%', AMBER),
    ('Machines',     '36-40%', AMBER),
    ('Supplies',     'Never*', AMBER),
    ('Furnishings',  '51-60%', TEAL),
    ('Binders',      '51-60%', TEAL),
    ('Appliances',   '61-80%', TEAL),
    ('Accessories',  'Never',  TEAL),
    ('Art',          'Never',  TEAL),
    ('Copiers',      'Never',  TEAL),
    ('Envelopes',    'Never',  TEAL),
    ('Fasteners',    'Never',  TEAL),
    ('Labels',       'Never',  TEAL),
    ('Paper',        'Never',  TEAL),
]
thresh_map = {'16-20%': 18, '26-30%': 28, '36-40%': 38, '41-50%': 45,
              '51-60%': 55, '61-80%': 70, 'Never': 85, 'Never*': 85}
names   = [t[0] for t in threshold_order]
xvals   = [thresh_map[t[1]] for t in threshold_order]
colors  = [t[2] for t in threshold_order]
tlabels = [t[1] for t in threshold_order]

bars = ax2.barh(names[::-1], xvals[::-1], color=colors[::-1],
                edgecolor='white', height=0.65)
ax2.axvline(20, color=GRAY, linewidth=1, linestyle=':', alpha=0.7)
ax2.text(20.5, len(names) - 0.3, 'Suggested\ncap (20%)', fontsize=7.5, color=GRAY)
ax2.set_title('Break-even threshold\nper sub-category', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax2.set_xlabel('Discount % at which losses begin', fontsize=9, color=GRAY)
ax2.set_facecolor('white')
ax2.tick_params(axis='y', labelsize=8.5)
ax2.tick_params(axis='x', labelsize=9)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color(GRAY)
ax2.set_xlim(0, 100)
for bar, lbl in zip(bars, tlabels[::-1]):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
             lbl, va='center', fontsize=7.5, color=DARK)
p1 = mpatches.Patch(color=CORAL, label='High risk (≤30%)')
p2 = mpatches.Patch(color=AMBER, label='Medium risk (31-50%)')
p3 = mpatches.Patch(color=TEAL,  label='Low risk (>50% or never)')
ax2.legend(handles=[p1, p2, p3], fontsize=7.5, frameon=False, loc='lower right')

# ── Chart 3: Loss rate trend — high-risk sub-categories ──────────────────────
ax3 = fig.add_subplot(gs[2, 0])
bin_numeric = {'0%': 0, '1-5%': 3, '6-10%': 8, '11-15%': 13, '16-20%': 18,
               '21-25%': 23, '26-30%': 28, '31-35%': 33, '36-40%': 38,
               '41-50%': 45, '51-60%': 55, '61-80%': 70}
high_risk       = ['Tables', 'Bookcases', 'Storage', 'Chairs']
high_risk_colors = {'Tables': CORAL, 'Bookcases': '#993C1D', 'Storage': PURPLE, 'Chairs': AMBER}
for sub in high_risk:
    sdf = sub_bin[(sub_bin['Sub-Category'] == sub) & (sub_bin['Count'] >= 5)].copy()
    sdf['x'] = sdf['Discount_Bin'].map(bin_numeric)
    sdf = sdf.dropna(subset=['x']).sort_values('x')
    ax3.plot(sdf['x'], sdf['Loss_Rate'], marker='o', markersize=5,
             linewidth=1.8, label=sub, color=high_risk_colors[sub])
ax3.axhline(50, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_title('Loss rate trend — high-risk\nsub-categories', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax3.set_xlabel('Discount %', fontsize=9, color=GRAY)
ax3.set_ylabel('Loss rate %', fontsize=9, color=GRAY)
ax3.set_facecolor('white')
ax3.tick_params(labelsize=8.5)
ax3.spines[['top', 'right']].set_visible(False)
ax3.spines[['left', 'bottom']].set_color(GRAY)
ax3.legend(fontsize=8, frameon=False)
ax3.set_ylim(0, 110)

# ── Chart 4: Avg margin trend — medium-risk sub-categories ───────────────────
ax4 = fig.add_subplot(gs[2, 1])
med_risk       = ['Phones', 'Machines', 'Binders', 'Furnishings']
med_risk_colors = {'Phones': BLUE, 'Machines': PURPLE, 'Binders': AMBER, 'Furnishings': TEAL}
for sub in med_risk:
    sdf = sub_bin[(sub_bin['Sub-Category'] == sub) & (sub_bin['Count'] >= 5)].copy()
    sdf['x'] = sdf['Discount_Bin'].map(bin_numeric)
    sdf = sdf.dropna(subset=['x']).sort_values('x')
    ax4.plot(sdf['x'], sdf['Avg_Margin'], marker='o', markersize=5,
             linewidth=1.8, label=sub, color=med_risk_colors[sub])
ax4.axhline(0, color=CORAL, linewidth=0.8, linestyle='--', alpha=0.7)
ax4.set_title('Avg margin trend — medium-risk\nsub-categories', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax4.set_xlabel('Discount %', fontsize=9, color=GRAY)
ax4.set_ylabel('Avg margin %', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(labelsize=8.5)
ax4.spines[['top', 'right']].set_visible(False)
ax4.spines[['left', 'bottom']].set_color(GRAY)
ax4.legend(fontsize=8, frameon=False)

# ── Panel 5: Key findings ─────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
ax5.set_facecolor('#F1EFE8')
ax5.axis('off')
ax5.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax5.transAxes)
findings = [
    ("Storage & Bookcases", "Loss begins at just 16-20%",      CORAL),
    ("Tables & Chairs",     "Loss begins at 26-30%",           CORAL),
    ("Phones & Machines",   "Loss begins at 36-40%",           AMBER),
    ("Binders & Furnishings", "Loss begins at 51-60%",         TEAL),
    ("Appliances",          "Survives until 61-80%",           TEAL),
    ("Paper, Labels, Art",  "Never loss-making in dataset",    TEAL),
    ("Supplies",            "Loss-making at 0% discount too",  CORAL),
    ("Max safe cap",        "20% across all sub-categories",   AMBER),
    ("Immediate action",    "Stop >30% on Tables & Chairs",    CORAL),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax5.add_patch(plt.Rectangle((0.02, y - 0.025), 0.008, 0.045,
                                color=color, transform=ax5.transAxes, clip_on=False))
    ax5.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax5.transAxes)
    ax5.text(0.07, y - 0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax5.transAxes)

plt.savefig('q4_discount_threshold.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
