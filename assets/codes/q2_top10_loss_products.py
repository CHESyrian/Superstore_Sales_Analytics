import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import textwrap

df = pd.read_csv('superstore_cleaned.csv')

# ── Aggregation ───────────────────────────────────────────────────────────────
product = df.groupby(['Product Name', 'Category', 'Sub-Category']).agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Total_Orders=('Order ID', 'nunique'),
    Total_Quantity=('Quantity', 'sum'),
    Avg_Discount=('Discount', 'mean'),
    Loss_Rows=('Is Loss', 'sum'),
    Total_Rows=('Profit', 'count')
).reset_index()
product['Profit_Margin'] = product['Total_Profit'] / product['Total_Sales'] * 100
product['Loss_Rate']     = product['Loss_Rows'] / product['Total_Rows'] * 100

top10 = product.nsmallest(10, 'Total_Profit').reset_index(drop=True)

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE   = '#378ADD'
CORAL  = '#D85A30'
TEAL   = '#1D9E75'
AMBER  = '#BA7517'
PURPLE = '#7F77DD'
GRAY   = '#888780'
DARK   = '#2C2C2A'
LIGHT  = '#F8F7F4'
CAT_COLORS = {'Furniture': PURPLE, 'Office Supplies': TEAL, 'Technology': BLUE}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(LIGHT)
plt.suptitle('Q2 — Top 10 loss-making products', fontsize=17,
             fontweight='500', color=DARK, y=0.98)

gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
              left=0.07, right=0.97, top=0.93, bottom=0.06)

short_names = [textwrap.fill(n, 32) for n in top10['Product Name']]

# ── Chart 1: Total profit horizontal bar ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0:2, 0:2])
colors_bar = [CAT_COLORS.get(c, CORAL) for c in top10['Category']]
bars = ax1.barh(short_names[::-1], top10['Total_Profit'][::-1] / 1e3,
                color=colors_bar[::-1], edgecolor='white', height=0.65)
ax1.axvline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax1.set_title('Total profit — top 10 loss-making products ($k)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax1.set_xlabel('Total profit ($k)', fontsize=9, color=GRAY)
ax1.set_facecolor('white')
ax1.tick_params(axis='y', labelsize=8)
ax1.tick_params(axis='x', labelsize=9)
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars, top10['Total_Profit'][::-1] / 1e3):
    ax1.text(val - 0.1, bar.get_y() + bar.get_height() / 2,
             f'${val:.1f}k', va='center', ha='right', fontsize=8,
             fontweight='500', color='white')
patches = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLORS.items()]
ax1.legend(handles=patches, fontsize=8.5, frameon=False, loc='lower right')

# ── Chart 2: Avg discount per product ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0:2, 2])
disc_colors = [CORAL if d > 0.30 else AMBER if d > 0.15 else TEAL
               for d in top10['Avg_Discount']]
bars2 = ax2.barh(short_names[::-1], top10['Avg_Discount'][::-1] * 100,
                 color=disc_colors[::-1], edgecolor='white', height=0.65)
ax2.axvline(15.6, color=GRAY, linewidth=1, linestyle=':', alpha=0.8)
ax2.text(16, len(top10) * 0.97, 'Dataset\navg 15.6%', fontsize=7, color=GRAY)
ax2.set_title('Avg discount per product (%)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax2.set_xlabel('Avg discount %', fontsize=9, color=GRAY)
ax2.set_facecolor('white')
ax2.tick_params(axis='y', labelsize=8)
ax2.tick_params(axis='x', labelsize=9)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars2, top10['Avg_Discount'][::-1] * 100):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
             f'{val:.0f}%', va='center', ha='left', fontsize=8, color=DARK)
p1 = mpatches.Patch(color=CORAL, label='>30% discount')
p2 = mpatches.Patch(color=AMBER, label='15–30%')
p3 = mpatches.Patch(color=TEAL,  label='<15%')
ax2.legend(handles=[p1, p2, p3], fontsize=7.5, frameon=False)

# ── Chart 3: Loss rate per product ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
lr_colors = [CORAL if r >= 75 else AMBER if r >= 50 else PURPLE
             for r in top10['Loss_Rate']]
bars3 = ax3.bar(range(len(top10)), top10['Loss_Rate'],
                color=lr_colors, edgecolor='white', width=0.6)
ax3.set_xticks(range(len(top10)))
ax3.set_xticklabels([f'P{i+1}' for i in range(len(top10))], fontsize=8.5)
ax3.set_title('Loss rate per product (%)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax3.set_ylabel('% loss rows', fontsize=9, color=GRAY)
ax3.set_facecolor('white')
ax3.tick_params(labelsize=9)
ax3.spines[['top', 'right']].set_visible(False)
ax3.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars3, top10['Loss_Rate']):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val:.0f}%', ha='center', fontsize=8, color=DARK)
ax3.set_ylim(0, 120)

# ── Chart 4: Sales vs profit scatter (bubble = order count) ──────────────────
ax4 = fig.add_subplot(gs[2, 1])
sc_colors = [CAT_COLORS.get(c, GRAY) for c in top10['Category']]
ax4.scatter(top10['Total_Sales'] / 1e3, top10['Total_Profit'] / 1e3,
            c=sc_colors, s=top10['Total_Orders'] * 30, alpha=0.85,
            edgecolors='white', linewidth=0.8, zorder=3)
ax4.axhline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
for i, row in top10.iterrows():
    ax4.annotate(f'P{i+1}',
                 (row['Total_Sales'] / 1e3, row['Total_Profit'] / 1e3),
                 textcoords='offset points', xytext=(5, 4),
                 fontsize=7.5, color=DARK)
ax4.set_title('Sales vs profit\n(bubble = order count)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax4.set_xlabel('Total sales ($k)', fontsize=9, color=GRAY)
ax4.set_ylabel('Total profit ($k)', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(labelsize=8.5)
ax4.spines[['top', 'right']].set_visible(False)
ax4.spines[['left', 'bottom']].set_color(GRAY)
patches = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLORS.items()]
ax4.legend(handles=patches, fontsize=7.5, frameon=False)

# ── Panel 5: Product index legend ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
ax5.set_facecolor('#F1EFE8')
ax5.axis('off')
ax5.text(0.5, 0.98, 'Product index', ha='center', va='top',
         fontsize=10, fontweight='500', color=DARK, transform=ax5.transAxes)
for i, row in top10.iterrows():
    y     = 0.90 - i * 0.089
    color = CAT_COLORS.get(row['Category'], GRAY)
    label = textwrap.fill(row['Product Name'], 30)
    ax5.add_patch(plt.Rectangle((0.01, y - 0.01), 0.06, 0.055,
                                color=color, transform=ax5.transAxes, clip_on=False))
    ax5.text(0.10, y + 0.02, f'P{i+1}', ha='left', va='center',
             fontsize=8, fontweight='500', color=color, transform=ax5.transAxes)
    ax5.text(0.10, y - 0.02, label[:38] + ('…' if len(label) > 38 else ''),
             ha='left', va='center', fontsize=6.8, color=GRAY, transform=ax5.transAxes)

plt.savefig('q2_top10_loss_products.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
