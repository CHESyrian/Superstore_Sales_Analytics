import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove this line if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

df = pd.read_csv('superstore_cleaned.csv')

# ── Aggregations ─────────────────────────────────────────────────────────────
cat = df.groupby('Category').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Rows=('Profit', 'count'),
    Loss_Rows=('Is Loss', 'sum'),
    Avg_Discount=('Discount', 'mean')
).reset_index()
cat['Profit_Margin'] = cat['Total_Profit'] / cat['Total_Sales'] * 100
cat['Loss_Rate']     = cat['Loss_Rows'] / cat['Rows'] * 100

sub = df.groupby(['Category', 'Sub-Category']).agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Rows=('Profit', 'count'),
    Loss_Rows=('Is Loss', 'sum'),
    Avg_Discount=('Discount', 'mean')
).reset_index()
sub['Profit_Margin'] = sub['Total_Profit'] / sub['Total_Sales'] * 100
sub['Loss_Rate']     = sub['Loss_Rows'] / sub['Rows'] * 100
sub = sub.sort_values('Total_Profit', ascending=True)

# ── Colors ───────────────────────────────────────────────────────────────────
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
plt.suptitle('Q1 — Profitability by category & sub-category', fontsize=17,
             fontweight='500', color=DARK, y=0.98)

gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
              left=0.07, right=0.97, top=0.93, bottom=0.06)

# ── Chart 1: Sales vs Profit by category (grouped bar) ───────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])
cats = cat['Category'].tolist()
x    = np.arange(len(cats))
w    = 0.35
b1 = ax1.bar(x - w/2, cat['Total_Sales'] / 1e3, w,
             color=[CAT_COLORS[c] for c in cats], alpha=0.45,
             label='Sales ($k)', edgecolor='white')
b2 = ax1.bar(x + w/2, cat['Total_Profit'] / 1e3, w,
             color=[CAT_COLORS[c] for c in cats],
             label='Profit ($k)', edgecolor='white')
ax1.set_xticks(x)
ax1.set_xticklabels(cats, fontsize=10)
ax1.set_ylabel('USD (thousands)', fontsize=9, color=GRAY)
ax1.set_title('Sales vs profit by category ($k)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax1.set_facecolor('white')
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_color(GRAY)
ax1.tick_params(labelsize=9)
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'${bar.get_height():.0f}k', ha='center', fontsize=8, color=GRAY)
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'${bar.get_height():.0f}k', ha='center', fontsize=8,
             color=DARK, fontweight='500')
patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in cats]
ax1.legend(handles=patches, fontsize=8, frameon=False, loc='upper right')

# ── Chart 2: Profit margin % by category ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
margins     = cat['Profit_Margin']
bar_colors  = [CORAL if m < 5 else CAT_COLORS[c] for m, c in zip(margins, cats)]
bars        = ax2.bar(cats, margins, color=bar_colors, width=0.5, edgecolor='white')
ax2.axhline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax2.set_title('Profit margin % by category', fontsize=11, fontweight='500', color=DARK, pad=8)
ax2.set_facecolor('white')
ax2.set_ylabel('Margin %', fontsize=9, color=GRAY)
ax2.tick_params(labelsize=9)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars, margins):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='500', color=DARK)
ax2.set_ylim(0, margins.max() * 1.35)

# ── Chart 3: Total profit by sub-category (horizontal bar) ───────────────────
ax3 = fig.add_subplot(gs[1, 0:2])
sub_sorted  = sub.sort_values('Total_Profit')
colors_sub  = [CORAL if p < 0 else CAT_COLORS.get(c, BLUE)
               for p, c in zip(sub_sorted['Total_Profit'], sub_sorted['Category'])]
bars3 = ax3.barh(sub_sorted['Sub-Category'], sub_sorted['Total_Profit'] / 1e3,
                 color=colors_sub, edgecolor='white', height=0.65)
ax3.axvline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_title('Total profit by sub-category ($k)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax3.set_facecolor('white')
ax3.set_xlabel('Profit ($k)', fontsize=9, color=GRAY)
ax3.tick_params(labelsize=8.5)
ax3.spines[['top', 'right']].set_visible(False)
ax3.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars3, sub_sorted['Total_Profit'] / 1e3):
    xpos = val + (0.3 if val >= 0 else -0.3)
    ha   = 'left' if val >= 0 else 'right'
    ax3.text(xpos, bar.get_y() + bar.get_height() / 2,
             f'${val:.1f}k', va='center', ha=ha, fontsize=7.5, color=DARK)
patches2 = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CAT_COLORS]
patches2.append(mpatches.Patch(color=CORAL, label='Loss-making'))
ax3.legend(handles=patches2, fontsize=8, frameon=False, loc='lower right')

# ── Chart 4: Profit margin % by sub-category ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
sub_m    = sub.sort_values('Profit_Margin')
colors_m = [CORAL if v < 0 else CAT_COLORS.get(c, BLUE)
            for v, c in zip(sub_m['Profit_Margin'], sub_m['Category'])]
ax4.barh(sub_m['Sub-Category'], sub_m['Profit_Margin'],
         color=colors_m, edgecolor='white', height=0.65)
ax4.axvline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax4.set_title('Profit margin % by sub-category', fontsize=11, fontweight='500', color=DARK, pad=8)
ax4.set_facecolor('white')
ax4.set_xlabel('Margin %', fontsize=9, color=GRAY)
ax4.tick_params(labelsize=8.5)
ax4.spines[['top', 'right']].set_visible(False)
ax4.spines[['left', 'bottom']].set_color(GRAY)

# ── Chart 5: Loss rate by sub-category ───────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0:2])
sub_l    = sub.sort_values('Loss_Rate', ascending=False)
colors_l = [CORAL if r > 30 else AMBER if r > 15 else TEAL for r in sub_l['Loss_Rate']]
bars5    = ax5.bar(sub_l['Sub-Category'], sub_l['Loss_Rate'],
                   color=colors_l, edgecolor='white', width=0.6)
ax5.set_title('Loss rate by sub-category (% of rows with negative profit)',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax5.set_facecolor('white')
ax5.set_ylabel('Loss rate %', fontsize=9, color=GRAY)
ax5.tick_params(axis='x', labelsize=8.5, rotation=35)
ax5.tick_params(axis='y', labelsize=9)
ax5.spines[['top', 'right']].set_visible(False)
ax5.spines[['left', 'bottom']].set_color(GRAY)
ax5.axhline(18.7, color=GRAY, linewidth=1, linestyle=':', alpha=0.8)
ax5.text(len(sub_l) - 0.5, 19.5, 'Dataset avg 18.7%', fontsize=7.5, color=GRAY, ha='right')
for bar, val in zip(bars5, sub_l['Loss_Rate']):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.0f}%', ha='center', fontsize=7.5, color=DARK)
p_high = mpatches.Patch(color=CORAL, label='>30% loss rate')
p_med  = mpatches.Patch(color=AMBER, label='15–30%')
p_low  = mpatches.Patch(color=TEAL,  label='<15%')
ax5.legend(handles=[p_high, p_med, p_low], fontsize=8, frameon=False)

# ── Panel 6: Key findings scorecard ──────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor('#F1EFE8')
ax6.axis('off')
ax6.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax6.transAxes)

findings = [
    ("Tables",     "−$17.7k loss | 63.6% loss rate", CORAL),
    ("Bookcases",  "−$3.5k loss | 47.8% loss rate",  CORAL),
    ("Supplies",   "−$1.2k loss | 17.4% loss rate",  CORAL),
    ("Furniture",  "Only 2.5% margin overall",         AMBER),
    ("Machines",   "1.8% margin | 38% loss rate",     AMBER),
    ("Binders",    "40.3% loss rate (high discount)",  AMBER),
    ("Copiers",    "37.2% margin — top performer",     TEAL),
    ("Paper",      "43.4% margin — most consistent",   TEAL),
    ("Labels",     "44.4% margin — zero losses",       TEAL),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax6.add_patch(plt.Rectangle((0.02, y - 0.025), 0.008, 0.045,
                                color=color, transform=ax6.transAxes, clip_on=False))
    ax6.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax6.transAxes)
    ax6.text(0.07, y - 0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax6.transAxes)

plt.savefig('q1_profitability_category.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
