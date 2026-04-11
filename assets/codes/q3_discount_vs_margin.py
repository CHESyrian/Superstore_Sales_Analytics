import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from scipy import stats

df = pd.read_csv('superstore_cleaned.csv')

# ── Discount bins ─────────────────────────────────────────────────────────────
bins   = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.01]
labels = ['0%', '1-5%', '6-10%', '11-15%', '16-20%', '21-25%', '26-30%',
          '31-35%', '36-40%', '41-50%', '51-60%', '61-80%']
df['Discount_Bin'] = pd.cut(df['Discount'], bins=bins, labels=labels, right=False)

bin_stats = df.groupby('Discount_Bin', observed=True).agg(
    Avg_Margin=('Profit Margin (%)', 'mean'),
    Median_Margin=('Profit Margin (%)', 'median'),
    Loss_Rate=('Is Loss', 'mean'),
    Count=('Profit', 'count')
).reset_index()
bin_stats['Loss_Rate'] *= 100

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
plt.suptitle('Q3 — Discount vs profit margin relationship', fontsize=17,
             fontweight='500', color=DARK, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
              left=0.07, right=0.97, top=0.93, bottom=0.06)

# ── Chart 1: Scatter — all transactions ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])
for cat, grp in df.groupby('Category'):
    sample = grp.sample(min(len(grp), 700), random_state=42)
    ax1.scatter(sample['Discount'] * 100,
                sample['Profit Margin (%)'].clip(-200, 100),
                color=CAT_COLORS[cat], alpha=0.25, s=10, label=cat)
x_all = df['Discount'].values * 100
y_all = df['Profit Margin (%)'].clip(-200, 100).values
slope, intercept, r, _, _ = stats.linregress(x_all, y_all)
xline = np.linspace(0, 80, 200)
ax1.plot(xline, slope * xline + intercept, color=DARK, linewidth=1.5, linestyle='--')
ax1.axhline(0, color=CORAL, linewidth=1, linestyle=':', alpha=0.8)
ax1.axvline(20, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.6)
ax1.text(21, -170, '20% mark', fontsize=7.5, color=GRAY)
ax1.set_title('Discount rate vs profit margin — all transactions', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax1.set_xlabel('Discount (%)', fontsize=9, color=GRAY)
ax1.set_ylabel('Profit margin (%)', fontsize=9, color=GRAY)
ax1.set_facecolor('white')
ax1.tick_params(labelsize=9)
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_color(GRAY)
cat_patches = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLORS.items()]
trend_line  = mlines.Line2D([], [], color=DARK, linestyle='--', label=f'Trend (r={r:.2f})')
ax1.legend(handles=cat_patches + [trend_line], fontsize=8, frameon=False, loc='upper right')

# ── Chart 2: Avg margin by discount bin ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
bar_colors = [CORAL if v < 0 else TEAL if v > 15 else AMBER
              for v in bin_stats['Avg_Margin']]
bars2 = ax2.bar(bin_stats['Discount_Bin'], bin_stats['Avg_Margin'],
                color=bar_colors, edgecolor='white', width=0.7)
ax2.axhline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax2.set_title('Avg profit margin by\ndiscount bin (%)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax2.set_xlabel('Discount bin', fontsize=9, color=GRAY)
ax2.set_ylabel('Avg margin %', fontsize=9, color=GRAY)
ax2.set_facecolor('white')
ax2.tick_params(axis='x', rotation=45, labelsize=7.5)
ax2.tick_params(axis='y', labelsize=9)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars2, bin_stats['Avg_Margin']):
    ypos = val + 2 if val >= 0 else val - 8
    ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
             f'{val:.0f}%', ha='center', fontsize=7, color=DARK)

# ── Chart 3: Loss rate by discount bin ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0:2])
lr_colors = [CORAL if v >= 80 else AMBER if v >= 20 else TEAL
             for v in bin_stats['Loss_Rate']]
bars3 = ax3.bar(bin_stats['Discount_Bin'], bin_stats['Loss_Rate'],
                color=lr_colors, edgecolor='white', width=0.7)
ax3.axhline(50, color=CORAL, linewidth=1.2, linestyle='--', alpha=0.8)
ax3.text(len(bin_stats) - 0.5, 52, '50% threshold', fontsize=8, color=CORAL, ha='right')
ax3.set_title('Loss rate by discount bin — % of transactions losing money',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax3.set_xlabel('Discount bin', fontsize=9, color=GRAY)
ax3.set_ylabel('Loss rate %', fontsize=9, color=GRAY)
ax3.set_facecolor('white')
ax3.tick_params(axis='x', rotation=45, labelsize=8)
ax3.tick_params(axis='y', labelsize=9)
ax3.spines[['top', 'right']].set_visible(False)
ax3.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars3, bin_stats['Loss_Rate']):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f'{val:.0f}%', ha='center', fontsize=7.5, color=DARK)
ax3.set_ylim(0, 115)
p_high = mpatches.Patch(color=CORAL, label='≥80% loss rate')
p_med  = mpatches.Patch(color=AMBER, label='20–79%')
p_low  = mpatches.Patch(color=TEAL,  label='<20%')
ax3.legend(handles=[p_high, p_med, p_low], fontsize=8, frameon=False)

# ── Chart 4: Per-category trend lines ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
for cat, grp in df.groupby('Category'):
    s = grp.sample(min(len(grp), 400), random_state=1)
    ax4.scatter(s['Discount'] * 100,
                s['Profit Margin (%)'].clip(-200, 100),
                color=CAT_COLORS[cat], alpha=0.3, s=8)
    xi = grp['Discount'].values * 100
    yi = grp['Profit Margin (%)'].clip(-200, 100).values
    sl, ic, rv, _, _ = stats.linregress(xi, yi)
    xl = np.linspace(0, 80, 100)
    ax4.plot(xl, sl * xl + ic, color=CAT_COLORS[cat], linewidth=1.8,
             label=f'{cat} (r={rv:.2f})')
ax4.axhline(0, color=CORAL, linewidth=0.8, linestyle=':', alpha=0.7)
ax4.set_title('Trend by category', fontsize=11, fontweight='500', color=DARK, pad=8)
ax4.set_xlabel('Discount (%)', fontsize=9, color=GRAY)
ax4.set_ylabel('Profit margin (%)', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(labelsize=8.5)
ax4.spines[['top', 'right']].set_visible(False)
ax4.spines[['left', 'bottom']].set_color(GRAY)
ax4.legend(fontsize=8, frameon=False)

# ── Chart 5: Transaction count per bin ───────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.bar(bin_stats['Discount_Bin'], bin_stats['Count'],
        color=BLUE, edgecolor='white', width=0.7, alpha=0.85)
ax5.set_title('Transaction count\nper discount bin', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax5.set_xlabel('Discount bin', fontsize=9, color=GRAY)
ax5.set_ylabel('Count', fontsize=9, color=GRAY)
ax5.set_facecolor('white')
ax5.tick_params(axis='x', rotation=45, labelsize=7.5)
ax5.tick_params(axis='y', labelsize=9)
ax5.spines[['top', 'right']].set_visible(False)
ax5.spines[['left', 'bottom']].set_color(GRAY)
for i, (_, row) in enumerate(bin_stats.iterrows()):
    ax5.text(i, row['Count'] + 30, str(int(row['Count'])),
             ha='center', fontsize=7.5, color=DARK)

# ── Chart 6: Boxplot — margin distribution per bin ───────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
bin_order = [b for b in labels if b in df['Discount_Bin'].cat.categories]
data_box  = [df[df['Discount_Bin'] == b]['Profit Margin (%)'].clip(-200, 100).values
             for b in bin_order]
bp = ax6.boxplot(data_box, patch_artist=True, widths=0.6,
                 medianprops=dict(color=DARK, linewidth=1.5),
                 whiskerprops=dict(color=GRAY),
                 capprops=dict(color=GRAY),
                 flierprops=dict(marker='.', markersize=2, color=GRAY, alpha=0.4))
box_colors = [CORAL if np.median(d) < 0 else TEAL if np.median(d) > 15 else AMBER
              for d in data_box]
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax6.axhline(0, color=CORAL, linewidth=0.8, linestyle='--', alpha=0.7)
ax6.set_xticks(range(1, len(bin_order) + 1))
ax6.set_xticklabels(bin_order, rotation=45, fontsize=7.5)
ax6.set_title('Margin distribution per\ndiscount bin (boxplot)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax6.set_ylabel('Profit margin %', fontsize=9, color=GRAY)
ax6.set_facecolor('white')
ax6.tick_params(labelsize=8.5)
ax6.spines[['top', 'right']].set_visible(False)
ax6.spines[['left', 'bottom']].set_color(GRAY)

# ── Panel 7: Key findings ─────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor('#F1EFE8')
ax7.axis('off')
ax7.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax7.transAxes)
findings = [
    ("r = −0.864",       "Near-perfect negative correlation",  CORAL),
    ("0% discount",      "34% avg margin, 0% loss rate",       TEAL),
    ("16–20% discount",  "17.7% margin, 14% loss rate",        AMBER),
    ("26–30% discount",  "−12% margin, 93% loss rate",         CORAL),
    (">40% discount",    "100% of transactions lose money",     CORAL),
    ("Furniture",        "Steepest decline (r = −0.888)",       PURPLE),
    ("Technology",       "Most resilient (r = −0.755)",         BLUE),
    ("Safe zone",        "Discount ≤ 20% keeps margin > 0",    TEAL),
    ("Break-even",       "~25% discount = margin turns zero",  AMBER),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax7.add_patch(plt.Rectangle((0.02, y - 0.025), 0.008, 0.045,
                                color=color, transform=ax7.transAxes, clip_on=False))
    ax7.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax7.transAxes)
    ax7.text(0.07, y - 0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax7.transAxes)

plt.savefig('q3_discount_vs_margin.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
