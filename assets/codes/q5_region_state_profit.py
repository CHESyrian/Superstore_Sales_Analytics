import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

df = pd.read_csv('superstore_cleaned.csv')

# ── Aggregations ──────────────────────────────────────────────────────────────
region = df.groupby('Region').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Orders=('Order ID', 'nunique'),
    Avg_Discount=('Discount', 'mean'),
    Loss_Rows=('Is Loss', 'sum'),
    Total_Rows=('Profit', 'count')
).reset_index()
region['Profit_Margin']    = region['Total_Profit'] / region['Total_Sales'] * 100
region['Loss_Rate']        = region['Loss_Rows'] / region['Total_Rows'] * 100
region['Profit_per_Order'] = region['Total_Profit'] / region['Orders']

state = df.groupby(['State', 'Region']).agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Orders=('Order ID', 'nunique'),
    Avg_Discount=('Discount', 'mean'),
    Loss_Rows=('Is Loss', 'sum'),
    Total_Rows=('Profit', 'count')
).reset_index()
state['Profit_Margin'] = state['Total_Profit'] / state['Total_Sales'] * 100
state['Loss_Rate']     = state['Loss_Rows'] / state['Total_Rows'] * 100

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE   = '#378ADD'
CORAL  = '#D85A30'
TEAL   = '#1D9E75'
AMBER  = '#BA7517'
GRAY   = '#888780'
DARK   = '#2C2C2A'
LIGHT  = '#F8F7F4'
REG_COLORS = {'West': BLUE, 'East': TEAL, 'Central': CORAL, 'South': AMBER}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor(LIGHT)
plt.suptitle('Q5 — Sales & profit by region and state', fontsize=17,
             fontweight='500', color=DARK, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
              left=0.07, right=0.97, top=0.93, bottom=0.05)

regs = region['Region'].tolist()

# ── Chart 1: Region — Sales vs Profit grouped bar ────────────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])
x, w = np.arange(len(regs)), 0.35
b1 = ax1.bar(x - w/2, region['Total_Sales'] / 1e3, w,
             color=[REG_COLORS[r] for r in regs], alpha=0.4, edgecolor='white')
b2 = ax1.bar(x + w/2, region['Total_Profit'] / 1e3, w,
             color=[REG_COLORS[r] for r in regs], edgecolor='white')
ax1.set_xticks(x)
ax1.set_xticklabels(regs, fontsize=10)
ax1.set_ylabel('USD (thousands)', fontsize=9, color=GRAY)
ax1.set_title('Sales vs profit by region ($k)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax1.set_facecolor('white')
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_color(GRAY)
ax1.tick_params(labelsize=9)
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             f'${bar.get_height():.0f}k', ha='center', fontsize=8, fontweight='500', color=DARK)
patches = [mpatches.Patch(color=v, label=k) for k, v in REG_COLORS.items()]
ax1.legend(handles=patches, fontsize=8, frameon=False)

# ── Chart 2: Region — margin % vs loss rate (twin axis) ──────────────────────
ax2 = fig.add_subplot(gs[0, 2])
x2   = np.arange(len(regs))
ax2b = ax2.twinx()
bars_m = ax2.bar(x2 - 0.2, region['Profit_Margin'], 0.35,
                 color=[REG_COLORS[r] for r in regs], edgecolor='white')
bars_l = ax2b.bar(x2 + 0.2, region['Loss_Rate'], 0.35,
                  color=[REG_COLORS[r] for r in regs], alpha=0.45, edgecolor='white')
ax2.set_xticks(x2)
ax2.set_xticklabels(regs, fontsize=9)
ax2.set_ylabel('Profit margin %', fontsize=9, color=DARK)
ax2b.set_ylabel('Loss rate %', fontsize=9, color=GRAY)
ax2.set_title('Margin % vs loss rate\nby region', fontsize=11, fontweight='500', color=DARK, pad=8)
ax2.set_facecolor('white')
ax2.spines[['top']].set_visible(False)
ax2.tick_params(labelsize=9)
for bar, val in zip(bars_m, region['Profit_Margin']):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', fontsize=8, fontweight='500', color=DARK)
for bar, val in zip(bars_l, region['Loss_Rate']):
    ax2b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
              f'{val:.1f}%', ha='center', fontsize=7.5, color=GRAY)

# ── Chart 3: State — total profit, top 15 + bottom 10 ────────────────────────
ax3 = fig.add_subplot(gs[1, 0:2])
top15    = state.nlargest(15, 'Total_Profit')
bot10    = state.nsmallest(10, 'Total_Profit')
combined = pd.concat([bot10, top15]).drop_duplicates().sort_values('Total_Profit')
bar_colors = [CORAL if p < 0 else REG_COLORS.get(r, GRAY)
              for p, r in zip(combined['Total_Profit'], combined['Region'])]
bars3 = ax3.barh(combined['State'], combined['Total_Profit'] / 1e3,
                 color=bar_colors, edgecolor='white', height=0.65)
ax3.axvline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_title('Total profit by state — top 15 and bottom 10 ($k)',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax3.set_xlabel('Total profit ($k)', fontsize=9, color=GRAY)
ax3.set_facecolor('white')
ax3.tick_params(axis='y', labelsize=8)
ax3.tick_params(axis='x', labelsize=9)
ax3.spines[['top', 'right']].set_visible(False)
ax3.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars3, combined['Total_Profit'] / 1e3):
    xpos = val + 0.5 if val >= 0 else val - 0.5
    ha   = 'left' if val >= 0 else 'right'
    ax3.text(xpos, bar.get_y() + bar.get_height() / 2,
             f'${val:.1f}k', va='center', ha=ha, fontsize=7.5, color=DARK)
patches = [mpatches.Patch(color=v, label=k) for k, v in REG_COLORS.items()]
patches.append(mpatches.Patch(color=CORAL, label='Loss-making'))
ax3.legend(handles=patches, fontsize=7.5, frameon=False, loc='lower right')

# ── Chart 4: State — sales vs margin scatter (bubble = orders) ───────────────
ax4 = fig.add_subplot(gs[1, 2])
sc_colors = [REG_COLORS.get(r, GRAY) for r in state['Region']]
ax4.scatter(state['Total_Sales'] / 1e3, state['Profit_Margin'],
            c=sc_colors, s=state['Orders'] * 0.8, alpha=0.7,
            edgecolors='white', linewidth=0.6, zorder=3)
ax4.axhline(0, color=CORAL, linewidth=0.8, linestyle='--', alpha=0.7)
notable = ['Texas', 'California', 'New York', 'Ohio', 'Pennsylvania', 'Illinois', 'Florida']
for _, row in state[state['State'].isin(notable)].iterrows():
    ax4.annotate(row['State'],
                 (row['Total_Sales'] / 1e3, row['Profit_Margin']),
                 textcoords='offset points', xytext=(4, 4), fontsize=7, color=DARK)
ax4.set_title('Sales vs margin % by state\n(bubble = order count)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax4.set_xlabel('Total sales ($k)', fontsize=9, color=GRAY)
ax4.set_ylabel('Profit margin %', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(labelsize=8.5)
ax4.spines[['top', 'right']].set_visible(False)
ax4.spines[['left', 'bottom']].set_color(GRAY)
patches = [mpatches.Patch(color=v, label=k) for k, v in REG_COLORS.items()]
ax4.legend(handles=patches, fontsize=7.5, frameon=False)

# ── Chart 5: Avg discount by region ──────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
disc_colors = [CORAL if d > 0.20 else AMBER if d > 0.15 else TEAL
               for d in region['Avg_Discount']]
bars5 = ax5.bar(regs, region['Avg_Discount'] * 100,
                color=disc_colors, edgecolor='white', width=0.5)
ax5.axhline(15.6, color=GRAY, linewidth=1, linestyle=':', alpha=0.8)
ax5.text(3.4, 16.3, 'Avg 15.6%', fontsize=7.5, color=GRAY, ha='right')
ax5.set_title('Avg discount by region (%)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax5.set_ylabel('Avg discount %', fontsize=9, color=GRAY)
ax5.set_facecolor('white')
ax5.tick_params(labelsize=9)
ax5.spines[['top', 'right']].set_visible(False)
ax5.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars5, region['Avg_Discount'] * 100):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='500', color=DARK)
ax5.set_ylim(0, region['Avg_Discount'].max() * 130)

# ── Chart 6: Profit per order by region ──────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
bars6 = ax6.bar(regs, region['Profit_per_Order'],
                color=[REG_COLORS[r] for r in regs], edgecolor='white', width=0.5)
ax6.set_title('Profit per order by region ($)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax6.set_ylabel('$ profit per order', fontsize=9, color=GRAY)
ax6.set_facecolor('white')
ax6.tick_params(labelsize=9)
ax6.spines[['top', 'right']].set_visible(False)
ax6.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars6, region['Profit_per_Order']):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'${val:.1f}', ha='center', fontsize=9, fontweight='500', color=DARK)

# ── Panel 7: Key findings ─────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor('#F1EFE8')
ax7.axis('off')
ax7.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax7.transAxes)
findings = [
    ("West",         "Best margin 14.9%, lowest loss 9.9%",  BLUE),
    ("East",         "Highest sales $679k, 13.5% margin",    TEAL),
    ("Central",      "Worst: 7.9% margin, 31.9% loss rate",  CORAL),
    ("Texas",        "Biggest loser: −$25.7k total loss",    CORAL),
    ("Ohio",         "−$17.0k on $78k sales (−21.7%)",       CORAL),
    ("Pennsylvania", "−$15.6k on $117k sales (−13.4%)",      CORAL),
    ("California",   "Top earner: +$76.4k profit",            BLUE),
    ("New York",     "+$74.0k profit, 23.8% margin",          TEAL),
    ("Central avg",  "24% discount — double West's 11%",      CORAL),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax7.add_patch(plt.Rectangle((0.02, y - 0.025), 0.008, 0.045,
                                color=color, transform=ax7.transAxes, clip_on=False))
    ax7.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax7.transAxes)
    ax7.text(0.07, y - 0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax7.transAxes)

plt.savefig('q5_region_state_profit.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
