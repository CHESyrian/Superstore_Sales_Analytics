import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

df = pd.read_csv('superstore_cleaned.csv')

# ── Aggregation ───────────────────────────────────────────────────────────────
city = df.groupby(['City', 'State', 'Region']).agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Orders=('Order ID', 'nunique'),
    Avg_Discount=('Discount', 'mean'),
    Loss_Rows=('Is Loss', 'sum'),
    Total_Rows=('Profit', 'count')
).reset_index()
city['Profit_Margin']    = (city['Total_Profit'] / city['Total_Sales'] * 100).clip(-100, 60)
city['Loss_Rate']        = city['Loss_Rows'] / city['Total_Rows'] * 100
city['Profit_per_Order'] = city['Total_Profit'] / city['Orders']

top15 = city.nlargest(15, 'Total_Profit')
bot15 = city.nsmallest(15, 'Total_Profit')
hv    = city[city['Orders'] >= 10].copy()   # cities with meaningful volume

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
plt.suptitle('Q6 — City volume vs profit margin', fontsize=17,
             fontweight='500', color=DARK, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
              left=0.07, right=0.97, top=0.93, bottom=0.05)

# ── Chart 1: Main scatter — sales vs margin, all cities with ≥10 orders ───────
ax1 = fig.add_subplot(gs[0:2, 0:2])
for reg, grp in hv.groupby('Region'):
    ax1.scatter(grp['Total_Sales'] / 1e3, grp['Profit_Margin'],
                c=REG_COLORS[reg], s=grp['Orders'] * 1.2,
                alpha=0.55, edgecolors='white', linewidth=0.5,
                label=reg, zorder=3)
ax1.axhline(0, color=CORAL, linewidth=1, linestyle='--', alpha=0.8)
label_cities = ['New York City', 'Los Angeles', 'Seattle', 'Philadelphia',
                'Houston', 'Chicago', 'San Antonio', 'Lancaster',
                'Burlington', 'Louisville', 'Detroit', 'Jacksonville']
for _, row in city[city['City'].isin(label_cities)].iterrows():
    color = CORAL if row['Total_Profit'] < 0 else DARK
    fw    = '500' if row['Total_Profit'] < 0 else 'normal'
    ax1.annotate(row['City'],
                 (row['Total_Sales'] / 1e3, row['Profit_Margin']),
                 textcoords='offset points', xytext=(5, 4),
                 fontsize=7.2, color=color, fontweight=fw)
ax1.set_title('City sales vs profit margin — bubble size = order count (cities ≥10 orders)',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax1.set_xlabel('Total sales ($k)', fontsize=9, color=GRAY)
ax1.set_ylabel('Profit margin %', fontsize=9, color=GRAY)
ax1.set_facecolor('white')
ax1.tick_params(labelsize=9)
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_color(GRAY)
ax1.text(0.99, 0.99, 'High sales, high margin', transform=ax1.transAxes,
         fontsize=7.5, color=TEAL, ha='right', va='top', style='italic')
ax1.text(0.99, 0.49, 'High sales, negative margin', transform=ax1.transAxes,
         fontsize=7.5, color=CORAL, ha='right', va='top', style='italic')
patches = [mpatches.Patch(color=v, label=k) for k, v in REG_COLORS.items()]
ax1.legend(handles=patches, fontsize=8.5, frameon=False, loc='upper left')

# ── Chart 2: Top 10 cities by profit ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
t10      = top15.head(10).sort_values('Total_Profit')
colors_t = [REG_COLORS.get(r, GRAY) for r in t10['Region']]
bars2    = ax2.barh(t10['City'], t10['Total_Profit'] / 1e3,
                    color=colors_t, edgecolor='white', height=0.65)
ax2.set_title('Top 10 cities\nby profit ($k)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax2.set_xlabel('Profit ($k)', fontsize=9, color=GRAY)
ax2.set_facecolor('white')
ax2.tick_params(axis='y', labelsize=8)
ax2.tick_params(axis='x', labelsize=8.5)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars2, t10['Total_Profit'] / 1e3):
    ax2.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f'${val:.0f}k', va='center', fontsize=7.5, color=DARK)

# ── Chart 3: Bottom 10 cities by profit ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
b10   = bot15.head(10).sort_values('Total_Profit', ascending=False)
bars3 = ax3.barh(b10['City'], b10['Total_Profit'] / 1e3,
                 color=CORAL, edgecolor='white', height=0.65)
ax3.axvline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_title('Bottom 10 cities\nby profit ($k)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax3.set_xlabel('Profit ($k)', fontsize=9, color=GRAY)
ax3.set_facecolor('white')
ax3.tick_params(axis='y', labelsize=8)
ax3.tick_params(axis='x', labelsize=8.5)
ax3.spines[['top', 'right']].set_visible(False)
ax3.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars3, b10['Total_Profit'] / 1e3):
    ax3.text(val - 0.2, bar.get_y() + bar.get_height() / 2,
             f'${val:.0f}k', va='center', ha='right',
             fontsize=7.5, color='white', fontweight='500')

# ── Chart 4: High-volume loss-making cities ───────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
hv_loss   = city[(city['Orders'] >= 20) & (city['Total_Profit'] < 0)].sort_values('Total_Profit')
colors_hv = [REG_COLORS.get(r, GRAY) for r in hv_loss['Region']]
bars4     = ax4.barh(hv_loss['City'], hv_loss['Total_Profit'] / 1e3,
                     color=colors_hv, edgecolor='white', height=0.65)
ax4.axvline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.4)
ax4.set_title('High-volume cities losing money\n(≥20 orders, negative profit)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax4.set_xlabel('Total profit ($k)', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(axis='y', labelsize=8)
ax4.tick_params(axis='x', labelsize=8.5)
ax4.spines[['top', 'right']].set_visible(False)
ax4.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars4, hv_loss['Total_Profit'] / 1e3):
    ax4.text(val - 0.15, bar.get_y() + bar.get_height() / 2,
             f'${val:.1f}k', va='center', ha='right',
             fontsize=7.5, color='white', fontweight='500')
patches = [mpatches.Patch(color=v, label=k) for k, v in REG_COLORS.items()]
ax4.legend(handles=patches, fontsize=7.5, frameon=False)

# ── Chart 5: Discount comparison — top 5 vs bottom 5 cities ──────────────────
ax5 = fig.add_subplot(gs[2, 1])
top5         = top15.head(5)[['City', 'Avg_Discount']].copy()
top5['Type'] = 'Top 5 profitable'
bot5         = bot15.head(5)[['City', 'Avg_Discount']].copy()
bot5['Type'] = 'Bottom 5 loss'
compare      = pd.concat([top5, bot5])
bar_colors   = [TEAL if t == 'Top 5 profitable' else CORAL for t in compare['Type']]
bars5        = ax5.barh(compare['City'], compare['Avg_Discount'] * 100,
                        color=bar_colors, edgecolor='white', height=0.65)
ax5.axvline(20, color=GRAY, linewidth=1, linestyle=':', alpha=0.7)
ax5.text(20.5, len(compare) - 0.5, '20% cap', fontsize=7.5, color=GRAY)
ax5.set_title('Avg discount: top 5 vs\nbottom 5 cities (%)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax5.set_xlabel('Avg discount %', fontsize=9, color=GRAY)
ax5.set_facecolor('white')
ax5.tick_params(axis='y', labelsize=8)
ax5.tick_params(axis='x', labelsize=8.5)
ax5.spines[['top', 'right']].set_visible(False)
ax5.spines[['left', 'bottom']].set_color(GRAY)
for bar, val in zip(bars5, compare['Avg_Discount'] * 100):
    ax5.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f'{val:.0f}%', va='center', fontsize=7.5, color=DARK)
p1 = mpatches.Patch(color=TEAL,  label='Top 5 profitable')
p2 = mpatches.Patch(color=CORAL, label='Bottom 5 loss')
ax5.legend(handles=[p1, p2], fontsize=7.5, frameon=False)

# ── Panel 6: Key findings ─────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor('#F1EFE8')
ax6.axis('off')
ax6.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax6.transAxes)
findings = [
    ("New York City",  "+$62k profit, 6% avg discount",   TEAL),
    ("Los Angeles",    "+$30k profit, 7% avg discount",    BLUE),
    ("Seattle",        "+$29k profit, 6% avg discount",    BLUE),
    ("Philadelphia",   "−$13.8k loss on $109k sales",      CORAL),
    ("Houston",        "−$10.2k loss, 38% avg discount",   CORAL),
    ("Chicago",        "−$6.7k loss, 38% avg discount",    CORAL),
    ("Lancaster OH",   "−87% margin — worst in dataset",   CORAL),
    ("Top cities",     "All carry ≤8% avg discount",       TEAL),
    ("Loss cities",    "All carry ≥28% avg discount",      CORAL),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax6.add_patch(plt.Rectangle((0.02, y - 0.025), 0.008, 0.045,
                                color=color, transform=ax6.transAxes, clip_on=False))
    ax6.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax6.transAxes)
    ax6.text(0.07, y - 0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax6.transAxes)

plt.savefig('q6_city_volume_margin.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
