import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

df = pd.read_csv('superstore_cleaned.csv')

# ── Aggregation ───────────────────────────────────────────────────────────────
cust = df.groupby(['Customer ID','Customer Name','Segment','Region']).agg(
    Total_Sales=('Sales','sum'),
    Total_Profit=('Profit','sum'),
    Orders=('Order ID','nunique'),
    Avg_Discount=('Discount','mean'),
    Loss_Rows=('Is Loss','sum'),
    Total_Rows=('Profit','count')
).reset_index()
cust['Profit_Margin']    = cust['Total_Profit'] / cust['Total_Sales'] * 100
cust['Loss_Rate']        = cust['Loss_Rows'] / cust['Total_Rows'] * 100
cust['Profit_per_Order'] = cust['Total_Profit'] / cust['Orders']

top10 = cust.nlargest(10, 'Total_Profit').reset_index(drop=True)
bot10 = cust.nsmallest(10, 'Total_Profit').reset_index(drop=True)

# Pareto curve
cust_sorted = cust.sort_values('Total_Profit', ascending=False).reset_index(drop=True)
cust_sorted['Cumulative_Profit'] = cust_sorted['Total_Profit'].cumsum()
cust_sorted['Customer_Pct']      = (cust_sorted.index + 1) / len(cust_sorted) * 100
total_profit = cust_sorted['Total_Profit'].sum()
cust_sorted['Cumulative_Pct']    = cust_sorted['Cumulative_Profit'] / total_profit * 100

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE   = '#378ADD'
CORAL  = '#D85A30'
TEAL   = '#1D9E75'
AMBER  = '#BA7517'
PURPLE = '#7F77DD'
GRAY   = '#888780'
DARK   = '#2C2C2A'
LIGHT  = '#F8F7F4'
SEG_COLORS = {'Consumer': BLUE, 'Corporate': TEAL, 'Home Office': PURPLE}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor(LIGHT)
plt.suptitle('Q8 — Top & bottom customers by profit contribution', fontsize=17,
             fontweight='500', color=DARK, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
              left=0.07, right=0.97, top=0.93, bottom=0.05)
patches = [mpatches.Patch(color=v, label=k) for k, v in SEG_COLORS.items()]

# ── Chart 1: Top 10 customers by profit ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0:2, 0])
colors_t = [SEG_COLORS.get(s, GRAY) for s in top10['Segment']]
bars1    = ax1.barh(top10['Customer Name'][::-1], top10['Total_Profit'][::-1],
                    color=colors_t[::-1], edgecolor='white', height=0.65)
ax1.set_title('Top 10 customers\nby total profit ($)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax1.set_xlabel('Total profit ($)', fontsize=9, color=GRAY)
ax1.set_facecolor('white')
ax1.tick_params(axis='y', labelsize=8.5)
ax1.tick_params(axis='x', labelsize=8.5)
ax1.spines[['top','right']].set_visible(False)
ax1.spines[['left','bottom']].set_color(GRAY)
for bar, (_, row) in zip(bars1, top10[::-1].iterrows()):
    ax1.text(bar.get_width()+80, bar.get_y()+bar.get_height()/2,
             f'${bar.get_width():,.0f}  {row["Profit_Margin"]:.0f}% margin',
             va='center', fontsize=7, color=DARK)
ax1.legend(handles=patches, fontsize=7.5, frameon=False)

# ── Chart 2: Bottom 10 customers by profit ────────────────────────────────────
ax2 = fig.add_subplot(gs[0:2, 1])
bars2 = ax2.barh(bot10['Customer Name'][::-1], bot10['Total_Profit'][::-1],
                 color=CORAL, edgecolor='white', height=0.65)
ax2.axvline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax2.set_title('Bottom 10 customers\nby total profit ($)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax2.set_xlabel('Total profit ($)', fontsize=9, color=GRAY)
ax2.set_facecolor('white')
ax2.tick_params(axis='y', labelsize=8.5)
ax2.tick_params(axis='x', labelsize=8.5)
ax2.spines[['top','right']].set_visible(False)
ax2.spines[['left','bottom']].set_color(GRAY)
for bar, (_, row) in zip(bars2, bot10[::-1].iterrows()):
    ax2.text(bar.get_width()-100, bar.get_y()+bar.get_height()/2,
             f'${bar.get_width():,.0f}  {row["Avg_Discount"]*100:.0f}% disc',
             va='center', ha='right', fontsize=7, color='white', fontweight='500')
ax2.legend(handles=patches, fontsize=7.5, frameon=False)

# ── Chart 3: Pareto curve ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0:2, 2])
ax3.plot(cust_sorted['Customer_Pct'], cust_sorted['Cumulative_Pct'],
         color=BLUE, linewidth=2, zorder=3)
ax3.fill_between(cust_sorted['Customer_Pct'], cust_sorted['Cumulative_Pct'],
                 alpha=0.12, color=BLUE)
for pct, color, label in [(10, CORAL, '10% customers\n= 82.6% profit'),
                           (20, AMBER, '20% customers\n= 103% profit')]:
    idx     = cust_sorted[cust_sorted['Customer_Pct'] <= pct].index[-1]
    cum_val = cust_sorted.loc[idx, 'Cumulative_Pct']
    ax3.axvline(pct, color=color, linewidth=1, linestyle='--', alpha=0.7)
    ax3.axhline(cum_val, color=color, linewidth=1, linestyle='--', alpha=0.7)
    ax3.scatter([pct], [cum_val], color=color, s=60, zorder=5)
    ax3.text(pct+0.5, cum_val-8, label, fontsize=7.5, color=color)
ax3.axhline(100, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.6)
ax3.set_title('Customer profit Pareto curve\n(cumulative % of total profit)',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax3.set_xlabel('% of customers (ranked by profit)', fontsize=9, color=GRAY)
ax3.set_ylabel('Cumulative % of total profit', fontsize=9, color=GRAY)
ax3.set_facecolor('white')
ax3.tick_params(labelsize=8.5)
ax3.spines[['top','right']].set_visible(False)
ax3.spines[['left','bottom']].set_color(GRAY)
ax3.set_xlim(0, 100)

# ── Chart 4: Customer profit distribution histogram ───────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
profit_clipped = cust['Total_Profit'].clip(-8000, 10000)
ax4.hist(profit_clipped[profit_clipped >= 0], bins=40,
         color=TEAL, alpha=0.75, label='Profitable', edgecolor='white', linewidth=0.3)
ax4.hist(profit_clipped[profit_clipped < 0], bins=20,
         color=CORAL, alpha=0.75, label='Loss', edgecolor='white', linewidth=0.3)
ax4.axvline(0, color=DARK, linewidth=0.9, linestyle='--', alpha=0.6)
ax4.set_title('Customer profit distribution ($)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax4.set_xlabel('Total profit ($)', fontsize=9, color=GRAY)
ax4.set_ylabel('Customer count', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(labelsize=8.5)
ax4.spines[['top','right']].set_visible(False)
ax4.spines[['left','bottom']].set_color(GRAY)
ax4.legend(fontsize=8, frameon=False)

# ── Chart 5: Customer avg discount vs total profit scatter ────────────────────
ax5 = fig.add_subplot(gs[2, 1])
sc_colors = [SEG_COLORS.get(s, GRAY) for s in cust['Segment']]
ax5.scatter(cust['Avg_Discount']*100,
            cust['Total_Profit'].clip(-8000, 10000),
            c=sc_colors, s=cust['Orders']*4, alpha=0.35,
            edgecolors='white', linewidth=0.4, zorder=2)
ax5.axhline(0, color=CORAL, linewidth=0.8, linestyle='--', alpha=0.7)
ax5.axvline(20, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.6)
for _, row in top10.head(3).iterrows():
    ax5.annotate(row['Customer Name'].split()[0],
                 (row['Avg_Discount']*100, min(row['Total_Profit'], 9500)),
                 fontsize=7, color=DARK, textcoords='offset points', xytext=(3,3))
for _, row in bot10.head(3).iterrows():
    ax5.annotate(row['Customer Name'].split()[0],
                 (row['Avg_Discount']*100, max(row['Total_Profit'], -7500)),
                 fontsize=7, color=CORAL, fontweight='500',
                 textcoords='offset points', xytext=(3,-8))
ax5.set_title('Customer avg discount vs\ntotal profit (bubble = orders)',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax5.set_xlabel('Avg discount %', fontsize=9, color=GRAY)
ax5.set_ylabel('Total profit ($)', fontsize=9, color=GRAY)
ax5.set_facecolor('white')
ax5.tick_params(labelsize=8.5)
ax5.spines[['top','right']].set_visible(False)
ax5.spines[['left','bottom']].set_color(GRAY)
ax5.legend(handles=patches, fontsize=7.5, frameon=False)

# ── Panel 6: Key findings ─────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor('#F1EFE8')
ax6.axis('off')
ax6.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax6.transAxes)
findings = [
    ("Tamara Chand",    "Top customer: +$8.7k, 47% margin",         TEAL),
    ("Raymond Buch",    "+$6.8k profit on just 2 orders",           TEAL),
    ("Cindy Stewart",   "Worst: −$6.9k on 1 order at 53% disc",     CORAL),
    ("Henry Goldwyn",   "−$2.9k at 80% discount — max in dataset",  CORAL),
    ("Luke Foster",     "−$3.8k at 66% discount",                   CORAL),
    ("Top 10%",         "250 customers = 82.6% of total profit",    TEAL),
    ("Bottom 10%",      "250 customers = −$108k losses",            CORAL),
    ("No repeat losers","0 customers with ≥10 orders still losing", AMBER),
    ("Loss customers",  "All carry ≥33% avg discount",              CORAL),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax6.add_patch(plt.Rectangle((0.02, y-0.025), 0.008, 0.045,
                                color=color, transform=ax6.transAxes, clip_on=False))
    ax6.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax6.transAxes)
    ax6.text(0.07, y-0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax6.transAxes)

plt.savefig('q8_customer_profit.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
