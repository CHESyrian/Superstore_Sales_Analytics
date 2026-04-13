import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

df = pd.read_csv('superstore_cleaned.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Order_YM']   = df['Order Date'].dt.to_period('M')
df['Order_Q']    = df['Order Date'].dt.to_period('Q')
df['Order_Y']    = df['Order Date'].dt.year
df['Quarter']    = df['Order Date'].dt.quarter

# ── Aggregations ──────────────────────────────────────────────────────────────
monthly = df.groupby('Order_YM').agg(
    Sales=('Sales','sum'), Profit=('Profit','sum'),
    Orders=('Order ID','nunique')
).reset_index()
monthly['Order_YM'] = monthly['Order_YM'].astype(str)
monthly['Margin']   = monthly['Profit'] / monthly['Sales'] * 100

yearly = df.groupby('Order_Y').agg(
    Sales=('Sales','sum'), Profit=('Profit','sum'),
    Orders=('Order ID','nunique')
).reset_index()
yearly['Margin']     = yearly['Profit'] / yearly['Sales'] * 100
yearly['Sales_YoY']  = yearly['Sales'].pct_change() * 100
yearly['Profit_YoY'] = yearly['Profit'].pct_change() * 100

q_cat = df.groupby(['Order_Q','Category'])['Profit'].sum().reset_index()
q_cat['Order_Q'] = q_cat['Order_Q'].astype(str)

q_profit = df.groupby(['Order_Y','Quarter'])['Profit'].sum().reset_index()

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
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor(LIGHT)
plt.suptitle('Q9 — Monthly & yearly sales and profit trends (2014–2017)',
             fontsize=17, fontweight='500', color=DARK, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
              left=0.06, right=0.97, top=0.93, bottom=0.05)

x_m = range(len(monthly))

# ── Chart 1: Monthly sales & profit dual-axis ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0:3])
ax1b = ax1.twinx()
ax1.fill_between(x_m, monthly['Sales']/1e3, alpha=0.15, color=BLUE)
ax1.plot(x_m, monthly['Sales']/1e3, color=BLUE, linewidth=1.6, label='Sales ($k)')
ax1b.plot(x_m, monthly['Profit'], color=TEAL, linewidth=1.8, label='Profit ($)')
ax1b.fill_between(x_m, [p if p < 0 else 0 for p in monthly['Profit']],
                  color=CORAL, alpha=0.35, label='Loss months')
ax1b.axhline(0, color=CORAL, linewidth=0.8, linestyle='--', alpha=0.6)
year_starts = [i for i, ym in enumerate(monthly['Order_YM']) if ym.endswith('-01')]
for ys in year_starts:
    ax1.axvline(ys, color=GRAY, linewidth=0.7, linestyle=':', alpha=0.6)
    ax1.text(ys+0.3, monthly['Sales'].max()/1e3*0.92, monthly['Order_YM'].iloc[ys][:4],
             fontsize=8, color=GRAY)
ax1.set_xticks(range(0, len(monthly), 3))
ax1.set_xticklabels([monthly['Order_YM'].iloc[i] for i in range(0, len(monthly), 3)],
                    rotation=45, fontsize=7.5)
ax1.set_ylabel('Sales ($k)', fontsize=9, color=BLUE)
ax1b.set_ylabel('Profit ($)', fontsize=9, color=TEAL)
ax1.set_title('Monthly sales & profit — 2014 to 2017', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax1.set_facecolor('white')
ax1.tick_params(labelsize=8.5)
ax1.spines[['top']].set_visible(False)
ax1.spines[['left','bottom']].set_color(GRAY)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, fontsize=8, frameon=False, loc='upper left')

# ── Chart 2: Yearly sales & profit ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
x2, w = np.arange(len(yearly)), 0.35
b1 = ax2.bar(x2 - w/2, yearly['Sales']/1e3, w, color=BLUE, alpha=0.45, edgecolor='white')
b2 = ax2.bar(x2 + w/2, yearly['Profit']/1e3, w, color=TEAL, edgecolor='white')
ax2.set_xticks(x2); ax2.set_xticklabels(yearly['Order_Y'], fontsize=9)
ax2.set_title('Yearly sales & profit ($k)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax2.set_ylabel('USD (thousands)', fontsize=9, color=GRAY)
ax2.set_facecolor('white')
ax2.tick_params(labelsize=9)
ax2.spines[['top','right']].set_visible(False)
ax2.spines[['left','bottom']].set_color(GRAY)
for bar in b2:
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'${bar.get_height():.0f}k', ha='center', fontsize=8, fontweight='500', color=DARK)
p1 = mpatches.Patch(color=BLUE, alpha=0.45, label='Sales ($k)')
p2 = mpatches.Patch(color=TEAL, label='Profit ($k)')
ax2.legend(handles=[p1,p2], fontsize=8, frameon=False)

# ── Chart 3: YoY growth rates ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
yoy_data = yearly.dropna(subset=['Sales_YoY'])
x3 = np.arange(len(yoy_data))
ax3.bar(x3 - 0.2, yoy_data['Sales_YoY'], 0.35, color=BLUE, edgecolor='white', label='Sales YoY %')
ax3.bar(x3 + 0.2, yoy_data['Profit_YoY'], 0.35, color=TEAL, edgecolor='white', label='Profit YoY %')
ax3.axhline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_xticks(x3); ax3.set_xticklabels(yoy_data['Order_Y'], fontsize=9)
ax3.set_title('Year-on-year growth (%)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax3.set_ylabel('Growth %', fontsize=9, color=GRAY)
ax3.set_facecolor('white')
ax3.tick_params(labelsize=9)
ax3.spines[['top','right']].set_visible(False)
ax3.spines[['left','bottom']].set_color(GRAY)
ax3.legend(fontsize=8, frameon=False)

# ── Chart 4: Profit by quarter per year ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
q_colors = {1: BLUE, 2: TEAL, 3: AMBER, 4: CORAL}
for q in [1,2,3,4]:
    qdf = q_profit[q_profit['Quarter']==q]
    ax4.plot(qdf['Order_Y'], qdf['Profit']/1e3, marker='o', markersize=5,
             linewidth=1.8, color=q_colors[q], label=f'Q{q}')
ax4.set_title('Profit by quarter per year ($k)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax4.set_ylabel('Profit ($k)', fontsize=9, color=GRAY)
ax4.set_xlabel('Year', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(labelsize=9)
ax4.spines[['top','right']].set_visible(False)
ax4.spines[['left','bottom']].set_color(GRAY)
ax4.legend(fontsize=8, frameon=False)

# ── Chart 5: Quarterly profit stacked by category ────────────────────────────
ax5 = fig.add_subplot(gs[2, 0:2])
quarters = sorted(q_cat['Order_Q'].unique())
bottom   = np.zeros(len(quarters))
for cat, color in CAT_COLORS.items():
    vals = [q_cat[(q_cat['Order_Q']==q) & (q_cat['Category']==cat)]['Profit'].values[0]
            if len(q_cat[(q_cat['Order_Q']==q) & (q_cat['Category']==cat)]) > 0 else 0
            for q in quarters]
    ax5.bar(range(len(quarters)), vals, bottom=bottom,
            color=color, edgecolor='white', linewidth=0.3, width=0.7, label=cat)
    bottom += np.array(vals)
ax5.set_xticks(range(len(quarters)))
ax5.set_xticklabels(quarters, rotation=45, fontsize=7.5)
ax5.axhline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax5.set_title('Quarterly profit stacked by category ($)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax5.set_ylabel('Profit ($)', fontsize=9, color=GRAY)
ax5.set_facecolor('white')
ax5.tick_params(labelsize=8.5)
ax5.spines[['top','right']].set_visible(False)
ax5.spines[['left','bottom']].set_color(GRAY)
patches = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLORS.items()]
ax5.legend(handles=patches, fontsize=8, frameon=False)

# ── Panel 6: Key findings ─────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor('#F1EFE8')
ax6.axis('off')
ax6.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax6.transAxes)
findings = [
    ("Revenue growth",  "Sales +51% from 2014 to 2017",           TEAL),
    ("Profit growth",   "Profit +89% from 2014 to 2017",           TEAL),
    ("Best year",       "2016: +32.7% profit YoY",                 TEAL),
    ("2015 dip",        "Sales fell 2.8% — only decline",          AMBER),
    ("Q4 dominance",    "38.6% of all profit generated in Q4",     CORAL),
    ("Jan 2015",        "Only month with negative total profit",    CORAL),
    ("Best month",      "Dec 2016: $17.9k profit, 18.4% margin",  TEAL),
    ("Margin trend",    "Margin stabilised at 12–13% from 2015",   AMBER),
    ("2017 Q4",         "Highest sales ever but margin dips to 9%",CORAL),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax6.add_patch(plt.Rectangle((0.02, y-0.025), 0.008, 0.045,
                                color=color, transform=ax6.transAxes, clip_on=False))
    ax6.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax6.transAxes)
    ax6.text(0.07, y-0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax6.transAxes)

plt.savefig('q9_time_trends.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
