import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

df = pd.read_csv('superstore_cleaned.csv')

# ── Aggregations ──────────────────────────────────────────────────────────────
seg = df.groupby('Segment').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Orders=('Order ID', 'nunique'),
    Customers=('Customer ID', 'nunique'),
    Avg_Discount=('Discount', 'mean'),
    Loss_Rows=('Is Loss', 'sum'),
    Total_Rows=('Profit', 'count')
).reset_index()
seg['Profit_Margin']       = seg['Total_Profit'] / seg['Total_Sales'] * 100
seg['Loss_Rate']           = seg['Loss_Rows'] / seg['Total_Rows'] * 100
seg['Profit_per_Order']    = seg['Total_Profit'] / seg['Orders']
seg['Profit_per_Customer'] = seg['Total_Profit'] / seg['Customers']

seg_cat = df.groupby(['Segment', 'Category']).agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Avg_Discount=('Discount', 'mean'),
    Loss_Rows=('Is Loss', 'sum'),
    Total_Rows=('Profit', 'count')
).reset_index()
seg_cat['Profit_Margin'] = seg_cat['Total_Profit'] / seg_cat['Total_Sales'] * 100
seg_cat['Loss_Rate']     = seg_cat['Loss_Rows'] / seg_cat['Total_Rows'] * 100

df['Order_YM'] = pd.to_datetime(df['Order Date']).dt.to_period('Q')
seg_q = df.groupby(['Segment', 'Order_YM']).agg(
    Total_Profit=('Profit', 'sum'),
    Total_Sales=('Sales', 'sum')
).reset_index()
seg_q['Order_YM'] = seg_q['Order_YM'].astype(str)

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
plt.suptitle('Q7 — Customer segment profitability', fontsize=17,
             fontweight='500', color=DARK, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
              left=0.07, right=0.97, top=0.93, bottom=0.05)

segs       = seg['Segment'].tolist()
cats       = ['Furniture', 'Office Supplies', 'Technology']
x3, w3    = np.arange(len(cats)), 0.25

# ── Chart 1: Sales vs Profit ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])
x, w = np.arange(len(segs)), 0.35
b1 = ax1.bar(x - w/2, seg['Total_Sales']/1e3, w,
             color=[SEG_COLORS[s] for s in segs], alpha=0.4, edgecolor='white')
b2 = ax1.bar(x + w/2, seg['Total_Profit']/1e3, w,
             color=[SEG_COLORS[s] for s in segs], edgecolor='white')
ax1.set_xticks(x); ax1.set_xticklabels(segs, fontsize=10)
ax1.set_ylabel('USD (thousands)', fontsize=9, color=GRAY)
ax1.set_title('Sales vs profit by segment ($k)', fontsize=11, fontweight='500', color=DARK, pad=8)
ax1.set_facecolor('white')
ax1.spines[['top','right']].set_visible(False)
ax1.spines[['left','bottom']].set_color(GRAY)
ax1.tick_params(labelsize=9)
for bar in b1:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
             f'${bar.get_height():.0f}k', ha='center', fontsize=8, color=GRAY)
for bar in b2:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
             f'${bar.get_height():.0f}k', ha='center', fontsize=8.5,
             fontweight='500', color=DARK)
patches = [mpatches.Patch(color=v, label=k) for k, v in SEG_COLORS.items()]
ax1.legend(handles=patches, fontsize=8, frameon=False)

# ── Chart 2: KPI comparison ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
x2, width = np.arange(4), 0.25
for i, (s, color) in enumerate(SEG_COLORS.items()):
    srow = seg[seg['Segment']==s].iloc[0]
    vals = [srow['Profit_Margin'], srow['Loss_Rate'],
            srow['Profit_per_Order']/10, srow['Profit_per_Customer']/20]
    ax2.bar(x2 + i*width, vals, width, color=color, edgecolor='white', label=s)
ax2.set_xticks(x2 + width)
ax2.set_xticklabels(['Margin\n%','Loss\nrate %','Profit\n/order ÷10','Profit\n/cust ÷20'], fontsize=8)
ax2.set_title('KPI comparison\n(scaled for display)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax2.set_facecolor('white')
ax2.tick_params(labelsize=8)
ax2.spines[['top','right']].set_visible(False)
ax2.spines[['left','bottom']].set_color(GRAY)
ax2.legend(fontsize=7.5, frameon=False)

# ── Chart 3: Profit margin by segment × category ─────────────────────────────
ax3 = fig.add_subplot(gs[1, 0:2])
for i, (s, color) in enumerate(SEG_COLORS.items()):
    margins = [seg_cat[(seg_cat['Segment']==s) & (seg_cat['Category']==c)]['Profit_Margin'].values[0]
               for c in cats]
    bars3 = ax3.bar(x3 + i*w3, margins, w3, color=color, edgecolor='white', label=s)
    for bar, val in zip(bars3, margins):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(0.3 if val>=0 else -1.5),
                 f'{val:.1f}%', ha='center', fontsize=7.5, color=DARK)
ax3.axhline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_xticks(x3 + w3); ax3.set_xticklabels(cats, fontsize=10)
ax3.set_ylabel('Profit margin %', fontsize=9, color=GRAY)
ax3.set_title('Profit margin by segment × category (%)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax3.set_facecolor('white')
ax3.tick_params(labelsize=9)
ax3.spines[['top','right']].set_visible(False)
ax3.spines[['left','bottom']].set_color(GRAY)
ax3.legend(fontsize=8, frameon=False)

# ── Chart 4: Loss rate by segment × category ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
for i, (s, color) in enumerate(SEG_COLORS.items()):
    lr = [seg_cat[(seg_cat['Segment']==s) & (seg_cat['Category']==c)]['Loss_Rate'].values[0]
          for c in cats]
    ax4.bar(x3 + i*w3, lr, w3, color=color, edgecolor='white', label=s)
ax4.set_xticks(x3 + w3); ax4.set_xticklabels(cats, fontsize=9)
ax4.set_ylabel('Loss rate %', fontsize=9, color=GRAY)
ax4.set_title('Loss rate by segment\n× category (%)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax4.set_facecolor('white')
ax4.tick_params(labelsize=9)
ax4.spines[['top','right']].set_visible(False)
ax4.spines[['left','bottom']].set_color(GRAY)
ax4.legend(fontsize=7.5, frameon=False)

# ── Chart 5: Quarterly profit trend ──────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0:2])
quarters = sorted(seg_q['Order_YM'].unique())
for s, color in SEG_COLORS.items():
    sdf = seg_q[seg_q['Segment']==s].set_index('Order_YM').reindex(quarters)
    ax5.plot(range(len(quarters)), sdf['Total_Profit'].values,
             marker='o', markersize=4, linewidth=1.8, color=color, label=s)
ax5.set_xticks(range(len(quarters)))
ax5.set_xticklabels(quarters, rotation=45, fontsize=7.5)
ax5.axhline(0, color=CORAL, linewidth=0.8, linestyle='--', alpha=0.6)
ax5.set_title('Quarterly profit trend by segment ($)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax5.set_ylabel('Profit ($)', fontsize=9, color=GRAY)
ax5.set_facecolor('white')
ax5.tick_params(labelsize=8.5)
ax5.spines[['top','right']].set_visible(False)
ax5.spines[['left','bottom']].set_color(GRAY)
ax5.legend(fontsize=8, frameon=False)

# ── Panel 6: Key findings ─────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor('#F1EFE8')
ax6.axis('off')
ax6.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax6.transAxes)
findings = [
    ("Home Office",  "Highest margin 14.0%, lowest loss 17.5%", PURPLE),
    ("Corporate",    "Best profit/order $60.8, margin 13.0%",    TEAL),
    ("Consumer",     "Largest revenue but lowest margin 11.6%",  BLUE),
    ("Furniture",    "All 3 segments <4% margin on Furniture",   CORAL),
    ("Consumer",     "Highest Furniture loss rate 34.8%",        CORAL),
    ("Technology",   "Best category across all 3 segments",      TEAL),
    ("Home Office",  "+$407 profit per customer — best ROI",     PURPLE),
    ("Consumer",     "+$328 profit per customer — lowest ROI",   BLUE),
    ("All segments", "Q4 spike every year — seasonal pattern",   AMBER),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax6.add_patch(plt.Rectangle((0.02, y-0.025), 0.008, 0.045,
                                color=color, transform=ax6.transAxes, clip_on=False))
    ax6.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax6.transAxes)
    ax6.text(0.07, y-0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax6.transAxes)

plt.savefig('q7_segment_profitability.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
