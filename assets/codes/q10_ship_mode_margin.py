import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

df = pd.read_csv('superstore_cleaned.csv')

# ── Aggregations ──────────────────────────────────────────────────────────────
ship = df.groupby('Ship Mode').agg(
    Total_Sales=('Sales','sum'),
    Total_Profit=('Profit','sum'),
    Orders=('Order ID','nunique'),
    Avg_Discount=('Discount','mean'),
    Avg_Lead=('Lead Time (Days)','mean'),
    Loss_Rows=('Is Loss','sum'),
    Total_Rows=('Profit','count')
).reset_index()
ship['Profit_Margin']    = ship['Total_Profit'] / ship['Total_Sales'] * 100
ship['Loss_Rate']        = ship['Loss_Rows'] / ship['Total_Rows'] * 100
ship['Profit_per_Order'] = ship['Total_Profit'] / ship['Orders']

ship_seg = df.groupby(['Ship Mode','Segment']).agg(
    Total_Sales=('Sales','sum'),
    Total_Profit=('Profit','sum'),
    Orders=('Order ID','nunique'),
    Loss_Rows=('Is Loss','sum'),
    Total_Rows=('Profit','count')
).reset_index()
ship_seg['Profit_Margin'] = ship_seg['Total_Profit'] / ship_seg['Total_Sales'] * 100
ship_seg['Loss_Rate']     = ship_seg['Loss_Rows'] / ship_seg['Total_Rows'] * 100

ship_cat = df.groupby(['Ship Mode','Category']).agg(
    Total_Profit=('Profit','sum'),
    Total_Sales=('Sales','sum'),
    Loss_Rows=('Is Loss','sum'),
    Total_Rows=('Profit','count')
).reset_index()
ship_cat['Profit_Margin'] = ship_cat['Total_Profit'] / ship_cat['Total_Sales'] * 100

seg_pref = df.groupby(['Segment','Ship Mode']).size().reset_index(name='Count')
seg_pref['Pct'] = seg_pref.groupby('Segment')['Count'].transform(lambda x: x/x.sum()*100)

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE   = '#378ADD'
CORAL  = '#D85A30'
TEAL   = '#1D9E75'
AMBER  = '#BA7517'
PURPLE = '#7F77DD'
GRAY   = '#888780'
DARK   = '#2C2C2A'
LIGHT  = '#F8F7F4'
SHIP_COLORS = {'Same Day': PURPLE, 'First Class': BLUE,
               'Second Class': TEAL, 'Standard Class': AMBER}
SEG_COLORS  = {'Consumer': BLUE, 'Corporate': TEAL, 'Home Office': PURPLE}
CAT_COLORS  = {'Furniture': CORAL, 'Office Supplies': AMBER, 'Technology': TEAL}
SHIP_ORDER  = ['Same Day','First Class','Second Class','Standard Class']

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor(LIGHT)
plt.suptitle('Q10 — Ship mode vs profit margin & segment preference', fontsize=17,
             fontweight='500', color=DARK, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
              left=0.07, right=0.97, top=0.93, bottom=0.05)

x4, w4 = np.arange(len(SHIP_ORDER)), 0.25

# ── Chart 1: Profit margin boxplot by ship mode ───────────────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])
data_box = [df[df['Ship Mode']==s]['Profit Margin (%)'].clip(-150,80).values for s in SHIP_ORDER]
bp = ax1.boxplot(data_box, patch_artist=True, widths=0.55,
                 medianprops=dict(color=DARK, linewidth=2),
                 whiskerprops=dict(color=GRAY, linewidth=1),
                 capprops=dict(color=GRAY, linewidth=1),
                 flierprops=dict(marker='.', markersize=2, color=GRAY, alpha=0.4))
for patch, mode in zip(bp['boxes'], SHIP_ORDER):
    patch.set_facecolor(SHIP_COLORS[mode]); patch.set_alpha(0.65)
ax1.axhline(0, color=CORAL, linewidth=0.9, linestyle='--', alpha=0.7)
ax1.set_xticklabels(SHIP_ORDER, fontsize=9.5)
ax1.set_title('Profit margin distribution by ship mode (boxplot)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax1.set_ylabel('Profit margin %', fontsize=9, color=GRAY)
ax1.set_facecolor('white')
ax1.tick_params(labelsize=9)
ax1.spines[['top','right']].set_visible(False)
ax1.spines[['left','bottom']].set_color(GRAY)
for i, mode in enumerate(SHIP_ORDER):
    med = np.median(df[df['Ship Mode']==mode]['Profit Margin (%)'].clip(-150,80))
    ax1.text(i+1, med+3, f'{med:.0f}%', ha='center', fontsize=8.5, fontweight='500', color=DARK)

# ── Chart 2: Avg margin % vs loss rate % ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ship_ord = ship.set_index('Ship Mode').reindex(SHIP_ORDER).reset_index()
x2 = np.arange(len(SHIP_ORDER))
bars_m = ax2.bar(x2-0.2, ship_ord['Profit_Margin'], 0.35,
                 color=[SHIP_COLORS[s] for s in SHIP_ORDER], edgecolor='white')
bars_l = ax2.bar(x2+0.2, ship_ord['Loss_Rate'], 0.35,
                 color=[SHIP_COLORS[s] for s in SHIP_ORDER], alpha=0.4, edgecolor='white')
ax2.set_xticks(x2)
ax2.set_xticklabels(['Same\nDay','First\nClass','Second\nClass','Standard\nClass'], fontsize=8)
ax2.set_title('Avg margin % vs loss rate %\nby ship mode', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax2.set_facecolor('white')
ax2.tick_params(labelsize=8.5)
ax2.spines[['top','right']].set_visible(False)
ax2.spines[['left','bottom']].set_color(GRAY)
for bar, val in zip(bars_m, ship_ord['Profit_Margin']):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{val:.1f}%', ha='center', fontsize=7.5, fontweight='500', color=DARK)
for bar, val in zip(bars_l, ship_ord['Loss_Rate']):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{val:.1f}%', ha='center', fontsize=7, color=GRAY)

# ── Chart 3: Segment ship mode preference (stacked bar) ──────────────────────
ax3 = fig.add_subplot(gs[1, 0])
segs   = ['Consumer','Corporate','Home Office']
x3     = np.arange(len(segs))
bottom = np.zeros(len(segs))
for mode in SHIP_ORDER:
    vals = [seg_pref[(seg_pref['Segment']==s) & (seg_pref['Ship Mode']==mode)]['Pct'].values[0]
            for s in segs]
    bars = ax3.bar(x3, vals, bottom=bottom, color=SHIP_COLORS[mode],
                   edgecolor='white', linewidth=0.5, width=0.55, label=mode)
    for bar, val in zip(bars, vals):
        if val > 8:
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                     f'{val:.0f}%', ha='center', va='center',
                     fontsize=8, color='white', fontweight='500')
    bottom += np.array(vals)
ax3.set_xticks(x3); ax3.set_xticklabels(segs, fontsize=9)
ax3.set_title('Ship mode preference\nby segment (%)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax3.set_ylabel('%', fontsize=9, color=GRAY)
ax3.set_facecolor('white')
ax3.tick_params(labelsize=9)
ax3.spines[['top','right']].set_visible(False)
ax3.spines[['left','bottom']].set_color(GRAY)
patches = [mpatches.Patch(color=SHIP_COLORS[m], label=m) for m in SHIP_ORDER]
ax3.legend(handles=patches, fontsize=7.5, frameon=False, loc='lower right')

# ── Chart 4: Profit margin by ship mode × segment ────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for i, (seg, color) in enumerate(SEG_COLORS.items()):
    margins = [ship_seg[(ship_seg['Ship Mode']==m) & (ship_seg['Segment']==seg)]['Profit_Margin'].values[0]
               for m in SHIP_ORDER]
    ax4.bar(x4 + i*w4, margins, w4, color=color, edgecolor='white', label=seg)
ax4.set_xticks(x4 + w4)
ax4.set_xticklabels(['Same\nDay','First\nClass','Second\nClass','Standard\nClass'], fontsize=8)
ax4.set_title('Profit margin by ship mode\n× segment (%)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax4.set_ylabel('Profit margin %', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(labelsize=9)
ax4.spines[['top','right']].set_visible(False)
ax4.spines[['left','bottom']].set_color(GRAY)
ax4.legend(fontsize=8, frameon=False)

# ── Chart 5: Profit margin by ship mode × category ───────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
for i, (cat, color) in enumerate(CAT_COLORS.items()):
    margins = [ship_cat[(ship_cat['Ship Mode']==m) & (ship_cat['Category']==cat)]['Profit_Margin'].values[0]
               for m in SHIP_ORDER]
    ax5.bar(x4 + i*w4, margins, w4, color=color, edgecolor='white', label=cat)
ax5.axhline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax5.set_xticks(x4 + w4)
ax5.set_xticklabels(['Same\nDay','First\nClass','Second\nClass','Standard\nClass'], fontsize=8)
ax5.set_title('Profit margin by ship mode\n× category (%)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax5.set_ylabel('Profit margin %', fontsize=9, color=GRAY)
ax5.set_facecolor('white')
ax5.tick_params(labelsize=9)
ax5.spines[['top','right']].set_visible(False)
ax5.spines[['left','bottom']].set_color(GRAY)
ax5.legend(fontsize=8, frameon=False)

# ── Chart 6: Lead time vs profit per order ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
ax6b  = ax6.twinx()
lead  = [ship.set_index('Ship Mode').loc[m,'Avg_Lead'] for m in SHIP_ORDER]
ppo   = [ship.set_index('Ship Mode').loc[m,'Profit_per_Order'] for m in SHIP_ORDER]
bars6 = ax6.bar(np.arange(len(SHIP_ORDER)), lead,
                color=[SHIP_COLORS[m] for m in SHIP_ORDER], edgecolor='white', width=0.5)
ax6b.plot(np.arange(len(SHIP_ORDER)), ppo, color=DARK, marker='o',
          markersize=7, linewidth=2, zorder=3)
ax6.set_xticks(np.arange(len(SHIP_ORDER)))
ax6.set_xticklabels(['Same\nDay','First\nClass','Second\nClass','Standard\nClass'], fontsize=8)
ax6.set_ylabel('Avg lead time (days)', fontsize=9, color=GRAY)
ax6b.set_ylabel('Profit per order ($)', fontsize=9, color=DARK)
ax6.set_title('Lead time vs profit\nper order', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax6.set_facecolor('white')
ax6.tick_params(labelsize=9)
ax6.spines[['top']].set_visible(False)
ax6.spines[['left','bottom']].set_color(GRAY)
for bar, val in zip(bars6, lead):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f'{val:.1f}d', ha='center', fontsize=8, color=DARK)
for x, val in zip(np.arange(len(SHIP_ORDER)), ppo):
    ax6b.text(x, val+0.8, f'${val:.0f}', ha='center', fontsize=8, color=DARK)

# ── Chart 7: Total orders by ship mode ───────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
ship_ord2 = ship.set_index('Ship Mode').reindex(SHIP_ORDER).reset_index()
ax7.bar(np.arange(len(SHIP_ORDER)), ship_ord2['Orders'],
        color=[SHIP_COLORS[m] for m in SHIP_ORDER], edgecolor='white', width=0.55)
ax7.set_xticks(np.arange(len(SHIP_ORDER)))
ax7.set_xticklabels(['Same\nDay','First\nClass','Second\nClass','Standard\nClass'], fontsize=8)
ax7.set_title('Total orders by ship mode', fontsize=11, fontweight='500', color=DARK, pad=8)
ax7.set_ylabel('Order count', fontsize=9, color=GRAY)
ax7.set_facecolor('white')
ax7.tick_params(labelsize=9)
ax7.spines[['top','right']].set_visible(False)
ax7.spines[['left','bottom']].set_color(GRAY)
for i, (_, row) in enumerate(ship_ord2.iterrows()):
    ax7.text(i, row['Orders']+15, str(int(row['Orders'])), ha='center',
             fontsize=9, fontweight='500', color=DARK)

# ── Panel 8: Key findings ─────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor('#F1EFE8')
ax8.axis('off')
ax8.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax8.transAxes)
findings = [
    ("First Class",    "Best margin 13.9%, profit/order $62",     BLUE),
    ("Same Day",       "Home Office 18.5% margin — segment fit",  PURPLE),
    ("Standard Class", "77% of all orders — lowest margin 12%",   AMBER),
    ("No speed effect","Faster ≠ higher margin across all segs",  GRAY),
    ("Same Day Corp",  "Corporate Same Day only 4% margin",        CORAL),
    ("Furniture",      "2–3% margin regardless of ship mode",      CORAL),
    ("Technology",     "14–20% margin across all ship modes",      TEAL),
    ("All segments",   "~60% choose Standard Class",               AMBER),
    ("Key insight",    "Ship mode has minimal profit impact",      TEAL),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax8.add_patch(plt.Rectangle((0.02, y-0.025), 0.008, 0.045,
                                color=color, transform=ax8.transAxes, clip_on=False))
    ax8.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax8.transAxes)
    ax8.text(0.07, y-0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax8.transAxes)

plt.savefig('q10_ship_mode_margin.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
