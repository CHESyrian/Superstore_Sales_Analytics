import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

df = pd.read_csv('superstore_cleaned.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'])

# ── RFM computation ───────────────────────────────────────────────────────────
ref_date = df['Order Date'].max() + pd.Timedelta(days=1)  # 2017-12-31

rfm = df.groupby(['Customer ID','Customer Name','Segment','Region']).agg(
    Last_Order=('Order Date','max'),
    Frequency=('Order ID','nunique'),
    Monetary=('Profit','sum')
).reset_index()
rfm['Recency'] = (ref_date - rfm['Last_Order']).dt.days

# Score 1–5 (higher = better for all three)
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5,
                          labels=[1,2,3,4,5]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5,
                          labels=[1,2,3,4,5]).astype(int)
rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

def assign_segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    if r >= 4 and f >= 4 and m >= 4:   return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3: return 'Loyal'
    elif r >= 4 and f <= 2:             return 'New Customers'
    elif r >= 3 and f >= 2 and m <= 2: return 'Potential Loyalists'
    elif r == 3 and f <= 3:             return 'Needs Attention'
    elif r == 2 and f >= 2:             return 'At Risk'
    elif r <= 2 and f >= 4:             return 'Cant Lose Them'
    elif r <= 2 and f <= 2 and m <= 2: return 'Lost'
    else:                               return 'Hibernating'

rfm['RFM_Segment'] = rfm.apply(assign_segment, axis=1)

seg_stats = rfm.groupby('RFM_Segment').agg(
    Count=('Customer ID','count'),
    Avg_Recency=('Recency','mean'),
    Avg_Frequency=('Frequency','mean'),
    Avg_Monetary=('Monetary','mean'),
    Total_Monetary=('Monetary','sum')
).reset_index().round(1)

# Save scored file
rfm.to_csv('rfm_scored.csv', index=False)

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE   = '#378ADD'
CORAL  = '#D85A30'
TEAL   = '#1D9E75'
AMBER  = '#BA7517'
PURPLE = '#7F77DD'
GRAY   = '#888780'
DARK   = '#2C2C2A'
LIGHT  = '#F8F7F4'
SEG_COLORS = {
    'Champions':          '#1D9E75',
    'Loyal':              '#378ADD',
    'Potential Loyalists':'#7F77DD',
    'New Customers':      '#5DCAA5',
    'Needs Attention':    '#BA7517',
    'At Risk':            '#EF9F27',
    'Cant Lose Them':     '#D85A30',
    'Hibernating':        '#888780',
    'Lost':               '#A32D2D',
}
seg_order_heat = ['Champions','Loyal','Potential Loyalists','New Customers',
                  'Needs Attention','At Risk','Cant Lose Them','Hibernating','Lost']

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 20))
fig.patch.set_facecolor(LIGHT)
plt.suptitle('RFM Segmentation — Superstore Customers', fontsize=17,
             fontweight='500', color=DARK, y=0.98)
gs = GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38,
              left=0.07, right=0.97, top=0.94, bottom=0.04)

# ── Chart 1: Customer count by segment ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])
seg_order = seg_stats.sort_values('Count', ascending=False)
colors_seg = [SEG_COLORS[s] for s in seg_order['RFM_Segment']]
bars = ax1.barh(seg_order['RFM_Segment'][::-1], seg_order['Count'][::-1],
                color=colors_seg[::-1], edgecolor='white', height=0.7)
ax1.set_title('Customer count by RFM segment', fontsize=11, fontweight='500', color=DARK, pad=8)
ax1.set_xlabel('Number of customers', fontsize=9, color=GRAY)
ax1.set_facecolor('white')
ax1.tick_params(axis='y', labelsize=9)
ax1.tick_params(axis='x', labelsize=8.5)
ax1.spines[['top','right']].set_visible(False)
ax1.spines[['left','bottom']].set_color(GRAY)
for bar, (_, row) in zip(bars, seg_order[::-1].iterrows()):
    pct = row['Count'] / len(rfm) * 100
    ax1.text(bar.get_width()+4, bar.get_y()+bar.get_height()/2,
             f'{int(row["Count"])}  ({pct:.1f}%)', va='center', fontsize=8.5, color=DARK)
ax1.set_xlim(0, seg_order['Count'].max()*1.28)

# ── Chart 2: Total profit by segment ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
mon_order  = seg_stats.sort_values('Total_Monetary', ascending=True)
bar_colors2 = [SEG_COLORS[s] if v >= 0 else CORAL
               for s, v in zip(mon_order['RFM_Segment'], mon_order['Total_Monetary'])]
ax2.barh(mon_order['RFM_Segment'], mon_order['Total_Monetary']/1e3,
         color=bar_colors2, edgecolor='white', height=0.7)
ax2.axvline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax2.set_title('Total profit contribution\nby segment ($k)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax2.set_xlabel('Total profit ($k)', fontsize=9, color=GRAY)
ax2.set_facecolor('white')
ax2.tick_params(axis='y', labelsize=8.5)
ax2.tick_params(axis='x', labelsize=8.5)
ax2.spines[['top','right']].set_visible(False)
ax2.spines[['left','bottom']].set_color(GRAY)

# ── Chart 3: RFM score heatmap ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0:2])
heat_data = rfm.groupby('RFM_Segment')[['R_Score','F_Score','M_Score']].mean().round(2)
heat_data.columns = ['Recency score', 'Frequency score', 'Monetary score']
heat_data = heat_data.reindex([s for s in seg_order_heat if s in heat_data.index])
sns.heatmap(heat_data, ax=ax3, cmap='RdYlGn', vmin=1, vmax=5,
            annot=True, fmt='.2f', linewidths=0.5, linecolor=LIGHT,
            cbar_kws={'label':'Score (1=low, 5=high)', 'shrink':0.8},
            annot_kws={'size':9.5})
ax3.set_title('Average R / F / M scores by segment (1 = low, 5 = high)',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax3.tick_params(axis='x', labelsize=9.5)
ax3.tick_params(axis='y', rotation=0, labelsize=9)

# ── Chart 4: Avg profit per customer by segment ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
avg_mon    = seg_stats.sort_values('Avg_Monetary', ascending=True)
bar_colors4 = [SEG_COLORS[s] if v >= 0 else CORAL
               for s, v in zip(avg_mon['RFM_Segment'], avg_mon['Avg_Monetary'])]
ax4.barh(avg_mon['RFM_Segment'], avg_mon['Avg_Monetary'],
         color=bar_colors4, edgecolor='white', height=0.7)
ax4.axvline(0, color=DARK, linewidth=0.8, linestyle='--', alpha=0.5)
ax4.set_title('Avg profit per customer\nby segment ($)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax4.set_xlabel('Avg profit ($)', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(axis='y', labelsize=8.5)
ax4.tick_params(axis='x', labelsize=8.5)
ax4.spines[['top','right']].set_visible(False)
ax4.spines[['left','bottom']].set_color(GRAY)

# ── Chart 5: Recency vs Frequency scatter ────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0:2])
for seg, color in SEG_COLORS.items():
    sdf = rfm[rfm['RFM_Segment'] == seg]
    if len(sdf) == 0:
        continue
    sizes = (sdf['Monetary'].clip(0, 5000) / 5000 * 180 + 20)
    ax5.scatter(sdf['Recency'], sdf['Frequency'],
                c=color, s=sizes, alpha=0.55,
                edgecolors='white', linewidth=0.4,
                label=seg, zorder=3)
ax5.set_title('Recency vs frequency — bubble size = profit contribution',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax5.set_xlabel('Recency (days since last order — lower is better)', fontsize=9, color=GRAY)
ax5.set_ylabel('Frequency (unique orders)', fontsize=9, color=GRAY)
ax5.set_facecolor('white')
ax5.tick_params(labelsize=8.5)
ax5.spines[['top','right']].set_visible(False)
ax5.spines[['left','bottom']].set_color(GRAY)
ax5.legend(fontsize=7.5, frameon=False, loc='upper right',
           markerscale=0.8, ncol=2)

# ── Chart 6: Region mix per segment ──────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
cross2     = pd.crosstab(rfm['RFM_Segment'], rfm['Region'])
cross2_pct = cross2.div(cross2.sum(axis=1), axis=0) * 100
cross2_pct = cross2_pct.reindex([s for s in seg_order_heat if s in cross2_pct.index])
reg_colors = {'Central': CORAL, 'East': TEAL, 'South': AMBER, 'West': BLUE}
bottom = np.zeros(len(cross2_pct))
for region in ['West','East','South','Central']:
    if region in cross2_pct.columns:
        vals = cross2_pct[region].values
        ax6.bar(range(len(cross2_pct)), vals, bottom=bottom,
                color=reg_colors[region], edgecolor='white', linewidth=0.4,
                width=0.7, label=region)
        bottom += vals
ax6.set_xticks(range(len(cross2_pct)))
ax6.set_xticklabels(cross2_pct.index, rotation=45, ha='right', fontsize=7.5)
ax6.set_title('Region mix\nper RFM segment (%)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax6.set_ylabel('%', fontsize=9, color=GRAY)
ax6.set_facecolor('white')
ax6.tick_params(labelsize=8.5)
ax6.spines[['top','right']].set_visible(False)
ax6.spines[['left','bottom']].set_color(GRAY)
patches = [mpatches.Patch(color=v, label=k) for k, v in reg_colors.items()]
ax6.legend(handles=patches, fontsize=7.5, frameon=False)

# ── Chart 7: RFM score distribution ──────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
ax7.hist(rfm['RFM_Score'], bins=13, range=(3,16),
         color=BLUE, edgecolor='white', linewidth=0.5, alpha=0.85)
ax7.axvline(rfm['RFM_Score'].mean(), color=CORAL, linewidth=1.5,
            linestyle='--', label=f'Mean: {rfm["RFM_Score"].mean():.1f}')
ax7.set_title('RFM score distribution\n(3 = worst, 15 = best)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax7.set_xlabel('RFM score', fontsize=9, color=GRAY)
ax7.set_ylabel('Customer count', fontsize=9, color=GRAY)
ax7.set_facecolor('white')
ax7.tick_params(labelsize=8.5)
ax7.spines[['top','right']].set_visible(False)
ax7.spines[['left','bottom']].set_color(GRAY)
ax7.legend(fontsize=8, frameon=False)

# ── Chart 8: Avg recency vs frequency per segment ────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
rf_data = seg_stats.set_index('RFM_Segment').reindex(
    [s for s in seg_order_heat if s in seg_stats['RFM_Segment'].values])
x8  = np.arange(len(rf_data))
ax8b = ax8.twinx()
ax8.bar(x8 - 0.2, rf_data['Avg_Recency'], 0.35,
        color=[SEG_COLORS[s] for s in rf_data.index], alpha=0.5, edgecolor='white')
ax8b.bar(x8 + 0.2, rf_data['Avg_Frequency'], 0.35,
         color=[SEG_COLORS[s] for s in rf_data.index], edgecolor='white')
ax8.set_xticks(x8)
ax8.set_xticklabels(rf_data.index, rotation=45, ha='right', fontsize=7.5)
ax8.set_ylabel('Avg recency (days)', fontsize=9, color=GRAY)
ax8b.set_ylabel('Avg frequency (orders)', fontsize=9, color=DARK)
ax8.set_title('Avg recency vs frequency\nper segment', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax8.set_facecolor('white')
ax8.tick_params(labelsize=8)
ax8.spines[['top']].set_visible(False)
ax8.spines[['left','bottom']].set_color(GRAY)
p1 = mpatches.Patch(color=GRAY, alpha=0.5, label='Recency (days, lower=better)')
p2 = mpatches.Patch(color=GRAY,             label='Frequency (orders, higher=better)')
ax8.legend(handles=[p1,p2], fontsize=7, frameon=False)

# ── Panel 9: Key findings ─────────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[3, 2])
ax9.set_facecolor('#F1EFE8')
ax9.axis('off')
ax9.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax9.transAxes)
findings = [
    ("Champions",         "330 customers, $138k profit, avg 82d", '#1D9E75'),
    ("Loyal",             "448 customers, $103k profit, avg 209d", BLUE),
    ("At Risk",           "380 customers — need re-engagement",    '#EF9F27'),
    ("Cant Lose Them",    "62 high-freq customers going quiet",    CORAL),
    ("Lost",              "275 customers, avg −$58 each",          '#A32D2D'),
    ("Pot. Loyalists",    "372 customers, avg −$152 each (!)",     PURPLE),
    ("Champions + Loyal", "778 customers = $241k profit (84%)",   '#1D9E75'),
    ("At Risk + Lost",    "655 customers actively hurting profit", CORAL),
    ("Action priority",   "Win back At Risk before they are Lost", AMBER),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax9.add_patch(plt.Rectangle((0.02, y-0.025), 0.008, 0.045,
                                color=color, transform=ax9.transAxes, clip_on=False))
    ax9.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax9.transAxes)
    ax9.text(0.07, y-0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax9.transAxes)

plt.savefig('rfm_segmentation.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
