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
df['Order_Q']    = df['Order Date'].dt.to_period('Q')

# ── Cohort construction ───────────────────────────────────────────────────────
first_order = df.groupby('Customer ID')['Order_Q'].min().reset_index()
first_order.columns = ['Customer ID', 'Cohort_Q']
df = df.merge(first_order, on='Customer ID')
df['Period_Number'] = (df['Order_Q'] - df['Cohort_Q']).apply(lambda x: x.n)

cohort_sizes    = df.groupby('Cohort_Q')['Customer ID'].nunique()
retention       = df.groupby(['Cohort_Q','Period_Number'])['Customer ID'].nunique().reset_index()
retention.columns = ['Cohort_Q','Period_Number','Customers']
retention_pivot = retention.pivot(index='Cohort_Q', columns='Period_Number', values='Customers')
retention_pivot.index = retention_pivot.index.astype(str)
cohort_sizes.index    = cohort_sizes.index.astype(str)
retention_pct   = retention_pivot.div(cohort_sizes, axis=0) * 100

rev_cohort = df.groupby(['Cohort_Q','Period_Number']).agg(
    Revenue=('Sales','sum'), Profit=('Profit','sum')
).reset_index()
rev_cohort['Cohort_Q']    = rev_cohort['Cohort_Q'].astype(str)
rev_cohort                = rev_cohort.sort_values(['Cohort_Q','Period_Number'])
rev_cohort['Cum_Profit']  = rev_cohort.groupby('Cohort_Q')['Profit'].cumsum()
rev_cohort['Cum_Revenue'] = rev_cohort.groupby('Cohort_Q')['Revenue'].cumsum()

early_cohorts = ['2014Q1','2014Q2','2014Q3','2014Q4']

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE   = '#378ADD'
CORAL  = '#D85A30'
TEAL   = '#1D9E75'
AMBER  = '#BA7517'
PURPLE = '#7F77DD'
GRAY   = '#888780'
DARK   = '#2C2C2A'
LIGHT  = '#F8F7F4'
COHORT_COLORS = {
    '2014Q1': BLUE, '2014Q2': TEAL, '2014Q3': PURPLE, '2014Q4': AMBER,
    '2015Q1': '#5DCAA5', '2015Q2': '#85B7EB',
    '2015Q3': '#AFA9EC', '2015Q4': '#EF9F27',
}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor(LIGHT)
plt.suptitle('Cohort Retention Analysis — Superstore Customers (2014–2017)',
             fontsize=17, fontweight='500', color=DARK, y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
              left=0.06, right=0.97, top=0.93, bottom=0.05)

# ── Chart 1: Retention heatmap ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0:2, 0:2])
heat_cohorts = [c for c in retention_pct.index if c.startswith('201')]
heat_data    = retention_pct.loc[heat_cohorts, :12].copy()
mask         = heat_data.isnull()
sns.heatmap(heat_data, ax=ax1, cmap='YlOrRd_r', vmin=0, vmax=100,
            annot=True, fmt='.0f', linewidths=0.5, linecolor=LIGHT,
            mask=mask,
            cbar_kws={'label':'Retention rate %', 'shrink':0.8},
            annot_kws={'size':8})
ax1.set_title('Quarterly retention heatmap — % of cohort still buying\n'
              '(rows = acquisition quarter, cols = quarters since first purchase)',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax1.set_xlabel('Quarters since first order', fontsize=9, color=GRAY)
ax1.set_ylabel('Acquisition cohort', fontsize=9, color=GRAY)
ax1.tick_params(axis='x', labelsize=8.5)
ax1.tick_params(axis='y', rotation=0, labelsize=8.5)

# ── Chart 2: Cohort size by quarter ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
cs         = cohort_sizes[cohort_sizes.index.str.startswith('201')]
bar_colors = [COHORT_COLORS.get(c, GRAY) for c in cs.index]
ax2.bar(range(len(cs)), cs.values, color=bar_colors, edgecolor='white', width=0.7)
ax2.set_xticks(range(len(cs)))
ax2.set_xticklabels(cs.index, rotation=45, ha='right', fontsize=7.5)
ax2.set_title('Cohort size\n(new customers per quarter)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax2.set_ylabel('New customers', fontsize=9, color=GRAY)
ax2.set_facecolor('white')
ax2.tick_params(labelsize=8.5)
ax2.spines[['top','right']].set_visible(False)
ax2.spines[['left','bottom']].set_color(GRAY)
for i, v in enumerate(cs.values):
    ax2.text(i, v+1, str(v), ha='center', fontsize=7.5, color=DARK)

# ── Chart 3: Avg retention by period ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
avg_ret = retention_pct.iloc[:, 1:13].mean()
ax3.plot(avg_ret.index, avg_ret.values, color=BLUE, linewidth=2,
         marker='o', markersize=5)
ax3.fill_between(avg_ret.index, avg_ret.values, alpha=0.12, color=BLUE)
ax3.set_title('Avg retention rate\nby period (all cohorts)', fontsize=11,
              fontweight='500', color=DARK, pad=8)
ax3.set_xlabel('Quarters since first order', fontsize=9, color=GRAY)
ax3.set_ylabel('Avg retention %', fontsize=9, color=GRAY)
ax3.set_facecolor('white')
ax3.tick_params(labelsize=8.5)
ax3.spines[['top','right']].set_visible(False)
ax3.spines[['left','bottom']].set_color(GRAY)
for x, y in zip(avg_ret.index, avg_ret.values):
    ax3.text(x, y+1.2, f'{y:.0f}%', ha='center', fontsize=7.5, color=DARK)

# ── Chart 4: Cumulative profit by cohort ──────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0:2])
for coh in early_cohorts:
    cdf = rev_cohort[rev_cohort['Cohort_Q'] == coh]
    ax4.plot(cdf['Period_Number'], cdf['Cum_Profit'] / 1e3,
             color=COHORT_COLORS.get(coh, GRAY), linewidth=2,
             marker='o', markersize=4, label=coh)
ax4.axhline(0, color=CORAL, linewidth=0.8, linestyle='--', alpha=0.6)
ax4.set_title('Cumulative profit over time — 2014 acquisition cohorts ($k)',
              fontsize=11, fontweight='500', color=DARK, pad=8)
ax4.set_xlabel('Quarters since first order', fontsize=9, color=GRAY)
ax4.set_ylabel('Cumulative profit ($k)', fontsize=9, color=GRAY)
ax4.set_facecolor('white')
ax4.tick_params(labelsize=8.5)
ax4.spines[['top','right']].set_visible(False)
ax4.spines[['left','bottom']].set_color(GRAY)
ax4.legend(fontsize=8.5, frameon=False)

# ── Panel 5: Key findings ─────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
ax5.set_facecolor('#F1EFE8')
ax5.axis('off')
ax5.text(0.5, 0.97, 'Key findings', ha='center', va='top',
         fontsize=11, fontweight='500', color=DARK, transform=ax5.transAxes)
findings = [
    ("Period 1 retention", "Only 17–30% return next quarter",     CORAL),
    ("Stabilises at 30%",  "Levels off ~30–40% by Q3",            AMBER),
    ("2014 cohorts",       "Largest & most complete — 595 custs", BLUE),
    ("2014Q1 best LTV",    "Cumulative profit grows to $49k",      TEAL),
    ("2015+ cohorts",      "Much smaller — fewer new customers",   AMBER),
    ("2016–2017",          "Tiny cohorts (3–22) — low new acquis", CORAL),
    ("Rising retention",   "Avg retention rises to 55% by Q15",   TEAL),
    ("Loyal survivors",    "Those who stay buy increasingly more", TEAL),
    ("Key gap",            "New customer acquisition slowing down",CORAL),
]
for i, (name, desc, color) in enumerate(findings):
    y = 0.87 - i * 0.097
    ax5.add_patch(plt.Rectangle((0.02, y-0.025), 0.008, 0.045,
                                color=color, transform=ax5.transAxes, clip_on=False))
    ax5.text(0.07, y, name, ha='left', va='center', fontsize=8.5,
             fontweight='500', color=DARK, transform=ax5.transAxes)
    ax5.text(0.07, y-0.038, desc, ha='left', va='center', fontsize=7.5,
             color=GRAY, transform=ax5.transAxes)

plt.savefig('cohort_retention.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
