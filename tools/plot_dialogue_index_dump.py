"""Export the index-dump comparison using actual saved grades and usage."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ORDER = ('full_context_batch', 'closing_6k', 'index_neighborhoods', 'index_hierarchy',
         'index_atomic', 'hierarchy_plus_neighborhoods')
LABELS = ('Full raw control · Astra*', 'Neighborhood evidence · Sol*',
          'Neighborhood index only', 'Entire hierarchy index only',
          'Atomic index only', 'Hierarchy + neighborhood evidence')
COLORS = ('#475569', '#94a3b8', '#60a5fa', '#3b82f6', '#1d4ed8', '#059669')


def plot(root):
    report = json.loads((root/'report.json').read_text(encoding='utf-8'))
    totals = report['totals']
    folder = root/'charts'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.titleweight':'bold', 'savefig.facecolor':'white'})
    y = np.arange(len(ORDER))
    scores = [totals[a]['checkpoint_score_pct'] for a in ORDER]
    inputs = [totals[a]['prompt_tokens']/1000 for a in ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), gridspec_kw={'width_ratios':[1.1, 1]})
    fig.subplots_adjust(left=.255, right=.965, bottom=.24, top=.79, wspace=.18)
    fig.suptitle('Summary index dumps vs. the saved full-context control',
                 x=.04, y=.955, ha='left', fontsize=19, fontweight='bold')
    fig.text(.04, .89, 'One engineering memory · 775,741 source tokens · 15 matched questions · 70 frozen checkpoints',
             ha='left', fontsize=11, color='#475569')
    ax = axes[0]
    ax.barh(y, scores, color=COLORS, height=.65, zorder=3)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0,20,40,60,80,100], [f'{v}%' for v in (0,20,40,60,80,100)])
    ax.set_title('Checkpoint coverage — higher is better', loc='left', fontsize=11, pad=16)
    ax.grid(axis='x', alpha=.16, zorder=0)
    for i, arm in enumerate(ORDER):
        ax.text(scores[i]+1.3, i, f"{scores[i]:.1f}%", va='center', fontsize=10, fontweight='bold',
                zorder=5, bbox=dict(facecolor='white', edgecolor='none', pad=.4))
    old = totals['full_context_batch']['previous_score_pct']
    ax.axvline(old, color='#b45309', linestyle='--', linewidth=1.4, zorder=4)
    ax.text(.02, -.15, f'Dashed line: previous control grade, {old:.1f}%', transform=ax.transAxes,
            fontsize=9, color='#92400e')
    ax = axes[1]
    ax.barh(y, inputs, color=COLORS, height=.65, zorder=3)
    ax.set_yticks(y, ['']*len(y))
    ax.invert_yaxis()
    ax.set_xlim(0, max(inputs)*1.24)
    ax.set_title('Total reader input — lower is better', loc='left', fontsize=11, pad=16)
    ax.set_xlabel('Thousands of input tokens across all 15 questions')
    ax.grid(axis='x', alpha=.16, zorder=0)
    for i, value in enumerate(inputs):
        ax.text(value+max(inputs)*.012, i, f'{value:,.1f}K', va='center', fontsize=10, fontweight='bold')
    fig.text(.04, .11,
        '* Saved answers reused; all six arms jointly regraded blind. Control: Astra / medium reasoning, one shared-context call.\n'
        'Other arms: Sol / no reasoning, one call per question. Model and call structure differ; this is not an isolated retrieval-loss estimate.\n'
        'Coverage is a selective reference score, not general accuracy. Index citations indicate summaries, not verified raw support.',
        fontsize=9, linespacing=1.5, color='#475569', va='center')
    fig.text(.04, .035, 'Display-label limit: index turn ranges were 1-based; raw T labels were 0-based. Summary IDs and source bindings were verified.',
             fontsize=8, color='#64748b')
    for suffix in ('png','svg','pdf'):
        fig.savefig(folder/f'comparison.{suffix}', dpi=180)
    plt.close(fig)

    rows = report['rows']
    matrix = np.asarray([[100*r['points'][a]/r['possible'] for r in rows] for a in ORDER])
    fig, ax = plt.subplots(figsize=(14.5, 5.2))
    fig.subplots_adjust(left=.255, right=.94, bottom=.19, top=.79)
    fig.suptitle('Coverage by question', x=.04, y=.96, ha='left', fontsize=18, fontweight='bold')
    fig.text(.04, .865, 'Each cell is the jointly graded percentage of that question’s available checkpoint points.',
             color='#475569', fontsize=10)
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(np.arange(len(rows)), [r['id'] for r in rows])
    ax.set_yticks(y, LABELS)
    for i in range(len(ORDER)):
        for j in range(len(rows)):
            ax.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center',
                    color='white' if matrix[i,j] >= 60 else '#0f172a', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.025, pad=.025, label='Checkpoint coverage (%)')
    fig.text(.04, .075, 'Q01–Q15 retain the original frozen wording. Full questions and per-checkpoint grades are in the saved report and grades.',
             fontsize=9, color='#475569')
    for suffix in ('png','svg','pdf'):
        fig.savefig(folder/f'per-question.{suffix}', dpi=180)
    plt.close(fig)
    with (folder/'comparison.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['arm','checkpoint_points','possible_points','checkpoint_coverage_pct',
                         'total_reader_input_tokens','reader_calls','reader_model'])
        for arm in ORDER:
            t = totals[arm]
            writer.writerow([arm,t['points'],t['possible'],t['checkpoint_score_pct'],
                             t['prompt_tokens'],t['model_calls'],t['reader_model']])
    print(json.dumps({'chart':str(folder/'comparison.png'), 'heatmap':str(folder/'per-question.png')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('eval_results/dialogue-index-dump-20261006-r2'))
    plot(parser.parse_args().root)
