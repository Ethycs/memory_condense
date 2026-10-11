"""Chart the five-question extension separately from the original 15 scores."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_dialogue_index_dump import ORDER, COLORS

LABELS = ('Full raw control · Astra', 'Neighborhood evidence · Sol',
          'Neighborhood index only', 'Entire hierarchy index only',
          'Atomic index only', 'Hierarchy + neighborhood evidence')


def plot(root):
    report = json.loads((root/'report.json').read_text(encoding='utf-8'))
    prior = json.loads((root/'previous-report.json').read_text(encoding='utf-8'))
    folder = root/'charts'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig, axes = plt.subplots(1,3,figsize=(15,6.6),sharey=True)
    fig.subplots_adjust(left=.25,right=.965,bottom=.24,top=.78,wspace=.16)
    fig.suptitle('Five more questions for every approach',x=.035,y=.95,ha='left',fontsize=20,fontweight='bold')
    fig.text(.035,.885,'Same 775,741-token archive · 6 approaches · 24 new checkpoint occurrences · 30 new answers',
             fontsize=11,color='#475569')
    groups = (prior['totals'],report['totals'],report['aggregate_totals'])
    titles = ('Original 15 questions','New 5 questions','Combined 20 questions*')
    for ax, totals, title in zip(axes,groups,titles,strict=True):
        scores = [totals[a]['checkpoint_score_pct'] for a in ORDER]
        ax.barh(np.arange(6),scores,color=COLORS,height=.63,zorder=3)
        ax.set_xlim(0,118)
        ax.set_xticks([0,25,50,75,100],[f'{x}%' for x in (0,25,50,75,100)])
        ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=15)
        ax.grid(axis='x',alpha=.15,zorder=0)
        ax.axvline(scores[0],color='#475569',linestyle=':',alpha=.6)
        for i,score in enumerate(scores):
            ax.text(score+1.5,i,f'{score:.1f}%',va='center',fontweight='bold',fontsize=10,
                    zorder=4,bbox=dict(facecolor='white',edgecolor='none',pad=.3))
    axes[0].set_yticks(np.arange(6),LABELS)
    axes[0].invert_yaxis()
    fig.text(.035,.135,
        'Checkpoint coverage, not general QA accuracy. New questions reuse the development archive and reference facts.\n'
        'Astra / medium answers questions jointly; Sol / no reasoning answers individually. Scores include model and call-structure differences.\n'
        '* Original grades remain unchanged. The new five use corrected zero-based index ranges; the original 15 had a display offset.',
        fontsize=9,color='#475569',linespacing=1.6,va='center')
    fig.text(.035,.035,'Each round is scored jointly with shuffled answer labels. Aggregate percentages are weighted by available checkpoint points.',
             fontsize=9,color='#64748b')
    for suffix in ('png','svg','pdf'):
        fig.savefig(folder/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    with (folder/'comparison.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['arm','original_points','original_possible','original_pct',
            'new_points','new_possible','new_pct','combined_points','combined_possible','combined_pct','new_input_tokens'])
        for a in ORDER:
            row = [a]
            for group in groups:
                row.extend(group[a][k] for k in ('points','possible','checkpoint_score_pct'))
            writer.writerow(row+[report['totals'][a]['prompt_tokens']])
    print(json.dumps(dict(chart=str(folder/'comparison.png'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('eval_results/dialogue-index-extension-20261006-r2'))
    plot(parser.parse_args().root)
