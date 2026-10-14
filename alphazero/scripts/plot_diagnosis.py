"""Render the README figure from measured results (optional matplotlib dependency)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=root/'alphazero/reports/diagnosis-20261013.json')
    parser.add_argument('--output',type=Path,default=root/'docs/assets/search-budget.png')
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    indexed={r['name']:r for r in data['matches']}
    rows=[indexed['confirm_last64'],indexed['confirm_last256'],data['search1000']['result']]
    assert all(r['games']==200 and r['seed']==40_000_023 for r in rows)
    labels=['64','256','1,000']
    bg='#f4f3ef';ink='#222b43';purple='#7664c4';green='#28846e'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink,
                         'axes.labelcolor':ink,'xtick.color':ink,'ytick.color':ink})
    fig,axes=plt.subplots(1,2,figsize=(12,7),gridspec_kw={'width_ratios':[1,1.2]})
    fig.patch.set_facecolor(bg)
    fig.subplots_adjust(left=.075,right=.965,bottom=.28,top=.70,wspace=.28)
    fig.text(.075,.92,'PENTOMINO / ALPHAZERO STUDIO',color=purple,size=11,weight='bold')
    fig.text(.075,.845,'More search. The second-player gap.',size=25,weight='bold')
    fig.text(.075,.785,'Measured results from one frozen network — no retraining between search budgets.',size=11)
    for ax in axes:
        ax.set_facecolor(bg)
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#d1d4dd')
        ax.set_axisbelow(True);ax.grid(axis='y',color='#dce0e4',linewidth=.7)
        ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100])
        ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
        ax.tick_params(axis='both',length=0,pad=8)
        ax.set_xlabel('MCTS simulations per move',labelpad=12)
    axes[0].set_title('Overall win rate',loc='left',pad=17,weight='bold',size=14)
    bars=axes[0].bar(labels,[100*r['win_rate'] for r in rows],color=['#beb5df','#9b8cd3',purple],width=.58)
    for b,r in zip(bars,rows):
        axes[0].text(b.get_x()+b.get_width()/2,b.get_height()+3,f'{100*r["win_rate"]:.1f}%',ha='center',weight='bold')
    axes[1].set_title('Win rate by playing order',loc='left',pad=17,weight='bold',size=14)
    for player,offset,color,label in [('0',-.19,purple,'First'),('1',.19,green,'Second')]:
        wins=[r['by_color'][player].get('1.0',0) for r in rows]
        bars=axes[1].bar([i+offset for i in range(3)],wins,width=.34,color=color,label=label)
        for b,w in zip(bars,wins):
            axes[1].text(b.get_x()+b.get_width()/2,w+3,f'{w}%',ha='center',size=10,weight='bold')
    axes[1].set_xticks(range(3),labels)
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,frameon=False)
    fig.text(.075,.135,'200 games per budget · 100 games per playing order · Uniform legal random opponent',size=10)
    fig.text(.075,.093,'Paired evaluation seeds. Exploratory comparisons; 99/100 is an observation, not a guaranteed win rate.',size=9,color='#626a7a')
    fig.text(.075,.055,'13 OCT 2026  /  Results and reproducible experiments in alphazero/reports/',size=9,color='#626a7a')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160,facecolor=bg)
    plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
