"""Six-panel time-to-entry decomposition, native IEEE width.

All five methods and all five disjoint bands are shown. Thin points are eight
training seeds; intervals are paired-seed bootstrap intervals, not window CIs.
Run once for preview, then with --export after visual review for vector masters.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from analyze_stratification import ROOT,METHODS
from analyze_ttc import BANDS

NAMES=['Fine-tuning','Reservoir','GSS','Two-support','Dual-LS']
LABELS=['Ongoing','New 0-3 s','New 3-5 s','New 5-10 s','No entry by 10 s']
COLORS=['#0072B2','#D55E00','#E69F00','#009E73','#C9C9C9']
HATCHES=['','///','...','xxx','']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,'xtick.labelsize':6.5,
    'ytick.labelsize':6.5,'legend.fontsize':6.5,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.6,'xtick.direction':'out',
    'ytick.direction':'out','legend.frameon':False,'pdf.fonttype':42,
    'svg.fonttype':'none','savefig.bbox':None})


def main(export=False):
    folder=ROOT/'results/ttc_control_20260916'
    summaries=pd.read_csv(folder/'summary.csv');seeds=pd.read_csv(folder/'per_seed.csv')
    support=pd.read_csv(folder/'support.csv')
    assert len(seeds)==320 and all(seeds.groupby(['method','stratum_name']).size()==8)
    fig,axs=plt.subplots(2,3,figsize=(182/25.4,116/25.4))
    fig.subplots_adjust(left=.124,right=.988,bottom=.19,top=.855,wspace=.88,hspace=.78)
    ax=axs.flat[0];tasks=['baseline','wind5','speed15','alt180','speed10']
    counts=support.pivot(index='task',columns='stratum',values='stratum_n').loc[tasks,list(BANDS)].to_numpy()
    assert np.all(counts.sum(axis=1)==2000)
    left=np.zeros(5)
    for j in range(5):
        ax.barh(np.arange(5),counts[:,j],left=left,height=.61,color=COLORS[j],
                edgecolor='white',linewidth=.35,hatch=HATCHES[j])
        left+=counts[:,j]
    ax.set_yticks(range(5),['Nominal','Wind 5','Speed 15','Alt. 180','Speed 10'])
    ax.invert_yaxis();ax.set_xlim(0,2000);ax.set_xticks([0,1000,2000])
    ax.set_xlabel('Frozen test windows',labelpad=3)
    ax.set_title('(a) Fixed support',loc='left',fontweight='bold',pad=8)
    jitter=np.linspace(-.19,.19,8)
    for panel,(method,label) in enumerate(zip(METHODS,NAMES),start=1):
        ax=axs.flat[panel]
        for y,band in enumerate(BANDS):
            f=seeds[(seeds.method==method)&(seeds.stratum_name==band)].sort_values('seed')
            row=summaries[(summaries.method==method)&(summaries.stratum_name==band)&
                          (summaries.metric=='stratum_minus_matched')].iloc[0]
            vals=f.stratum_minus_matched.to_numpy()
            ax.scatter(vals,y+jitter,s=10,marker='o',facecolor='white',edgecolor='#7D7D7D',lw=.45,zorder=2)
            ax.errorbar(row['mean'],y,xerr=[[row['mean']-row.ci_low],[row.ci_high-row['mean']]],
                        fmt='D',ms=3.3,color=COLORS[y] if y!=4 else '#555555',lw=1.15,capsize=2,zorder=3)
        ax.axvline(0,color='#8A8A8A',ls='--',lw=.65,zorder=1)
        ax.set_yticks(range(5),LABELS);ax.set_ylim(4.5,-.5)
        ax.set_xlim((-3.,6.5) if method=='finetune' else (-.8,1.12))
        ax.set_xticks([-2,0,2,4,6] if method=='finetune' else [-.5,0,.5,1])
        ax.set_xlabel('Matched excess\nforgetting (m)',labelpad=3)
        ax.set_title(f'({chr(97+panel)}) {label}',loc='left',fontweight='bold',pad=8)
    fig.legend([Patch(facecolor=c,edgecolor='#555555',linewidth=.35,hatch=h) for c,h in zip(COLORS,HATCHES)],
               LABELS,ncol=5,loc='upper center',bbox_to_anchor=(.5,.99),columnspacing=1.5,handlelength=1.5)
    fig.text(.5,.045,'Open circles: eight seeds     Diamonds and bars: mean and 95% seed-bootstrap interval',
             ha='center',fontsize=6.5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];overlaps=[]
    rendered_text=list(fig.texts)
    for ax in axs.flat:
        rendered_text.extend([ax.xaxis.label,ax.yaxis.label,ax.title,ax._left_title])
        rendered_text.extend(ax.get_xticklabels()+ax.get_yticklabels()+list(ax.texts))
    for legend in fig.legends:rendered_text.extend(legend.get_texts())
    for text in rendered_text:
        if not text.get_visible() or not text.get_text():continue
        box=text.get_window_extent(renderer)
        if box.x0<-1 or box.y0<-1 or box.x1>fig.bbox.width+1 or box.y1>fig.bbox.height+1:outside.append(text.get_text())
    for ai,ax in enumerate(axs.flat):
        for labels in [ax.get_xticklabels(),ax.get_yticklabels()]:
            boxes=[t.get_window_extent(renderer) for t in labels if t.get_visible() and t.get_text()]
            for a,b in zip(boxes[:-1],boxes[1:]):
                if a.overlaps(b):overlaps.append(ai)
    assert not outside and not overlaps,(outside,overlaps)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    stem='figS5_ttc_control'
    fig.savefig(out/f'{stem}.png',dpi=300,bbox_inches=None)
    Image.open(out/f'{stem}.png').convert('L').save(out/f'{stem}_grayscale.png',dpi=(300,300))
    if export:
        fig.savefig(out/f'{stem}.pdf',bbox_inches=None,metadata={'CreationDate':None,'ModDate':None})
        svg=out/f'{stem}.svg'
        fig.savefig(svg,bbox_inches=None)
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    report={'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'summary.csv',folder/'per_seed.csv',folder/'support.csv']},
            'size_mm':[182,116],'min_font_pt':6.5,'source_per_seed_rows':320,'seed_dots_shown':200,
            'disjoint_band_count':5,'matched_contrasts_shown':25,'support_cells':25,
            'outside_text':outside,'overlapping_ticks':overlaps,'vector_exported':export,
            'inference':'Intervals condition on fixed data/order; no multiplicity correction or significance stars.',
            'scale':'Fine-tuning has a separate horizontal scale; four replay panels share the same limits.',
            'profile_interpretation':'Seed is a categorical replicate ID. CPA-conditional NA values outside the defined four bands are structural, not missing observations. No outcome is imputed.',
            'selection':'All methods and disjoint time bands shown. Cumulative 3/5/10-second effects are tabulated separately.'}
    (ROOT/'qa').mkdir(exist_ok=True)
    (ROOT/'qa'/f'{stem}.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.close(fig);print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    main(parser.parse_args().export)
