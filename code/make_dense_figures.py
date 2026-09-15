"""Compact, source-complete experimental panels for the ten-page RIFT paper.

Scientific-figure-making: native IEEE size, shared legends and vector export.
SciPilot: seed-level replication, paired uncertainty, ordered-time semantics.
Existing validated forest/scatter grammar is reused; no numerical outcomes change.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from analyze_stratification import ROOT
import make_mechanism_figure as mechanism

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,'xtick.labelsize':6.3,
    'ytick.labelsize':6.3,'legend.fontsize':6.5,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.6,'xtick.direction':'out',
    'ytick.direction':'out','legend.frameon':False,'pdf.fonttype':42,
    'svg.fonttype':'none','savefig.bbox':None})
OUT=ROOT/'figures'
METHODS=['finetune','reservoir','gss_adapted','cgsm_dual']
LABELS=['Fine-tuning','Reservoir','GSS','Two-support']
TASKS=['MA','FT','LN','ZS2','OF','EP0','GL','ZS0']
COLORS=['#0072B2','#D55E00']

def save(fig,stem,report):
    OUT.mkdir(exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text(): continue
        # Unused locator ticks may sit outside the axes but are never rendered.
        if text.axes is None and text not in fig.texts: continue
        box=text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            outside.append(text.get_text())
    report['outside_text']=outside
    assert not outside, outside
    fig.savefig(OUT/f'{stem}.png',dpi=300,bbox_inches=None)
    Image.open(OUT/f'{stem}.png').convert('L').save(OUT/f'{stem}_grayscale.png',dpi=(300,300))
    fig.savefig(OUT/f'{stem}.pdf',bbox_inches=None,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(OUT/f'{stem}.svg',bbox_inches=None)
    report.update(width_mm=182,height_mm=round(fig.get_figheight()*25.4,2),min_font_pt=6.0,
                  intervals='No significance stars or multiplicity-adjusted tests are introduced.')
    (ROOT/'qa'/f'{stem}.json').write_text(json.dumps(report,indent=2))
    plt.close(fig)

def road():
    path=ROOT/'results/interaction_summary/all_checkpoint_metrics.csv'
    frame=pd.read_csv(path)
    assert len(frame)==576 and not frame[['fde','risk_fde']].isna().any().any()
    assert not frame.duplicated(['method','seed','learn_step','eval_task_index']).any()
    assert set(frame.seed)=={7,11,23,47}
    diag=frame[frame.learn_step==frame.eval_task_index][['method','seed','eval_task_index','fde','risk_fde']]
    full=frame.merge(diag,on=['method','seed','eval_task_index'],suffixes=('','_initial'),validate='many_to_one')
    full['ordinary']=full.fde-full.fde_initial
    full['cpa']=full.risk_fde-full.risk_fde_initial
    history=full[full.learn_step>full.eval_task_index]
    curves=history.groupby(['method','seed','learn_step'])[['ordinary','cpa']].mean().reset_index()
    assert len(curves)==112
    curves.to_csv(ROOT/'source_data/road_dynamics_seed_curves.csv',index=False)
    final=history[history.learn_step==7]
    cells=final.groupby(['method','eval_task_index','eval_task'])[['ordinary','cpa']].mean().reset_index()
    cells.to_csv(ROOT/'source_data/road_dynamics_final_cells.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(182/25.4,98/25.4))
    fig.subplots_adjust(left=.082,right=.982,bottom=.185,top=.89,wspace=.47,hspace=.64)
    positions=[(0,0),(0,1),(0,2),(1,0)]
    for mi,(method,label,(rr,cc)) in enumerate(zip(METHODS,LABELS,positions)):
        ax=axs[rr,cc]
        f=curves[curves.method==method]
        for metric,color,marker,ls in zip(['ordinary','cpa'],COLORS,['o','s'],['-','--']):
            for seed,part in f.groupby('seed'):
                ax.plot(part.learn_step+1,part[metric],color=color,alpha=.22,lw=.5,ls=ls)
            mean=f.groupby('learn_step')[metric].mean()
            ax.plot(mean.index+1,mean.values,color=color,marker=marker,ms=2.8,lw=1.2,ls=ls)
        ax.axhline(0,color='#AAAAAA',lw=.55,ls=':')
        ax.set_xlim(1.7,8.3)
        ax.set_xticks(range(2,9))
        ax.set_ylim((-.25,3.1) if mi==0 else (-.85,.95))
        ax.set_yticks([0,1,2,3] if mi==0 else [-.5,0,.5])
        ax.set_title(f'({chr(97+mi)}) {label}',loc='left',fontweight='bold',pad=5)
        ax.set_xlabel('Update step',labelpad=2)
        ax.set_ylabel('Historical forgetting (m)',labelpad=3)
    matrices=[]
    for ci,metric in enumerate(['ordinary','cpa']):
        ax=axs[1,ci+1]
        vals=cells.pivot(index='method',columns='eval_task_index',values=metric).loc[METHODS,range(7)].to_numpy()
        matrices.append(vals)
        im=ax.imshow(vals,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-4.1,vcenter=0,vmax=4.1),aspect='auto',interpolation='nearest')
        for y in range(4):
            for x in range(7):
                ax.text(x,y,f'{vals[y,x]:.2f}',ha='center',va='center',fontsize=6,
                        color='white' if abs(vals[y,x])>2.6 else '#222222')
        ax.set_xticks(range(7),TASKS[:7],fontsize=6)
        ax.set_yticks(range(4),['FT','Res.','GSS','Two'],fontsize=6.3)
        ax.tick_params(length=0)
        ax.set_xlabel('Historical domain',labelpad=3)
        ax.set_title(f'({chr(101+ci)}) Final '+('ordinary' if ci==0 else 'CPA')+' forgetting',loc='left',fontweight='bold',pad=5)
    cbax=fig.add_axes([.468,.072,.43,.023])
    cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[-4,-2,0,2,4])
    cb.set_label('Forgetting (m); negative = improvement',fontsize=6.5,labelpad=1)
    cb.ax.tick_params(labelsize=6,length=2,pad=1)
    fig.legend(handles=[Line2D([],[],color=COLORS[0],marker='o',ms=3,lw=1,label='Ordinary'),
                        Line2D([],[],color=COLORS[1],marker='s',ms=3,lw=1,ls='--',label='CPA stratum'),
                        Line2D([],[],color='#888888',lw=.6,label='Thin lines: four seeds')],
               loc='upper center',bbox_to_anchor=(.52,1.0),ncol=3,columnspacing=1.7)
    final_seed=final.groupby(['method','seed'])[['ordinary','cpa']].mean()
    reference=pd.read_csv(ROOT/'results/interaction_summary/per_seed.csv').set_index(['method','seed'])
    for a,b in [('ordinary','ordinary_forgetting'),('cpa','cpa_forgetting')]:
        np.testing.assert_allclose(final_seed[a].sort_index(),reference[b].sort_index(),atol=1e-12)
    save(fig,'fig3_road_dynamics',{'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_rows':576,'diagonal_rows_used_as_baselines':128,'historical_rows':448,
        'seed_curves':112,'heatmap_cells':56,'seed_count':4,
        'center':'Equal historical-domain mean within seed, then mean across four seeds.',
        'uncertainty':'Every seed shown as a thin curve; heatmap cells are descriptive four-seed means.',
        'time_axis':'Step 2 through 8; the historical domain set expands at each update.',
        'scale':'Fine-tuning has a separate vertical scale; all three replay panels share limits. Heatmaps share one symmetric zero-centered scale.',
        'cpa_counts':frame.groupby('eval_task').cpa_n.first().to_dict(),
        'interpretation':'CPA and ordinary retention vary with update and domain. LN has only six CPA windows; ZS0 is current at the final step, so no final historical forgetting is defined for it.'})

def mechanism_dense():
    cells,cover,forest,stats=mechanism.prepare_data()
    fig,axs=plt.subplots(1,4,figsize=(182/25.4,61/25.4))
    fig.subplots_adjust(left=.058,right=.99,bottom=.275,top=.72,wspace=.62)
    ax=axs[0]
    vals=cells.pivot(index='gradient_quintile',columns='risk_quintile',values='observed_expected_ratio').to_numpy()
    counts=cells.pivot(index='gradient_quintile',columns='risk_quintile',values='n').to_numpy()
    im=ax.imshow(vals,origin='lower',cmap='RdBu_r',vmin=.5,vmax=1.5,aspect='equal')
    for y in range(5):
        for x in range(5): ax.text(x,y,str(int(counts[y,x])),ha='center',va='center',fontsize=6,color='white' if abs(vals[y,x]-1)>.34 else '#222222')
    ax.set_xticks(range(5),range(1,6)); ax.set_yticks(range(5),range(1,6))
    ax.set_xlabel('CPA-score quintile',labelpad=2); ax.set_ylabel('Gradient quintile',labelpad=2)
    box=ax.get_position()
    cbax=fig.add_axes([box.x0,.095,box.width,.027])
    cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[.5,1,1.5]);cb.ax.tick_params(labelsize=6,length=2,pad=1)
    cb.set_label('Observed / expected',fontsize=6,labelpad=1)
    for i,(xc,yc,matched) in enumerate([('weighted_cover_mean','forgetting_risk_fde',False),('cover_residual','forgetting_residual_m',True)],start=1):
        ax=axs[i]
        # Show every run, without drawing lines between categorical methods.
        for row in cover.itertuples():
            ax.scatter(getattr(row,xc),getattr(row,yc),s=12,marker=mechanism.METHOD_MARKER[row.method],
                       color=mechanism.BUDGET_COLOR[int(row.condition)],edgecolors='white',linewidths=.25)
        mechanism.regression_line(ax,cover[xc],cover[yc],COLORS[1] if matched else '#333333')
        ax.axhline(0,color='#BBBBBB',lw=.5,ls=':')
        if matched: ax.axvline(0,color='#BBBBBB',lw=.5,ls=':')
        ax.set_xlabel('Cover residual' if matched else 'Cover distance',labelpad=2)
        ax.set_ylabel('Forgetting residual (m)' if matched else 'CPA forgetting (m)',labelpad=2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax=axs[3]
    paired=forest[forest.row_type=='paired_seed']; summary=forest[forest.row_type=='bootstrap_mean_95ci']
    for y,m in enumerate(mechanism.SELECTOR_ORDER):
        p=paired[paired.method==m].sort_values('seed');s=summary[summary.method==m].iloc[0]
        ax.scatter(p.method_minus_gss_risk_fde_m,y+np.linspace(-.13,.13,8),s=9,facecolors='none',edgecolors='#777777',lw=.5)
        mean=s.method_minus_gss_risk_fde_m
        ax.errorbar(mean,y,xerr=[[mean-s.ci95_low],[s.ci95_high-mean]],fmt='D',color=COLORS[0],ms=3,capsize=2,lw=.8)
    ax.set_yticks(range(4),['CPA top','CPA state','CPA grad.','Two-support']);ax.set_ylim(3.5,-.5)
    ax.axvline(0,color='#AAAAAA',lw=.6,ls=':');ax.set_xlabel('Selector - GSS\nCPA-FDE (m)',labelpad=2)
    ax.set_xticks([-.5,0,.5]);ax.set_xlim(-.56,.88)
    titles=['(a) CPA / gradient','(b) Aggregate cover','(c) Matched cover','(d) Selector effects']
    subtitles=[rf'$n=4,000;\ \rho={stats["rho_rank"]:.3f}$',rf'$\rho={stats["rho_forgetting_raw"]:+.3f}$',rf'$\rho={stats["rho_forgetting_matched"]:+.3f}$','8 paired seeds; 95% CI']
    for ax,title,subtitle in zip(axs,titles,subtitles):
        box=ax.get_position()
        fig.text(box.x0,.925,title,fontsize=7.2,fontweight='bold',va='top')
        fig.text(box.x0,.845,subtitle,fontsize=6.5,va='top')
    handles=[Line2D([],[],marker=m,ls='',color='#444444',ms=3,label=l) for m,l in [('o','GSS'),('s','H2C'),('^','Reservoir')]]
    handles += [Line2D([],[],marker='o',ls='',color=c,ms=3,label=str(b)) for b,c in mechanism.BUDGET_COLOR.items()]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.58,.015),ncol=7,fontsize=6,handletextpad=.25,columnspacing=.7)
    save(fig,'fig4_mechanism_dense',{'statistics':stats,'source_files':[str(p.relative_to(ROOT)) for p in [mechanism.RANK_SOURCE,mechanism.COVER_SOURCE,mechanism.FOREST_SOURCE]],
        'panels':'4 separately labeled panels; all 4,000 windows, 48 coverage runs and eight paired selector seeds retained.',
        'legend':'Coverage marker encodes method; blue intensity encodes memory budget.',
        'interpretation':'Associations are descriptive; selector intervals are paired seed bootstrap intervals.'})

if __name__=='__main__':
    road()
    mechanism_dense()
