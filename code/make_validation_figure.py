# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})
# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"
# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

# Reuses the paired-forest grammar from the project's checked mechanism figure.
# All declared method outcomes appear: fine-tuning's larger panel-(a) estimate
# is shown as a numeric interval, avoiding a scale that hides replay differences.
from pathlib import Path
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from analyze_stratification import ROOT

LABEL={'finetune':'Fine-tuning','reservoir':'Reservoir','gss_adapted':'GSS','cgsm_dual':'Two-support','dualls_adapted':'Dual-LS'}
ALL=list(LABEL)

def estimate(f,method,metric):
    row=f[(f.method==method)&(f.metric==metric)]
    assert len(row)==1
    row=row.iloc[0]
    assert np.isfinite([row['mean'],row.ci_low,row.ci_high]).all()
    return row

def interval(ax,row,y,color,marker='o',scale=1):
    ax.errorbar(row['mean']*scale,y,xerr=np.array([[row['mean']-row.ci_low],[row.ci_high-row['mean']]])*scale,
                fmt=marker,color=color,markersize=4.2,capsize=2.4,lw=1.0,zorder=4)

def decorate(ax,label,title,methods):
    ax.set_yticks(range(len(methods)),[LABEL[m] for m in methods])
    ax.set_ylim(len(methods)-.45,-.65)
    ax.axvline(0,color='#BBBBBB',lw=.7,ls='--',zorder=0)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=5)
    ax.text(-.29,1.11,'('+label+')',transform=ax.transAxes,fontweight='bold',fontsize=8)
    ax.set_title(title,loc='left',pad=17,fontweight='bold')

def main():
    strata=pd.read_csv(ROOT/'results/stratification/summary.csv')
    road=pd.read_csv(ROOT/'results/interaction_summary/summary.csv')
    road_seeds=pd.read_csv(ROOT/'results/interaction_summary/per_seed.csv')
    app=pd.read_csv(ROOT/'results/application/summary.csv')
    inci=pd.read_csv(ROOT/'results/application_incident/summary.csv')
    fig,axs=plt.subplots(2,2,figsize=(182/25.4,97/25.4))
    fig.subplots_adjust(left=.15,right=.98,bottom=.12,top=.855,wspace=.64,hspace=.83)
    ax=axs[0,0]
    methods=ALL[1:]
    decorate(ax,'a','Beyond initial difficulty',methods)
    for i,m in enumerate(methods):
        interval(ax,estimate(strata,m,'cpa_minus_random'),i-.12,CATEGORICAL[0],'o')
        interval(ax,estimate(strata,m,'cpa_minus_matched'),i+.12,CATEGORICAL[1],'s')
    ax.set_ylim(4.75,-.7)
    ax.set_xlim(-.055,.37)
    ax.set_xlabel('CPA excess forgetting (m)')
    ax.legend(handles=[Line2D([],[],color=CATEGORICAL[0],marker='o',ls='',label='Run matched'),
        Line2D([],[],color=CATEGORICAL[1],marker='s',ls='',label='+ Initial-error decile')],
        loc='lower left',bbox_to_anchor=(-.03,1.0),fontsize=6.3,ncol=2,handletextpad=.3,columnspacing=.7,borderaxespad=0)
    fine=estimate(strata,'finetune','cpa_minus_matched')
    ax.text(0,4.00,'Fine-tuning, difficulty matched:\n'+f'{fine["mean"]:.3f} [{fine.ci_low:.3f}, {fine.ci_high:.3f}] m',fontsize=6.5,va='center')
    ax=axs[0,1]
    methods=ALL[:-1]
    decorate(ax,'b','Independent road stream',methods)
    for i,m in enumerate(methods):
        vals=road_seeds[road_seeds.method==m].gap.to_numpy()
        assert len(vals)==4
        ax.scatter(vals,i+np.linspace(-.14,.14,len(vals)),s=13,facecolors='none',edgecolors='#999999',lw=.6,zorder=2)
        interval(ax,estimate(road,m,'gap'),i,CATEGORICAL[4])
    ax.set_xlabel('CPA minus ordinary forgetting (m)')
    ax.text(0,1.025,'8 domains; 4 seeds; 3 s endpoint',transform=ax.transAxes,fontsize=6.5,color='#555555')
    for ax,summary,subdir,label,title in [(axs[1,0],app,'application','c','Future-approach alerts'),
                                        (axs[1,1],inci,'application_incident','d','New approaches only')]:
        decorate(ax,label,title,ALL)
        seeds=pd.read_csv(ROOT/f'results/{subdir}/per_seed.csv')
        for i,m in enumerate(ALL):
            v=seeds[seeds.method==m].delta_fnr.to_numpy()*100
            assert len(v)==8
            ax.scatter(v,i+np.linspace(-.14,.14,len(v)),s=9,facecolors='none',edgecolors='#BBBBBB',lw=.6,zorder=2)
            interval(ax,estimate(summary,m,'delta_fnr'),i,CATEGORICAL[0] if label=='c' else CATEGORICAL[2],scale=100)
        ax.set_xlabel('Missed-event change (percentage points)')
        ax.text(0,1.025,'2,979 events; 8 seeds' if label=='c' else '669 incident events; sensitivity',transform=ax.transAxes,fontsize=6.5,color='#555555')
    output=ROOT/'figures/fig2_validation_evidence'
    with mpl.rc_context({'savefig.bbox':None}):
        fig.savefig(output.with_suffix('.pdf'),bbox_inches=None,pad_inches=0,metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(output.with_suffix('.png'),bbox_inches=None,pad_inches=0,dpi=300)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for ax in fig.axes:
        # Matplotlib may create locator labels beyond the limits but does not draw them.
        xlo,xhi=sorted(ax.get_xlim())
        ylo,yhi=sorted(ax.get_ylim())
        xt=[t for p,t in zip(ax.get_xticks(),ax.get_xticklabels()) if xlo<=p<=xhi]
        yt=[t for p,t in zip(ax.get_yticks(),ax.get_yticklabels()) if ylo<=p<=yhi]
        for t in [*ax.texts,ax.xaxis.label,*yt,*xt,ax.title]:
            b=t.get_window_extent(renderer)
            if t.get_visible() and (b.x0<0 or b.x1>fig.bbox.width or b.y0<0 or b.y1>fig.bbox.height):
                outside.append(t.get_text())
    assert not outside,outside
    report={'all_five_lowaltitude_methods_shown':True,'all_four_road_methods_shown':True,
        'lowaltitude_seeds':8,'road_seeds':4,'width_mm':182,'height_mm':97,
        'outside_text':outside,'panel_d':'secondary incident-only sensitivity, not initial primary freeze',
        'source_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'results/stratification/summary.csv',ROOT/'results/interaction_summary/summary.csv',
                      ROOT/'results/application/summary.csv',ROOT/'results/application_incident/summary.csv']}}
    (ROOT/'qa/validation_figure.json').write_text(json.dumps(report,indent=2))
    plt.close(fig)
    print(output)

if __name__=='__main__':
    main()
