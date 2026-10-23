"""Render audited fixed-cut engineering results as a standalone figure."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    source = ROOT / 'artifacts/gate1/20261023_upstream_frontier_full_audit_v1/recomputed_summary.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    assert data['status'] == 'PASS_SAVED_UPSTREAM_FRONTIER_ARRAYS_ONLY' and data['images_verified'] == 109
    target = OUT / 'frontier_comparison.png'
    assert not target.exists()
    config = OUT / 'mpl_config'; config.mkdir(exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(config)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    cuts = [-1,4,6,10,22]
    labels = ['Input\n(cut -1)', 'After block 4', 'After block 6', 'Backbone end\n(after block 10)', 'Neck end\n(after block 22)']
    series = []
    for prefix,suffix,label,color in [('fp32','fp16','FP32 prefix / FP16 suffix','#2477b3'),
                                      ('fp16','fp32','FP16 prefix / FP32 suffix','#d97732')]:
        rows = [data['comparisons'][f'reference_fp32_vs_cut_{cut}_source_{prefix}_suffix_{suffix}'] for cut in cuts]
        series.append((label,color,[row['boxes']['violations'] for row in rows],[row['fail_images'] for row in rows]))
    x = np.arange(5)
    fig,axes = plt.subplots(1,2,figsize=(12.4,4.9),dpi=160,layout='constrained')
    for position,(label,color,violations,images) in enumerate(series):
        shift = -.19 if position == 0 else .19
        for ax,values in zip(axes,(violations,images)):
            bars=ax.bar(x+shift,values,.34,label=label,color=color,zorder=3)
            for bar,value in zip(bars,values):
                ax.annotate(f'{value:,}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords='offset points',
                            ha='center',va='bottom',fontsize=8,color='#202733')
    axes[0].set_yscale('symlog',linthresh=1)
    axes[0].set_ylabel('Violating box elements (symlog; zero retained)')
    axes[0].set_title('Fixed all-anchor box tolerance')
    axes[1].set_ylabel('Images with any fixed-guard violation / 109')
    axes[1].set_ylim(0,124); axes[1].set_title('Image-level diagnostic failures')
    for ax in axes:
        ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.22,zorder=0)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='y',labelsize=8)
    handles,legends=axes[0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='outside lower center',ncols=2,frameon=False,fontsize=9)
    fig.suptitle('109 reused debug images | every intervention decoded in FP32',fontsize=12,fontweight='bold')
    axes[0].text(0,-.29,'Cuts are separate interventions; counts are not additive causal contributions.\nInput and block-22 endpoints include previously observed controls.',
                 transform=axes[0].transAxes,fontsize=7.5,color='#4a5260')
    fig.savefig(target,bbox_inches='tight')
    fig.savefig(OUT / 'frontier_comparison.svg',bbox_inches='tight',metadata={'Date':None})
    plt.close(fig)
    result={'source':str(source),'source_sha256':sha(source),'plot_source_sha256':sha(Path(__file__)),
            'figure_sha256':sha(target),'matplotlib':matplotlib.__version__,'cuts':cuts,
            'series':[{'label':label,'box_violations':boxes,'failed_images':images} for label,_,boxes,images in series],
            'scientific_gate_pass':False,'original_guard_changed':False}
    (OUT / 'plot_provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figure':str(target),'figure_sha256':sha(target)}))


if __name__ == '__main__':
    main()
