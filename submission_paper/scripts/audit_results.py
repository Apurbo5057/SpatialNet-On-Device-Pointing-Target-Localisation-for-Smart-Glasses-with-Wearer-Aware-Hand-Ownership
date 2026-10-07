"""Recompute the selected manuscript numbers and vector figures from saved outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
NAMES = ["main_run","seed_1","seed_2","img224","img320","A1_no_copypaste","A2_no_geometry_loss","A3_no_roi_geometry","A4_no_anchor_chain"]
data, audit = {}, {}
for name in NAMES:
    folders = list((ROOT/"runs"/name).glob("*/evaluation"))
    assert len(folders)==1, (name,folders)
    ev=folders[0]
    t=pd.read_csv(ev/"predictions/test.csv").set_index("name")
    mh=pd.read_csv(ev/"predictions/test_multihand.csv").set_index("name")
    pos=t[t.gt_T==1]
    assert len(t)==3029 and len(pos)==2604
    data[name]=(pos,mh)
    a={"n_all":len(t),"n_target":len(pos),"p30":100*(pos.target_iou>=.3).mean(),
       "p50":100*(pos.target_iou>=.5).mean(),"p70":100*(pos.target_iou>=.7).mean(),
       "miou":100*pos.target_iou.mean(),"hand_p50":100*(pos.hand_iou>=.5).mean(),
       "direction_mean_deg":pos.ray_err_deg.mean(),
       "stress_p50":100*(mh[mh.gt_T==1].target_iou>=.5).mean(),
       "stress_nohand_rejection":100*(mh[mh.gt_H==0].pred_scene=="No hand").mean(),
       "actually_pasted_positives":int(((mh.gt_T==1)&(mh.n_distractors>0)).sum())}
    a["groups"]={g:{"n":len(q),"p50":100*(q.target_iou>=.5).mean(),"miou":100*q.target_iou.mean()} for g,q in pos.groupby("gap_bin")}
    comp=ROOT/"runs"/name/"comparison_vs_main_run.json"
    if comp.exists(): a["paired_comparisons"]=json.loads(comp.read_text())["rows"]
    audit[name]=a
for k in ("p50","p70","miou"):
    vals=[audit[n][k] for n in ("main_run","seed_1","seed_2")]
    audit.setdefault("three_seed_summary",{})[k]={"mean":float(np.mean(vals)),"sample_sd":float(np.std(vals,ddof=1))}
audit["notes"]=["Localisation denominator is every GT-target-positive frame, independent of gate predictions.",
 "Paired intervals are inherited from saved comparison_vs_main_run.json; test-image uncertainty only.",
 "Training seeds 1/2 retain evaluation seed 42. Stress-test condition includes positives without successful paste.",
 "Device observations exist in original manuscript; no standalone measurement log was found.",
 "Published VistaRef Base Hybrid values were checked against Literature review/Vista-Ref.pdf Table 1."]
(BASE/"evidence.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
colors=["#24678D","#278377","#AC682C"]
fig,axes=plt.subplots(1,2,figsize=(6.8,2.55),layout="constrained")
xs=[.231199232,.452650496,.923234432]
for key,label,marker in [("p50","P@0.5","o"),("p70","P@0.7","s")]:
    ys=[audit[n][key] for n in ["main_run","img224","img320"]]
    axes[0].plot(xs,ys,marker=marker,label=label,lw=1.6)
    for x,y,n in zip(xs,ys,[160,224,320]): axes[0].annotate(str(n),(x,y),xytext=(0,6),textcoords="offset points",ha="center",fontsize=7)
axes[0].set(xlabel="Counted computation (GFLOPs)",ylabel="Target localisation (%)",ylim=(25,72),title="Same 722K parameters")
axes[0].legend(frameon=False,loc="lower right",fontsize=7)
groups=["on-object","near","far"]
xx=np.arange(3)
for i,(n,c) in enumerate(zip(["main_run","img224","img320"],colors)):
    ys=[audit[n]["groups"][g]["p50"] for g in groups]
    axes[1].bar(xx+(i-1)*.23,ys,width=.22,color=c,label=["160 px","224 px","320 px"][i],edgecolor="white")
axes[1].set(xticks=xx,xticklabels=["Overlap\nn=1,796","Near\nn=599","Large gap\nn=209"],ylabel="Target P@0.5 (%)",ylim=(0,83),title="Image-plane separation")
axes[1].legend(frameon=False,fontsize=7,ncol=3,loc="upper center",handlelength=.8,columnspacing=.6)
for ax in axes: ax.grid(axis="y",alpha=.18); ax.set_axisbelow(True)
fig.savefig(BASE/"figures/resolution.pdf")
fig.savefig(BASE/"qa/resolution.png",dpi=170)
plt.close(fig)
# One balanced effect display, including the result contrary to the original ray hypothesis.
effects=[("A1_no_copypaste","Target P@0.5 with pasted hands","A1: stress-test P@0.5"),
 ("A1_no_copypaste","No-hand kept with pasted hands (%)","A1: no-hand rejection"),
 ("A4_no_anchor_chain","Target P@0.7","A4: target P@0.7"),
 ("A3_no_roi_geometry","Target P@0.7","A3: target P@0.7"),
 ("A2_no_geometry_loss","Target P@0.5","A2: target P@0.5")]
fig,ax=plt.subplots(figsize=(6.8,2.2),layout="constrained")
for i,(n,k,l) in enumerate(effects):
    r=next(r for r in audit[n]["paired_comparisons"] if r["metric"]==k)
    v,lo,hi=-r["diff"],-r["ci_high"],-r["ci_low"]
    ax.errorbar(v,len(effects)-i-1,xerr=[[v-lo],[hi-v]],fmt="o",color="#24678D" if v>0 else "#AC682C",capsize=3)
ax.axvline(0,color=".5",lw=.8)
ax.set(yticks=np.arange(len(effects)),yticklabels=[x[2] for x in effects][::-1],
 xlabel="Full model minus ablation (percentage points; 95% paired interval)")
ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
fig.savefig(BASE/"figures/ablations.pdf")
fig.savefig(BASE/"qa/ablations.png",dpi=170)
plt.close(fig)
# Architecture has a separate authoring script; no model execution is needed.
import runpy
runpy.run_path(str(BASE/"scripts/make_architecture.py"), run_name="__main__")
print("Verified 9 runs. Written evidence.json and three vector figures.")
print(json.dumps(audit["three_seed_summary"],indent=2))
for name in ["img224","img320","A1_no_copypaste","A2_no_geometry_loss","A3_no_roi_geometry","A4_no_anchor_chain"]:
    print(name,[(r["metric"],round(r["diff"],2),round(r["ci_low"],2),round(r["ci_high"],2)) for r in audit[name]["paired_comparisons"] if r["metric"] in ["Target P@0.5","Target P@0.7","Direction error, far (deg)","No-hand kept with pasted hands (%)"]])
