"""Draw publication examples from saved test predictions, without model execution."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent
sys.path.insert(0,str(ROOT))
from annotation import parse_annotation
CSV=ROOT/"runs/main_run/outputs_spatialnet/evaluation/predictions/test.csv"
rows=pd.read_csv(CSV).set_index("name")
success=[("img_0002020","(a) Overlap: banana"),("img_0011652","(b) Near: truck"),
 ("img_0011638","(c) Large gap: traffic light"),("img_0016703","(d) Small target: key")]
failure=[("img_0014779","(a) Repeated tableware"),("img_0015166","(b) Dense bookshelf"),
 ("img_0015705","(c) Cluttered garden"),("img_0005618","(d) Layered harbour scene")]
colors={"hand":"#1874d1","target":"#00a36c","tip":"#ef8139"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42})
def iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    a0=a[:2]-a[2:]/2;a1=a[:2]+a[2:]/2;b0=b[:2]-b[2:]/2;b1=b[:2]+b[2:]/2
    inter=np.maximum(0,np.minimum(a1,b1)-np.maximum(a0,b0)).prod()
    return float(inter/(a[2:].prod()+b[2:].prod()-inter+1e-8))
records=[]
for kind,choices in [("teaser",success),("qualitative",failure)]:
    fig,axs=plt.subplots(1,4,figsize=(7,2.4))
    for ax,(name,title) in zip(axs,choices):
        row=rows.loc[name]
        target=row[["pred_target_cx","pred_target_cy","pred_target_w","pred_target_h"]].to_numpy(dtype=float)
        hand=row[["pred_hand_cx","pred_hand_cy","pred_hand_w","pred_hand_h"]].to_numpy(dtype=float)
        tip=row[["pred_tip_x","pred_tip_y"]].to_numpy(dtype=float)
        ann=parse_annotation(ROOT/"EgoPoint_spatialnet/test/labels"/(name+".txt"))
        checked=iou(target,ann["roi"])
        assert abs(checked-row.target_iou)<3e-5,(name,checked,row.target_iou)
        assert row.pred_scene=="Pointing at target"
        assert (row.target_iou>.8 if kind=="teaser" else row.target_iou<.25)
        original=ImageOps.exif_transpose(Image.open(ROOT/"EgoPoint_dataset/dataset/img/test"/(name+".jpg"))).convert("RGB").resize((640,640),Image.Resampling.LANCZOS)
        ax.imshow(original,extent=(0,1,1,0))
        def box(b,c,ls="-",lw=1.5):
            ax.add_patch(Rectangle((b[0]-b[2]/2,b[1]-b[3]/2),b[2],b[3],fill=False,ec=c,lw=lw,ls=ls))
        # Draw the annotation last so it remains visible for high-IoU predictions.
        box(hand,colors["hand"],lw=1.15);box(target,colors["target"],lw=1.6)
        box(ann["roi"],"white","--",1.05)
        ax.plot(*tip,"o",color=colors["tip"],mec="white",mew=.6,ms=3.8)
        ax.set(xlim=(0,1),ylim=(1,0));ax.axis("off")
        ax.set_title(title,fontsize=7.6,pad=3)
        ax.text(.97,.96,f'IoU {row.target_iou:.2f}',transform=ax.transAxes,ha="right",va="top",fontsize=7,color="white",bbox=dict(facecolor="black",alpha=.65,edgecolor="none",pad=2))
        records.append({"figure":kind,"image":name,"title":title,"split":"test","gap_bin":row.gap_bin,"target_iou_saved":float(row.target_iou),"target_iou_checked":checked,
          "hand_iou":float(row.hand_iou),"saved_target_box":target.tolist(),"ground_truth_target":ann["roi"],
          "selection":"Deliberately selected successful example" if kind=="teaser" else "Deliberately selected complex failure; not all failures occur in complex scenes."})
    handles=[Patch(fill=False,ec=colors["hand"],label="Predicted wearer hand"),
             plt.Line2D([],[],marker="o",ls="",color=colors["tip"],label="Predicted fingertip",ms=4),
             Patch(fill=False,ec=colors["target"],label="Predicted target"),
             Patch(fill=False,ec=".45",ls="--",label="Annotated target")]
    fig.legend(handles=handles,loc="lower center",ncol=4,frameon=False,fontsize=7,handlelength=1.2,columnspacing=1)
    fig.subplots_adjust(left=.006,right=.994,top=.90,bottom=.14,wspace=.035)
    fig.savefig(BASE/"figures"/(kind+".pdf"),dpi=300)
    fig.savefig(BASE/"qa"/(kind+".png"),dpi=220)
    plt.close(fig)
(BASE/"example_provenance.json").write_text(json.dumps({"source":str(CSV.relative_to(ROOT)),"checkpoint":"runs/main_run/outputs_spatialnet/best_model.pth","input_size":160,"notes":"Boxes and fingertip are saved evaluation predictions. Original-resolution source images are shown in the same normalized square coordinate system. Direction omitted for clarity. No model execution by this script.","examples":records},indent=2),encoding="utf-8")
print([(r["figure"],r["image"],round(r["target_iou_checked"],3)) for r in records])
