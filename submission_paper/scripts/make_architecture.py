from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BASE=Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42})
# Compact architecture: appearance fan-out, conditioned progression and training-only cue.
fig,ax=plt.subplots(figsize=(6.8,2.4))
ax.set(xlim=(0,10),ylim=(0,3.55)); ax.axis("off")
def box(x,y,w,h,title,detail,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=.07",facecolor=fc,edgecolor="#507080",lw=.7))
    ax.text(x+w/2,y+h*.68,title,ha="center",va="center",fontsize=8.5,weight="bold")
    ax.text(x+w/2,y+h*.28,detail,ha="center",va="center",fontsize=7.3)
def arrow(a,b,**kw):
    ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=.8,color="#405765",**kw))
box(.1,2.65,1.05,.65,"RGB","160 x 160","#f3f4f5")
box(1.5,2.65,8.25,.65,"Shared visual map S","ShuffleNetV2 + bidirectional fusion + coordinate channels + local mixing","#e6eef3")
arrow((1.17,2.98),(1.48,2.98))
items=[(.15,"Wearer hand","heatmap + local size"),(2.65,"Fingertip","hand-relative heatmap"),(5.15,"Direction","hand + tip descriptors"),(7.65,"Target","visual + ray logits")]
ax.plot([1.2,8.7],[2.38,2.38],color="#405765",lw=.8)
arrow((5.7,2.62),(5.7,2.38))
for x,title,detail in items:
    box(x,.95,2.1,.95,title,detail,"#edf4f1")
    arrow((x+1.05,2.38),(x+1.05,1.92))
for a,b in [(2.27,2.62),(4.77,5.12),(7.27,7.62)]: arrow((a,1.42),(b,1.42))
ax.text(2.49,2.1,"box / offsets",ha="center",fontsize=7)
ax.text(7.48,2.1,"ray coordinates",ha="center",fontsize=7)
ax.text(5,.53,"Continuous peak decoding; local hand/target size regression; conditional confidence gates",ha="center",fontsize=8)
ax.text(5,.12,"Training only: pasted hands retain wearer labels and receive negative heatmap weighting",ha="center",fontsize=7.5,color="#8c5127")
fig.subplots_adjust(left=.005,right=.995,bottom=.025,top=.99)
fig.savefig(BASE/"figures/architecture.pdf")
fig.savefig(BASE/"qa/architecture.png",dpi=180)
plt.close(fig)
