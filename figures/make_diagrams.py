import math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family": "Lato", "font.size": 7.2})
W = 4.75
DARK, MID, LIGHT, PALE = "#1a1a1a", "#6b6b6b", "#c8c8c8", "#ededed"
OUT = "/home/claude/p5/fig"

def fig(h):
    f = plt.figure(figsize=(W, h)); ax = f.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100 * h / W); ax.axis("off")
    return f, ax

def box(ax, x, y, w, h, t, fill=PALE, bold=False, fs=7.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=DARK,
        lw=0.8, ls=ls, zorder=3))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", wrap=True)
    return (x, y, w, h)

def arr(ax, a, b, style="-|>", ls="-", rad=0.0, lw=0.9, col=DARK, sh=2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7,
        lw=lw, ls=ls, color=col, connectionstyle=f"arc3,rad={rad}",
        shrinkA=sh, shrinkB=sh, zorder=1))

def down(ax, b1, b2, **k):
    arr(ax, (b1[0], b1[1] - b1[3]/2), (b2[0], b2[1] + b2[3]/2), **k)

def chain(ax, labels, x, top, step, w, h, **k):
    bs = [box(ax, x, top - i*step, w, h, t, **k) for i, t in enumerate(labels)]
    for a, b in zip(bs, bs[1:]): down(ax, a, b)
    return bs

def save(f, n):
    f.savefig(f"{OUT}/F{n:02d}.png", dpi=300); plt.close(f)

# F1 lifecycle
f, ax = fig(3.9); cx, cy, r = 50, 41, 29
st = ["Define","Measure","Evaluate","Diagnose","Improve","Regression\nTest","Release","Monitor","Learn"]
part = ["I","II–III","II–III","II–III","IV","IV","IV","V","V"]
pts = []
for i, s in enumerate(st):
    a = math.pi/2 - 2*math.pi*i/9
    p = (cx + r*math.cos(a), cy + r*math.sin(a)*0.95); pts.append(p)
    box(ax, p[0], p[1], 17, 8.5, f"{s}\nPart {part[i]}", fill=[PALE,LIGHT,LIGHT,LIGHT,"#d9d9d9","#d9d9d9","#d9d9d9",PALE,PALE][i])
for i in range(9):
    arr(ax, pts[i], pts[(i+1) % 9], rad=-0.25, sh=17)
ax.text(cx, cy, "Evidence at\nevery stage", ha="center", va="center", fontsize=8, fontweight="bold")
save(f, 1)

# F2 quality evidence loop
f, ax = fig(6.9)
L = ["Quality Objectives","Evaluation Strategy","Evaluation Dataset","System Under Test"]
b = chain(ax, L, 42, 124, 11, 36, 6.5)
ev = [box(ax, 20+i*22, 80, 19, 6.5, t) for i, t in enumerate(["Code","LLM Judge","Human"])]
for e in ev: arr(ax, (42, b[-1][1]-3.3), (e[0], e[1]+3.3))
box(ax, 88, 80, 20, 9, "Evaluator\nValidation", fill=LIGHT, ls="--")
arr(ax, (86, 84.5), (64, 83.5), ls="--", rad=0.3); arr(ax, (64, 76.5), (86, 75.5), ls="--", rad=0.3)
R = ["Quality Evidence\n(with uncertainty)","Diagnosis (error analysis)","Improvement","Release Quality Gate","Production","Tracing & Online Evaluation","Feedback & Failure Triage"]
b2 = chain(ax, R, 42, 67, 9.4, 36, 6.5)
for e in ev: arr(ax, (e[0], e[1]-3.3), (42, b2[0][1]+3.3))
box(ax, 86, 30, 22, 7, "Regression Set", fill=LIGHT)
arr(ax, (60, b2[-1][1]), (86, 26.5), ls="--")
arr(ax, (86, 33.5), (60, b[2][1]), ls="--", rad=0.25)
box(ax, 10, 60, 17, 10, "Revised\nQuality\nObjectives", fill=LIGHT, fs=6.6)
arr(ax, (24, b2[-1][1]), (10, 55), ls="--", rad=-0.2)
arr(ax, (10, 65), (24, b[0][1]), ls="--", rad=-0.2)
save(f, 2)

# F3 Compass across projects
f, ax = fig(4.4)
S = [("Single-turn assistant","P1 Python LLM Evaluation Framework"),
     ("RAG over shipping policies","P2 RAG Evaluation Pipeline"),
     ("Agent: reroute, credits, claims","P3 AI Agent Evaluation System"),
     ("Released through CI","P4 AI Quality Gates in CI/CD"),
     ("Production with tracing and feedback","P5 Production AI Quality\nEngineering Platform")]
for i, (a, p) in enumerate(S):
    y = 84 - i*15
    b1 = box(ax, 27, y, 44, 10, f"Stage {i+1}\n{a}")
    b2 = box(ax, 75, y, 40, 10, p, fill=LIGHT)
    arr(ax, (49.5, y), (55, y))
    if i: arr(ax, (27, y + 10), (27, y + 5))
ax.add_patch(Rectangle((5, 3), 90, 7, fc=DARK, ec=DARK))
ax.text(50, 6.5, "aiqe core: every project extends one package", color="white", ha="center", va="center", fontsize=7.5, fontweight="bold")
save(f, 3)

# F4 dataset architecture
f, ax = fig(4.6)
src = ["Hand-written","Synthetic","Production-\nsampled","Adversarial","Observed\nfailures"]
for i, s in enumerate(src):
    b = box(ax, 10 + i*20, 88, 17, 8, s, fs=6.5); arr(ax, (b[0], 83.5), (50, 76))
cp = box(ax, 50, 72, 40, 7, "Candidate pool")
rv = box(ax, 50, 60, 40, 7, "Review & de-identify (gate)", fill=LIGHT, bold=True); down(ax, cp, rv)
ax.add_patch(Rectangle((12, 16), 76, 32, fc=PALE, ec=DARK, lw=0.8))
ax.text(50, 44, "Versioned evaluation dataset", ha="center", fontweight="bold")
for i, t in enumerate(["Golden","Regression","Adversarial","Production-\nsampled"]):
    box(ax, 22 + i*19, 30, 16, 12, t + "\nDev | Holdout", fill="white", fs=6.3)
arr(ax, (50, 56.5), (50, 48))
box(ax, 88, 58, 20, 9, "Dataset card +\ncontent hash", fill=LIGHT, fs=6.5)
box(ax, 50, 7, 40, 6, "Evaluation runs (case → objective IDs)", fs=6.5)
arr(ax, (50, 16), (50, 10))
save(f, 4)

# F5 error analysis workflow
f, ax = fig(4.8)
L5 = ["Traces (dev-split outputs; production samples)","Review: binary verdict + note on the first\nthing that went wrong (open coding)",
      "Group and name notes in discussion (axial coding)","Failure modes: definition, included and\nexcluded examples, severity",
      "Prioritise: count × severity weight"]
b = chain(ax, L5, 50, 94, 15.5, 70, 10)
outs = ["Objective confirmed\nor created (gaps)","Candidate\nevaluator chosen","Failing traces →\nregression cases"]
for i, t in enumerate(outs):
    o = box(ax, 18 + i*32, 12, 28, 10, t, fill=LIGHT, fs=6.5)
    arr(ax, (50, b[-1][1] - 5), (o[0], o[1] + 5))
arr(ax, (86, b[3][1]), (86, b[1][1]), ls="--", rad=0.4)
ax.text(92, b[2][1], "review\nmore\ntraces", ha="center", va="center", fontsize=6.0)
save(f, 5)

# F6 evaluator spectrum
f, ax = fig(3.2)
for j, (lab, l, r_) in enumerate([("Cost per judgement","low","high"),("Determinism","high","low"),("Scale","every case","samples")]):
    y = 61 - j*6
    ax.add_patch(Rectangle((30, y-1.2), 50, 2.4, fc=LIGHT, ec=DARK, lw=0.5))
    ax.text(29, y, f"{lab}: {l}", ha="right", va="center", fontsize=6.3)
    ax.text(81, y, r_, ha="left", va="center", fontsize=6.3)
for i, (t, fill) in enumerate([("Code evaluators", PALE), ("LLM judges", LIGHT), ("Human reviewers", "#a8a8a8")]):
    box(ax, 18 + i*32, 32, 27, 9, t, fill=fill, bold=True)
arr(ax, (4, 18), (96, 18)); ax.text(50, 13.5, "Semantic reach: narrow → broad", ha="center")
arr(ax, (31.5, 29), (36.5, 29), ls="--"); ax.text(34, 25.5, "defer", ha="center", fontsize=6)
arr(ax, (63.5, 29), (68.5, 29), ls="--"); ax.text(66, 25.5, "sample for validation", ha="center", fontsize=6)
arr(ax, (82, 36.5), (50, 36.5), ls="--", rad=0.35); ax.text(66, 44, "labels validate judges", ha="center", fontsize=6)
save(f, 6)

# F7 judge architecture
f, ax = fig(4.4)
for i, t in enumerate(["Evaluation case\n(input, reference)","System output","Rubric\n(id, version, criterion)"]):
    b = box(ax, 17 + i*33, 86, 28, 10, t, fs=6.5); arr(ax, (b[0], 81), (50, 74))
b = chain(ax, ["Judge prompt template (versioned file, SHA)","Provider: Replay (default) | Live (recording)",
               "Parse & validate (JSON schema)"], 50, 69, 15, 62, 8)
s1 = box(ax, 28, 16, 34, 9, "Score (pass/fail + reasoning)", fill=LIGHT); s2 = box(ax, 74, 16, 30, 9, "Deferred (unparseable)", ls="--")
arr(ax, (50, b[-1][1]-4), (s1[0], 20.5)); arr(ax, (50, b[-1][1]-4), (s2[0], 20.5))
box(ax, 50, 4.5, 90, 6, "Judge configuration record: judge id · model snapshot · template SHA · rubric id/version · criterion", fill=PALE, fs=6.1)
save(f, 7)

# F8 validation loop
f, ax = fig(4.2)
L8 = ["Judge (config, template,\nmodel snapshot)","Run on labelled outputs","Compare with adjudicated\nhuman labels","Error analysis of\ndisagreements","Improve judge on\nDEV labels","Re-validate on\nTEST labels"]
pts = []
for i, t in enumerate(L8):
    a = math.pi/2 - 2*math.pi*i/6; p = (50 + 30*math.cos(a), 45 + 30*math.sin(a)); pts.append(p)
    box(ax, p[0], p[1], 27, 10, t, fs=6.4)
for i in range(6): arr(ax, pts[i], pts[(i+1) % 6], rad=-0.2, sh=24)
ax.text(50, 45, "Status: validated\nor suspended\n→ judge record", ha="center", va="center", fontsize=6.6, fontweight="bold")
ax.text(50, 3, "Triggers: model snapshot · template · rubric · system change · deferral rise · schedule", ha="center", fontsize=6.2)
save(f, 8)

# F9 schematic agreement matrix (no measured values)
f, ax = fig(3.3)
ax.text(58, 64, "Judge verdict", ha="center", fontweight="bold")
ax.text(10, 38, "Human\nlabel", ha="center", va="center", fontweight="bold")
cells = [("TP\nboth pass", 40, 46), ("FN\nhuman pass,\njudge fail", 76, 46), ("FP\nhuman fail,\njudge pass", 40, 22), ("TN\nboth fail", 76, 22)]
ax.text(40, 58.5, "pass", ha="center"); ax.text(76, 58.5, "fail", ha="center")
ax.text(22, 46, "pass", ha="center", va="center"); ax.text(22, 22, "fail", ha="center", va="center")
for t, x, y in cells:
    ax.add_patch(Rectangle((x-17, y-11), 34, 22, fc=PALE, ec=DARK, lw=0.8)); ax.text(x, y, t, ha="center", va="center", fontsize=6.8)
ax.text(58, 5.5, "TPR = TP / (TP + FN)     TNR = TN / (TN + FP)     deferred counted separately", ha="center", fontsize=6.4)
save(f, 9)

# F10 schematic variance chart (no data)
f, ax2 = plt.subplots(figsize=(W, 2.9))
ax2.set_xlim(0, 6); ax2.set_ylim(0, 1); ax2.set_xticks(range(1, 6)); ax2.set_xticklabels([f"trial {i}" for i in range(5)])
ax2.set_yticks([]); ax2.set_ylabel("pass rate (per objective)")
ax2.axhspan(0.45, 0.62, color=LIGHT); ax2.text(3, 0.535, "95% case-bootstrap interval", ha="center", va="center", fontsize=7)
ax2.text(3, 0.85, "One point per recorded trial: produced by the Project 1\nrepeated-trial run (fixtures trial-0 … trial-4)", ha="center", va="center", fontsize=7)
ax2.text(3, 0.2, "Strip below: number of flaky cases (verdicts differ across trials)", ha="center", fontsize=6.6)
for s in ["top", "right"]: ax2.spines[s].set_visible(False)
f.tight_layout(); save(f, 10)

# F11 RAG evaluation architecture
f, ax = fig(4.8)
cs = box(ax, 16, 85, 28, 12, "RAG case\n(question, relevant\nIDs, criteria)", fs=6.4)
rt = box(ax, 50, 85, 22, 9, "Retriever"); gn = box(ax, 84, 85, 22, 9, "Generator")
arr(ax, (30.5, 85), (38.5, 85)); arr(ax, (61.5, 85), (72.5, 85))
ax.text(67, 90.5, "trace: query, ranked\npassages, answer", ha="center", fontsize=5.8)
lanes = ["Retrieval metrics\nrecall@k · precision@k\nMRR · nDCG","Generation checks\ngroundedness judge\ncitation validity","End-to-end objectives\nclaim-deadline check\ncredit cascade"]
for i, t in enumerate(lanes):
    l = box(ax, 17 + i*33, 55, 30, 16, t, fill=LIGHT, fs=6.3); arr(ax, (l[0], 47), (50, 32))
box(ax, 50, 24, 70, 13, "Attribution: OK · Retrieval · Generation · Ungrounded", fill=PALE, bold=True, fs=6.8)
arr(ax, (50, 80.5), (17, 63.5), ls="--"); arr(ax, (84, 80.5), (50, 63.5), ls="--"); arr(ax, (84, 80.5), (83, 63.5), ls="--")
save(f, 11)

# F12 outcome vs trajectory
f, ax = fig(3.6)
steps = ["Instruction","Model\nstep 1","Tool\ncall A","Model\nstep 2","Tool\ncall B","Final\nmessage"]
for i, t in enumerate(steps):
    b = box(ax, 8 + i*16.8, 50, 13, 10, t, fs=6.2)
    if i: arr(ax, (8 + (i-1)*16.8 + 6.9, 50), (8 + i*16.8 - 6.9, 50))
ax.add_patch(Rectangle((2, 30), 20, 7, fc=LIGHT, ec=DARK, lw=0.6)); ax.text(12, 33.5, "state before", ha="center", va="center", fontsize=6.3)
ax.add_patch(Rectangle((78, 30), 20, 7, fc=LIGHT, ec=DARK, lw=0.6)); ax.text(88, 33.5, "state after", ha="center", va="center", fontsize=6.3)
ax.plot([12, 12, 88, 88], [27, 22, 22, 27], color=DARK, lw=0.8)
ax.text(50, 14, "OUTCOME (Ch 13): required checks · permitted changes · side effects\nself-report vs reality · reliability over trials", ha="center", va="center", fontsize=6.3)
ax.plot([25, 25, 75, 75], [57, 62, 62, 57], color=DARK, lw=0.8)
ax.text(50, 68, "TRAJECTORY (Ch 14): tool selection · arguments · order · efficiency · recovery", ha="center", fontsize=6.3)
save(f, 12)

# F13 safety pipeline
f, ax = fig(5.0)
for i, t in enumerate(["Threat\nmodel","Known\npatterns","Red-team\nsessions","Incidents","Automated\ngenerators"]):
    b = box(ax, 10 + i*20, 96, 17, 8, t, fs=6.2); arr(ax, (b[0], 91.5), (50, 86))
pool = box(ax, 50, 82, 50, 6.5, "Adversarial candidate pool → review")
a1 = box(ax, 30, 70, 34, 7, "Adversarial set (private)", fill=LIGHT); a2 = box(ax, 72, 70, 34, 7, "Benign sensitive set", fill=LIGHT)
arr(ax, (50, 78.5), (30, 74)); arr(ax, (50, 78.5), (72, 74))
cfg = ["Guardrail alone","System without guardrail","System with guardrail"]
for i, t in enumerate(cfg):
    c = box(ax, 18 + i*32, 55, 29, 7, t, fs=6.4); arr(ax, (50, 66), (c[0], 59))
    arr(ax, (c[0], 51.5), (50, 44))
box(ax, 50, 39, 86, 8, "Evaluators: canary check · trajectory report · sandbox diff · validated refusal judge", fs=6.3)
box(ax, 50, 24, 80, 9, "Refusal matrix per configuration and category:\nunsafe rate + over-refusal rate, with intervals", fill=PALE, bold=True, fs=6.4)
down(ax, (50, 39, 0, 8), (50, 24, 0, 9))
box(ax, 50, 7, 70, 7, "Successful attacks → adversarial regression set", fill=LIGHT, ls="--")
arr(ax, (50, 19.5), (50, 10.5), ls="--"); arr(ax, (86, 7), (90, 70), ls="--", rad=0.3)
save(f, 13)

# F14 change-impact map (Table 17 routing)
f, ax = fig(5.2)
ch = ["Prompt","System model","Provider SDK","Retriever","Corpus","Tools / descriptions","Sandbox","Judge model","Judge template","Rubric","Dataset"]
cons = ["Re-record outputs","Re-record verdicts","Re-validate judges","Critical / task suites","RAG suites","Agent suites","Safety suites","Reliability suite","Judge validation","Human labels","Smoke suite","Regression set"]
route = {"Prompt":[0,1,3,6],"System model":[0,1,2,3,4,5,6,7],"Provider SDK":[0,1,10],"Retriever":[0,1,2,3,4],"Corpus":[0,1,2,3,4],
         "Tools / descriptions":[0,1,5,6],"Sandbox":[0,1,5],"Judge model":[1,2,8],"Judge template":[1,2,8],"Rubric":[1,2,8,9],"Dataset":[3]}
ys = [104 - i*9 for i in range(11)]; yc = [104 - i*8.3 for i in range(12)]
for i, c in enumerate(ch): box(ax, 15, ys[i], 26, 6, c, fs=6.2)
for j, c in enumerate(cons): box(ax, 82, yc[j], 30, 5.8, c, fill=LIGHT if j < 3 else PALE, fs=6.1)
for i, c in enumerate(ch):
    for j in route[c] + [11]:
        ax.plot([28.5, 66.5], [ys[i], yc[j]], color=MID, lw=0.35)
ax.text(50, 3, "Every change routes to the regression set (Table 17)", ha="center", fontsize=6.4)
save(f, 14)

# F15 regression lifecycle
f, ax = fig(4.2)
L15 = ["Change\n(manifest)","Impact\nanalysis","Re-record\nas required","Run routed suites\n+ regression set","Paired comparison\nvs baseline","Decision","New baseline\n+ manifest"]
pts = []
for i, t in enumerate(L15):
    a = math.pi/2 - 2*math.pi*i/7; p = (50 + 31*math.cos(a), 46 + 31*math.sin(a)); pts.append(p)
    box(ax, p[0], p[1], 24, 10, t, fs=6.3)
for i in range(7): arr(ax, pts[i], pts[(i+1) % 7], rad=-0.2, sh=22)
ax.text(50, 46, "Triaged failures feed the\nregression set; regressed\ncases feed error analysis", ha="center", va="center", fontsize=6.4)
save(f, 15)

# F16 schematic Pareto (no data)
f, ax2 = plt.subplots(figsize=(W, 3.0))
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_xlabel("cost per task (from recorded tokens and a dated price table)"); ax2.set_ylabel("groundedness pass rate")
ax2.step([0.1, 0.35, 0.6, 0.9], [0.3, 0.55, 0.75, 0.9], where="post", color=DARK, lw=1)
ax2.text(0.62, 0.45, "Pareto frontier drawn from the\nProject 4 migration candidates;\ndominated candidates in grey;\nmarker shape shows the\nOBJ-REL-01 pass^3 limit", fontsize=6.8, va="center")
for s in ["top", "right"]: ax2.spines[s].set_visible(False)
f.tight_layout(); save(f, 16)

# F17 CI gate
f, ax = fig(5.0)
box(ax, 50, 99, 88, 7, "Nightly: live canary (budgeted) → compare with recordings → alert", fill=LIGHT, ls="--", fs=6.3)
L17 = ["Change (code, prompt, configuration)","manifest: fingerprint components","plan: impact analysis (Ch 17)",
       "Routed suites + regression set on REPLAY\n(no model calls, no secrets)","Evidence builder: interval bounds,\ncounts, paired comparisons",
       "gate: pull-request policy (release policy\nat release) → overrides (review only)","PASS · PASS-WITH-OVERRIDE · REVIEW · FAIL\n→ exit code → branch protection"]
b = chain(ax, L17, 44, 87, 12.3, 66, 8.5)
box(ax, 90, 30, 18, 12, "Report\non the PR\n+ artifacts", fill=PALE, fs=6.2); arr(ax, (77, b[-1][1]), (83, 26))
save(f, 17)

# F18 production evaluation
f, ax = fig(4.6)
top = ["Live\nconversation","Compass\n(retriever,\nmodel, tools)","Guardrails\n(response path)","Customer"]
for i, t in enumerate(top):
    box(ax, 12 + i*25.3, 86, 20, 13, t, fs=6.3)
    if i: arr(ax, (12 + (i-1)*25.3 + 10.5, 86), (12 + i*25.3 - 10.5, 86))
box(ax, 37, 66, 60, 8, "TracedProvider spans (gen_ai.*) + compass.* attributes", fill=LIGHT, fs=6.4)
arr(ax, (37, 79), (37, 70.5)); box(ax, 84, 66, 26, 8, "OTLP → trace\nbackend", fs=6.3); arr(ax, (67.5, 66), (70.5, 66))
bot = ["Sampler: risk tags always;\nhash-based base rate","Redaction","Online evaluators: invariants, pre-filters,\nvalidated judges, trajectory checks, humans",
       "OnlineScore store (kept separate from offline)","Quality signals (Table 22) → triage (Ch 21)"]
chain(ax, bot, 45, 54, 11, 76, 7.5, fs=6.3)
arr(ax, (37, 62), (45, 57.8))
save(f, 18)

# F19 production-to-regression loop
f, ax = fig(4.4)
L19 = ["Production\ntraffic","Signals","Triage queue\nnew → confirmed |\ndismissed","Promote: de-identify,\nreference, PII check","Regression set\n(dev, fm: tags)","CI gate on\nevery change","Release"]
pts = []
for i, t in enumerate(L19):
    a = math.pi/2 - 2*math.pi*i/7; p = (50 + 31*math.cos(a), 50 + 31*math.sin(a)); pts.append(p)
    box(ax, p[0], p[1], 25, 11, t, fs=6.2)
for i in range(7): arr(ax, pts[i], pts[(i+1) % 7], rad=-0.2, sh=22)
ax.text(50, 50, "Side branches:\ntaxonomy grows (Ch 4)\nno objective → revise plan (Ch 2)\ndrift → new golden cases", ha="center", va="center", fontsize=6.2)
save(f, 19)

# F20 platform layers
f, ax = fig(4.6)
ax.add_patch(Rectangle((3, 66), 94, 26, fc=PALE, ec=DARK, lw=0.8)); ax.text(50, 88, "Systems under test", ha="center", fontweight="bold")
box(ax, 26, 75, 38, 10, "Compass: plan · datasets ·\npolicy · sandbox", fs=6.4); box(ax, 72, 75, 38, 10, "Other Tidewater assistants\n(own plans and policies)", ls="--", fs=6.4)
ax.add_patch(Rectangle((3, 42), 94, 20, fc=LIGHT, ec=DARK, lw=0.8)); ax.text(50, 58, "Adapters (optional, isolated)", ha="center", fontweight="bold")
ax.text(50, 48, "providers · DeepEval · Inspect AI · Ragas (own environment) · OTLP backends", ha="center", fontsize=6.4)
ax.add_patch(Rectangle((3, 4), 94, 34, fc="#dcdcdc", ec=DARK, lw=0.8)); ax.text(50, 34, "aiqe core (vendor-neutral)", ha="center", fontweight="bold")
mods = ["strategy","datasets + registry","evaluators + judge records","runner + replay","stats","experiments","changes + manifest","policy + evidence + CI","tracing + sampling","online · drift · triage","status page"]
for i, m in enumerate(mods):
    ax.text(8 + (i % 4)*23, 27 - (i // 4)*8, m, fontsize=6.3)
save(f, 20)

# F21 maturity staircase
f, ax = fig(3.4)
lv = [("1 Ad hoc","demos, spot checks"),("2 Defined","plan, owners,\nversioned data"),("3 Measured","error analysis,\nvalidated judges,\nintervals"),("4 Gated","baselines, change\ndetection, CI gate"),("5 Continuous","tracing, online eval,\ndrift, triage,\nrelease records")]
for i, (t, c) in enumerate(lv):
    ax.add_patch(Rectangle((4 + i*19, 8), 18, 12 + i*12, fc=[PALE,"#e3e3e3",LIGHT,"#bcbcbc","#a8a8a8"][i], ec=DARK, lw=0.8))
    ax.text(13 + i*19, 22 + i*12, t, ha="center", fontweight="bold", fontsize=6.8)
    ax.text(13 + i*19, 14 + i*6, c, ha="center", va="center", fontsize=5.9)
ax.text(50, 3, "Levels are cumulative; new systems start again at level 2", ha="center", fontsize=6.4)
save(f, 21)
print("done")
