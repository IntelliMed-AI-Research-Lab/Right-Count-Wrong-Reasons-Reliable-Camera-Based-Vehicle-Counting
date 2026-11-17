# Right Count, Wrong Reasons
![image alt](https://github.com/IntelliMed-AI-Research-Lab/Right-Count-Wrong-Reasons/blob/9992f37c557b51e88875398ab1b413d5e805db5c/Figure_3.png)
### Size–Occlusion-Adjusted Recall and Hidden Counting Errors in Camera-Based Vehicle Sensing

**Research implementation · Computer vision · Object detection · Vehicle counting · Reproducibility**

A vehicle detector can produce a plausible **total count** while missing real vehicles, adding false detections, or placing predictions in the wrong locations. This repository studies the gap between *aggregate count agreement* and *object-level detection reliability* using YOLOv8 detectors, VisDrone imagery, a separate highway dataset, and an illustrative video-tracking case study.

**Scope:** An empirical evaluation and reproducibility workflow. **This project does not claim a new detection architecture, training method, or evaluation metric.** GAME and standard detection/counting metrics are established tools used for diagnosis.

## Research questions

1. **Image count and recall:** How do sparse-versus-dense image-count recall differences change after standardizing for apparent vehicle size and annotated occlusion?
2. **Count-error cancellation:** When predicted totals appear accurate, how many false positives (FP) and false negatives (FN) are hidden, both within and across images?
3. **Spatial reliability:** Can established grid-based counting metrics reveal discrepancies that image-level mean absolute error misses?
4. **Operating-point sensitivity:** How do confidence thresholds, non-maximum suppression (NMS), and input resolution change detector outputs and illustrative video-derived measurements?

> **Interpretation:** The “dense” and “sparse” groups refer to **annotated vehicles per image**, not calibrated vehicles per square metre. Size–occlusion adjustment yields an observational comparison—not a causal estimate of crowding. Annotated occlusion may itself mediate crowding effects.

## Research workflow at a glance

```text
Roadside/highway images + VisDrone2019-DET
           │
           ▼
COCO-pretrained YOLOv8n/s/m ──► VisDrone fine-tuned YOLOv8n (seeds 0, 1, 2)
           │                         │
           └─────────────┬───────────┘
                         ▼
         Ignored-region filtering + one-to-one IoU matching
                         │
                ┌────────┴────────┐
                ▼                 ▼
     Size × occlusion        Per-image TP / FP / FN
     standardized recall         │
                │             ┌───┴──────────────┐
                ▼             ▼                  ▼
     Image-bootstrap CIs   Count bias / MAE   GAME(0–3)
                │             │                  │
                └─────────────┴──────────────────┘
                              ▼
                     Reproducibility reports

Separate highway detector ──► ByteTrack ──► single-clip sensitivity analysis
```

## Main findings reported in the accompanying manuscript

The values below are from the **reviewed manuscript analysis**. The uploaded notebook is a separate **post-review rerun workflow** and includes an executed run; its exported result tables must be checked before claiming that it independently reproduces every number here.

| Analysis | Reported result | Interpretation |
|---|---|---|
| VisDrone validation | **548 images; 17,040** annotated vehicles | Car, van, truck, and bus categories |
| COCO-pretrained YOLOv8n/s/m | **+20.3 to +29.4 pp** unadjusted sparse-minus-dense recall differences | Apparent associations depend strongly on object composition |
| Exploratory size–occlusion-standardized COCO comparison | **+6.4 to +16.7 pp** | Positive residual associations remain in these models |
| Original fine-tuned YOLOv8n seed 0 | **−1.83 pp**, 95% image-bootstrap CI **[−5.01, +1.45]** | No positive adjusted deficit detected under the original directional CI rule |
| Exploratory three-seed fine-tuned mean | **−2.24 pp**, 95% image-bootstrap CI **[−5.26, +0.90]** | Conditional on these three checkpoints; **equivalence is not established** |
| Fine-tuned seed 0, confidence threshold 0.25 | **13,552 TP; 3,488 FN; 2,914 FP** | **17,040** GT vehicles versus **16,466** predicted instances |
| Aggregate signed count bias | **−3.37%** | The net shortfall is **574** predictions despite **6,402 FP + FN errors** |
| Image-level and spatial error | **MAE = GAME(0) = 4.78; GAME(3) = 8.68** | The GAME(3)/GAME(0) ratio is **1.82×**, not a percentage of localization errors |
| Near-correct image counts | **76 of 208** images (**36.5%**) | These images have an absolute count error ≤ 1 **and** at least two FP and two FN |
| Illustrative highway video | **92.5%** detections/frame, **21.5%** speed-proxy, **3.3%** tracker-derived flow variation | Ranges across 12 settings relative to reference; **not validated traffic accuracy** |

These estimates should not be read as proof that higher traffic density improves detection. The analyses are conditioned on specific images, trained weights, annotations, and choices of stratification.

## Datasets and models

| Dataset | Configuration | Important caveat |
|---|---|---|
| **VisDrone2019-DET** | 6,471 original training images; 548 official validation images | Official validation was inspected earlier in development, although the revised checkpoints were selected using an internal split |
| **Top-view / roadside highway images** | 536 available training images and 90 evaluation images | Filename/perceptual-hash checks suggest possible shared source footage; source-video independence is not established |
| **Highway video** | One illustrative 601-frame, 30 fps clip | No manual ground truth for crossings, track identities, or physical speed |

**Detector configurations:**

- **COCO-pretrained:** YOLOv8n, YOLOv8s, and YOLOv8m. These are **sizes within one model family**, not independent architectures.
- **VisDrone fine-tuned:** YOLOv8n, seeds **0/1/2**, trained using an internal fit/validation split of the original training set; 35 epochs, input size 960 px.
- **Highway fine-tuned:** A **different YOLOv8n checkpoint**, trained for 100 epochs, input size 640 px. Do not conflate it with the VisDrone weights.
- **Additional exploratory checks:** A car-only training/evaluation control and a **pretrained** RT-DETR-L baseline. The latter is **not** a second-family fine-tuning replication.

The COCO models use their available vehicle labels; VisDrone fine-tuning merges car/van/truck/bus into one vehicle category. These class-space differences limit causal interpretations of pretrained-versus-fine-tuned comparisons.

**Data:** Original dataset archives and model weights are not included in this README or the supplied notebook attachment. Obtain datasets from their licensed sources and follow the notebook's preparation cells.

## Notebook and current execution record

The supplied implementation is a **Kaggle-oriented Jupyter notebook**. If uploading it to this repository with its current name:

[`last-updated-code-cvpr-part-1(3).ipynb`](last-updated-code-cvpr-part-1%283%29.ipynb)

It contains embedded source modules (written to disk during execution), rather than requiring a separately installed `rcwr` Python package:

| Component | Purpose |
|---|---|
| `rcwr_core.py` | Matching, counting, size–occlusion estimands, bootstrap helpers, internal unit tests |
| `rcwr_data.py` | Dataset preparation, annotations, training split, provenance checks |
| `rcwr_train.py` | YOLO fine-tuning jobs and checkpoints |
| `rcwr_analysis.py` | Standardized recall, paired comparisons, class/neighbor analyses and diagnostics |
| `rcwr_video.py` | ByteTrack operating-point runs and video outputs |
| `rcwr_report.py` | Figures, CSV summaries, and LaTeX tables generated from saved results |
| `rcwr_runner.py` | End-to-end orchestration, run status, recovery and exports |

### Recorded run status

The **execution outputs embedded in the uploaded notebook** report:

```json
{
  "primary_status": "COMPLETE",
  "missing_primary": [],
  "failures_count": 0,
  "analysis_errors_count": 0,
  "elapsed_hours": 6.5373,
  "additional_testdev_present": true,
  "all_requested_stages_complete": true
}
```

This is a record of the saved Kaggle execution, **not an independent rerun or external validation** of the experiment by this README's author. The separately generated full results ZIP and small evidence ZIP were **not included in the files used to prepare this README**. Check the actual exported CSVs and hashes before relying on their contents.

## Reproduce with Kaggle

1. Create a new Kaggle notebook and import the supplied `.ipynb` file.
2. In **Settings**, enable **GPU T4 ×2** and **Internet On**. The complete pipeline requires CUDA; it will not silently switch to a reduced CPU experiment.
3. Review the **frozen settings** cell *before* running any evaluation. In particular, confirm dataset locations and any optional source-video mappings.
4. Run the notebook in cell order, or use **Save Version → Save & Run All** to preserve the execution outputs.
5. When the pipeline finishes, inspect `RUN_STATUS.json`, `tables/coverage.csv`, and `logs/pipeline.log` before using any result.
6. Download the relevant archives from `/kaggle/working/` and retain the executed notebook with them.

The notebook installs/pins `ultralytics==8.3.230` and `statsmodels==0.14.5` and uses the Kaggle PyTorch/CUDA environment. Other dependencies include NumPy, pandas, OpenCV, SciPy, Matplotlib, Pillow, YAML and a LAP assignment package. The saved configuration specifies a **12-hour total budget**, including a training-stage allocation, but neither platform uptime nor successful completion is guaranteed on a fresh run.

### Key settings in the provided notebook

```text
Three VisDrone seeds:      0, 1, 2
VisDrone epochs:           35
Highway epochs:            100
Training/evaluation size:  960 px (primary VisDrone)
Primary confidence:        0.25
NMS IoU:                   0.70
Matching IoU:              >= 0.50
Image bootstrap draws:     2,000
Confidence sweep:          0.10, 0.25, 0.40, 0.50, 0.60, 0.70
Resolution checks:         640, 960, 1280, 1600 px
NMS checks:                0.5, 0.7, 0.9
Video confidence:          0.25, 0.40, 0.55, 0.70
Video input sizes:         480, 640, 960 px
```

The notebook exposes optional `resume_source`, `source_group_csv`, `manual_frame_csv`, and `manual_crossings` settings. These do not manufacture missing independent data: a complete source-video mapping is necessary for a source-disjoint claim, and manual ground truth must actually exist before video accuracy can be evaluated.

### Modes and output archives

- **`FULL` (default):** Train, run detection/analysis, generate plots and export result archives.
- **`REPORT_ONLY`:** Rebuild reports from *compatible existing results*; this is not a substitute for missing training or inference outputs.
- **Full results ZIP:** Intended to preserve code-derived results, logs, checkpoint records, and metadata (subject to generated archive contents).
- **Small scientific ZIP:** A convenience subset of tables/plots/diagnostics; **not a complete reproducibility archive**.
- **Evidence inventory ZIP:** An index of files included/omitted by the smaller archive; **not the omitted files themselves**.

Common generated paths under the experiment directory include:

```text
RUN_STATUS.json
requested_config.json
sources/            # generated Python modules
manifests/          # image, split, provenance and run records
training/           # training logs and checkpoints
images/             # image-level predictions/counts
tables/             # analysis CSVs, coverage and metrics
strata/             # size–occlusion details
diagnostics/        # sanity checks and sensitivity results
video/              # per-frame and per-operating-point outputs
figures/            # generated scientific figures
latex_tables/       # LaTeX table exports
logs/               # pipeline logs
```

**Repository size note:** Raw data downloads and checkpoint files can be large. Avoid committing Kaggle downloads, private source videos, temporary caches, and large checkpoints directly to Git. Use appropriate licensed storage or release assets if redistribution is permitted.

## Evaluation protocol and interpretation

**Primary descriptive comparison:** Sparse and dense image-count tertiles are compared after standardizing recall across four apparent-size bands and three annotated-occlusion classes. The manuscript reports **11 of 12 eligible strata** under the primary minimum-count rule; the excluded tiny/heavily occluded stratum removes **215** compared objects. Standardization therefore applies to the **retained** target population, not automatically to all vehicles.

**Uncertainty:** Bootstrap resampling uses **whole images**, with fixed pooled stratum weights, and conditions on observed model weights. The original fine-tuned seed-0 **directional CI rule** preceded its initial standardized comparison; this does **not** imply independent preregistration of every model or setting. Additional seeds, COCO-adjusted comparisons, equivalence margins, RT-DETR, resolution/NMS sweeps, and other robustness checks are exploratory/post-review.

**Counting:** Per image, signed count error satisfies `predicted − ground truth = FP − FN`. A small signed aggregate bias can coexist with large FP and FN totals. Within-image opposing errors and across-image cancellation should not be conflated.

**GAME:** Grid-based Average Mean Absolute Error is an **existing** metric. `GAME(0)` equals per-image counting MAE; progressively finer spatial grids can reveal greater absolute discrepancy. The ratio `GAME(3) / GAME(0)` is **not** the percentage of errors attributable to mislocalization.

**Video:** The separate highway detector is evaluated with **Ultralytics ByteTrack** on a single unvalidated clip. Detections/frame is a box-count proxy, flow is tracker-derived line crossings, and speed is **vehicle lengths per second** (not calibrated road speed). No independent manual crossing/track/physical-speed ground truth is claimed.

## Limits and reproducibility cautions

- **Not a causal study:** Image-count tertiles, apparent size, annotated occlusion, viewpoint, scene and capture sequence do not identify the causal effect of physical traffic density.
- **Not proof of equivalence:** A 95% CI spanning zero only means the specified positive deficit was not detected; it does not demonstrate that detector performance is equivalent between groups.
- **Not external replication:** Three seeds of a single detector family do not substitute for independent sites, model families, or source-disjoint test video.
- **Validation exposure:** Revised checkpoint selection uses an internal training split, but VisDrone official validation was examined during earlier development; test-dev exposure history may be uncertain.
- **Potential highway leakage:** Image provenance checks flag possible shared source footage; do not claim reliable unseen-video generalization.
- **Data/model differences:** COCO-pretrained and VisDrone-fine-tuned detectors have different class labels and training distributions.
- **Supplemental material:** A notebook, its saved outputs, and scientific figures are related but not interchangeable artifacts. Do not claim raw manifests, checkpoints, or scripts are included unless those files are actually released.

## Repository organization

If you publish only the uploaded notebook and this README, the repository can look like:

```text
Right-Count-Wrong-Reasons/
├── README.md
└── last-updated-code-cvpr-part-1(3).ipynb
```

The code modules and result folders shown earlier are generated **when the notebook is executed**; they are not automatically independent repository files. A future release can use a simpler notebook filename and a structured `results/` directory, with any actual evidence archives and figure assets explicitly listed.

## Related manuscript

**Beyond Raw Counts: Size–Occlusion-Adjusted Recall Associations and Counting Errors in Camera-Based Vehicle Sensing**

Publication information and a verified bibliographic citation can be added when available. This repository does not claim conference acceptance or imply that its current notebook outputs have been independently reproduced outside the saved run.

## Acknowledgments and license

This research uses the **Ultralytics YOLOv8** ecosystem, **VisDrone2019-DET**, a top-view highway vehicle dataset, and the **ByteTrack** tracking method. Credit and dataset/model licenses remain with their respective creators.

**AI assistance:** ChatGPT assisted with experiment-code development/debugging, analysis planning and documentation. Researchers remain responsible for running the software, verifying results, and checking scientific claims.

**License:** Check the repository's actual `LICENSE` file before advertising reuse rights. In the absence of a license, do not assume the code has an open-source license by default.
