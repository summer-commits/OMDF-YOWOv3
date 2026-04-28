# OMDF-YOWOv3: Orthogonal Motion-Guided Decoupled Fusion for Fine-Grained Spatio-Temporal Action Detection in Sports Videos

> \\\*\\\*Paper status:\\\*\\\* under submission / under review.  
> \\\*\\\*Code status:\\\*\\\* core model components are provided; more checkpoints, scripts, and dataset tools will be updated progressively.

## Introduction

Fine-grained spatio-temporal action detection in figure skating is challenging because action stages are compact, pose deformation is severe, and inter-class differences are often reflected by subtle local configurations rather than large motion displacement.

OMDF-YOWOv3 addresses these issues by reconstructing the information flow of a two-stream real-time detector through three coordinated mechanisms:

1. **STPE: Spatially-Aware Texture and Pattern Enhancement**  
Refines key-frame appearance features by enhancing local structure, suppressing irrelevant background responses, and preserving fine-grained posture details.
2. **LTCC: Long-term Temporal Compactness Condensation**  
Compresses long-term 3D temporal responses into a compact motion prior that emphasizes stage transitions and short-term rhythm cues.
3. **DOCF: Decoupled Orthogonal Cross-modal Fusion**  
Removes appearance-aligned redundant motion components and injects complementary motion residuals into classification and localization branches with different intensities.

The overall design follows the principle:

> \\\*\\\*Make appearance accurate first, compact motion second, and inject complementary motion in a task-oriented manner.\\\*\\\*

\---

## Main Contributions

* We propose **STPE** to enhance fine-grained local structures in the 2D key-frame branch.
* We propose **LTCC** to condense long-term video responses into a compact and injectable motion prior.
* We propose **DOCF** to perform orthogonal residual decomposition and task-differentiated motion injection.
* We construct a self-annotated **AVA-style figure skating dataset** for fine-grained spatio-temporal action detection.

\---

## Framework Overview

The model follows the dual-branch structure of YOWOv3:

```text
Input video clip
      │
      ├── 2D branch: YOLOv8 + DarkFPN
      │        └── STPE: fine-grained appearance refinement
      │
      ├── 3D branch: I3D
      │        └── LTCC: compact motion condensation
      │
      └── DOCF: decoupled orthogonal cross-modal fusion
               ├── classification-specific feature path
               └── localization-specific feature path
```

The final detection head outputs action categories and bounding boxes at multiple feature scales.

\---

## Main Results

### UCF101-24

|Model|Type|mAP (%)|FLOPs (G)|Params (M)|
|-|-:|-:|-:|-:|
|YOWOv3 baseline|Real-time dual-branch|88.33|37.40|52.74|
|**OMDF-YOWOv3**|Real-time dual-branch|**91.18**|**38.06**|**59.26**|

### Self-constructed AVA-style Figure Skating Dataset

|Model|mAP@0.5 (%)|FLOPs (G)|Params (M)|
|-|-:|-:|-:|
|YOWOv3 baseline|22.39|37.45|53.77|
|**OMDF-YOWOv3**|**26.11**|**38.05**|**59.24**|



