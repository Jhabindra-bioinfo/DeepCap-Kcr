# DeepCap-Kcr

**DeepCap-Kcr: Accurate Identification and Investigation of Protein Lysine Crotonylation Sites Based on Capsule Network**

DeepCap-Kcr is a deep learning framework for predicting **protein lysine crotonylation (Kcr) sites** using a capsule neural network architecture.
The model integrates convolutional feature extraction, sequential learning, and dynamic capsule routing to automatically learn discriminative sequence patterns associated with lysine crotonylation.

This repository provides **Python/Jupyter notebooks, datasets, pretrained model weights, prediction scripts, visualization tools, and interpretability analyses** used in the published study.


## Overview

Lysine crotonylation (**Kcr**) is an important post-translational modification involved in multiple biological processes.

DeepCap-Kcr integrates:

* **One-hot sequence encoding**
* **Conv1D** for local sequence feature extraction
* **LSTM** for sequential dependency learning
* **PrimaryCaps** for capsule-based feature representation
* **Dynamic routing** between capsule layers
* **KcrCaps** for Kcr/non-Kcr classification
* Capsule-based feature visualization
* Sequence motif analysis
* **In-silico mutagenesis**

---

## Model Architecture

<img width="617" height="248" alt="image" src="https://github.com/user-attachments/assets/3396f3ad-392a-4810-9cef-91cd11e9920f" />

Each protein sequence is represented as a **31-amino-acid sequence window**, with the target lysine residue located at the center.

The overall architecture is:

```text
Protein Sequence
      ↓
One-hot Encoding
   (31 × 20)
      ↓
Conv1D
32 filters, kernel = 7
      ↓
LSTM
128 units
      ↓
PrimaryCaps
      ↓
Dynamic Routing
      ↓
KcrCaps
   (2 × 8)
      ↓
L2 Norm
      ↓
Kcr / non-Kcr Prediction
```

The final two KcrCaps represent:

* `Kcr`
* `non-Kcr`

The **L2 norm of each capsule vector** is used as the corresponding class score.

---

## Model Interpretability

DeepCap-Kcr supports multiple analyses for investigating learned sequence representations, including:

* Capsule feature visualization
* **t-SNE visualization**
* Dynamic-routing coefficient analysis
* Sequence motif investigation
* PrimaryCaps activation analysis
* **In-silico mutagenesis**
* Amino-acid substitution analysis
* Residue-position importance analysis

---

## Applications

* Protein lysine crotonylation site prediction
* Post-translational modification prediction
* Kcr sequence motif identification
* Sequence feature representation
* Capsule-based protein sequence analysis
* Amino-acid importance investigation
* Cross-species Kcr prediction

---

## Sequence Data Requirements

Each input sample should contain:

1. **Protein Sequence Window**

   * Length: **31 amino acids**
   * The candidate lysine (`K`) should be located at the center position

2. **Class Label**

   * Binary label:

     * `1` → Kcr
     * `0` → non-Kcr

### Example Input Data Format

Each row represents a 31-residue protein sequence centered on lysine with its corresponding label.

```text
AAAAAAAAAAAAAAAKAAAAAAAAAAAAAAA  1
GGGGGGGGGGGGGGGKGGGGGGGGGGGGGGG  0
```

---

## Sequence Encoding

Each amino acid is represented using a **20-dimensional one-hot vector**.

Therefore, each sequence is transformed into:

```text
31 × 20
```

where:

```text
31 = sequence length
20 = amino-acid encoding dimensions
```

---

## Dataset

After redundancy reduction, the dataset used in the DeepCap-Kcr study contained:

| Dataset          |   Kcr | non-Kcr |  Total |
| ---------------- | ----: | ------: | -----: |
| Training         | 6,975 |   6,975 | 13,950 |
| Independent Test | 2,989 |   2,989 |  5,978 |
| Total            | 9,964 |   9,964 | 19,928 |

The datasets used in the study are available in:

```text
all_data_used_in_this_study/
```

---

## Performance

### Five-Fold Cross-Validation

```text
Sensitivity  : 0.813
Specificity  : 0.862
Accuracy     : 0.834
MCC          : 0.6741
AUC          : 0.940
```

### Independent Test

```text
Sensitivity  : 0.824
Specificity  : 0.836
Accuracy     : 0.823
MCC          : 0.656
AUC          : 0.910
```

---

## Cross-Species Evaluation

DeepCap-Kcr was also evaluated using lysine crotonylation data from **papaya**.

```text
Positive samples : 3,453
Negative samples : 3,453
Mean AUC          : 0.820
```

This analysis was performed to investigate the generalization ability of the model across species.

---

## Repository Contents

```text
DeepCap-Kcr/
│
├── all_data_used_in_this_study/
│
├── five_fold_model_weights/
│
├── mutagenesis_file/
│
├── Sequences_for_motif-logo.rar
│
├── training.ipynb
│
├── independent test.ipynb
│
├── predictions.ipynb
│
├── predictions.py
│
├── visualizations.ipynb
│
├── DeepCap-Kcr_architecture.png
│
└── README.md
```

---

## Key Dependencies

```bash
- TensorFlow
- Keras
- NumPy
- pandas
- scikit-learn
- matplotlib
- Jupyter Notebook
```

---

## Utilized Versions

The original implementation was developed using:

```bash
- Python: 3.7.4
- Keras: 2.2.4
- TensorFlow backend
```

Because the model was developed using an older Keras/TensorFlow environment, a dedicated legacy environment is recommended for reproducing the original results.

---

## Environment Setup

```bash
conda create -n deepcap-kcr python=3.7.4 -y
conda activate deepcap-kcr
```

Install Keras:

```bash
pip install keras==2.2.4
```

Install additional libraries:

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

A TensorFlow version compatible with **Keras 2.2.4** should also be installed according to the operating system and GPU environment.

---

## Model Training

Run:

```text
training.ipynb
```

The basic workflow is:

```text
Protein Sequence
      ↓
31-residue window
      ↓
One-hot Encoding
      ↓
Conv1D
      ↓
LSTM
      ↓
PrimaryCaps
      ↓
Dynamic Routing
      ↓
KcrCaps
      ↓
Kcr Prediction
```

The trained five-fold model weights are available in:

```text
five_fold_model_weights/
```

---

## In-Silico Mutagenesis

DeepCap-Kcr also provides an **in-silico mutagenesis analysis** to investigate the contribution of individual amino acids.

The workflow is:

```text
Original Sequence
       ↓
Single Amino-Acid Mutation
       ↓
DeepCap-Kcr Prediction
       ↓
Prediction Change
       ↓
Residue Importance
```

Associated files are available in:

```text
mutagenesis_file/
```

---

## Motif Analysis

Sequences associated with highly activated capsule representations were investigated to identify sequence patterns related to lysine crotonylation.

Files used for motif analysis are available in:

```text
Sequences_for_motif-logo.rar
```

---

## Publication

**DeepCap-Kcr: accurate identification and investigation of protein lysine crotonylation sites based on capsule network**

Jhabindra Khanal, Hilal Tayara, Quan Zou, and Kil To Chong

*Briefings in Bioinformatics*, Volume 23, Issue 1, 2022, bbab492

**Article:**
https://academic.oup.com/bib/article/23/1/bbab492/6457166

**DOI:**
https://doi.org/10.1093/bib/bbab492

---

## Citation

If you use DeepCap-Kcr in your research, please cite:

```bibtex
@article{khanal2022deepcapkcr,
  title={DeepCap-Kcr: accurate identification and investigation of protein lysine crotonylation sites based on capsule network},
  author={Khanal, Jhabindra and Tayara, Hilal and Zou, Quan and Chong, Kil To},
  journal={Briefings in Bioinformatics},
  volume={23},
  number={1},
  pages={bbab492},
  year={2022},
  publisher={Oxford University Press},
  doi={10.1093/bib/bbab492}
}
```

---

## Contact

For questions, issues, or collaboration inquiries, please contact:

📧 **jhabindra@jbnu.ac.kr or jbkkor2014@gmail.com**


---



---

## License

This project is intended for **academic and research use**.

If you use the source code, datasets, pretrained models, or associated resources, please cite the original DeepCap-Kcr publication.
