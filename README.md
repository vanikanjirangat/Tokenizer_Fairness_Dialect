This repo consists the scripts and notebooks related to our EMNLP 2025 main paper: 
## Tokenization and Representation Biases in Multilingual Models on Dialectal NLP Tasks 

## Intrinsic Analysis

Intrinsic analysis was performed using **Tokenization Parity (TP)** and **Information Parity (IP)** metrics on the **FLORES-200 parallel dataset**.

### 1️⃣ Tokenization Parity (TP)
- Adapted from: [https://github.com/AleksandarPetrov/tokenization-fairness](https://github.com/AleksandarPetrov/tokenization-fairness)

### 2️⃣ Information Parity (IP)
- Adapted from: [https://github.com/tsalex1992/Information-Parity](https://github.com/tsalex1992/Information-Parity)
---

## Extrinsic Analysis

### 1️⃣ Dialect Identification

**Mono-label DI:**
- **Arabic:** NADI-2023 — [https://nadi.dlnlp.ai/2023/](https://nadi.dlnlp.ai/2023/)  
- **Hindi:** ILI — [https://github.com/kmi-linguistics/vardial2018](https://github.com/kmi-linguistics/vardial2018)  
- **Swiss German:** GDI-2018 — [https://drive.switch.ch/index.php/s/DZycFA9DPC8FgD9](https://drive.switch.ch/index.php/s/DZycFA9DPC8FgD9)  

**Multi-label DI (Spanish & French):**
- [https://sites.google.com/view/vardial-2024/shared-tasks/dsl-ml](https://sites.google.com/view/vardial-2024/shared-tasks/dsl-ml)

---

### 2️⃣ Topic Classification & Extractive QA

- Dataset: **DialectBench** — [https://github.com/ffaisal93/DialectBench/tree/main](https://github.com/ffaisal93/DialectBench/tree/main)  
- The folder contains scripts used to fine-tune models for downstream tasks.

---
  ## Example Usage

  Run the dialect classification script with custom hyperparameters:

```bash
python llama3_classification_hyper.py --lora_r 8 --epochs 8 --dropout 0.1


python llama3_classification_inference.py --experiment OUTPUT-PATH
 




