This repo consists the scripts and notebooks related to our ACL 2025 main paper: ## Script Biases and Language Disparities in LLMs 

Datasets
============
Intrinsic Analysis
------------

Extrinsic Analysis
------------
* Dialect Identifications
  
   > Mono-label DI (NADI-2023 for Arabic [https://nadi.dlnlp.ai/2023/], ILI for Hindi [https://github.com/kmi-linguistics/vardial2018], GDI-2018 for Swiss German [https://drive.switch.ch/index.php/s/DZycFA9DPC8FgD9])
   
   > Multi-label (Spanish and French)- [https://sites.google.com/view/vardial-2024/shared-tasks/dsl-ml]
   
 *  Topic Classification & Extractive QA
   > From DialectBench dataset: https://github.com/ffaisal93/DialectBench/tree/main

 


Codes
============
Extrinsic Analysis
------------

* The LLM_FT folder consists of the scripts used to fine-tune the decoder-only (Llama 3.2, Phi 3.5, SILMA, etc.) models and encoder-only models
  * You can use a shell script or run for instance, python llama3_classification_hyper.py --lora_r 8 --epochs 8 --dropout 0.1
  * For inference: python llama3_classification_inference.py --experiment OUTPUT-PATH
 
Intrinsic Analysis
------------

To evaluate the Tokenization Parity (TP), we adapted the code from https://github.com/AleksandarPetrov/tokenization-fairness.
To evaluate the Information Parity (IP), we adapted codes from https://github.com/tsalex1992/Information-Parity

* The colab notebooks consists of the tokenizer analysis including vocabulary, parity etc. as mentioned in the paper.
  * The code can be easily enhanced to analyze any language within the FLORES dataset.

Requirements
============
