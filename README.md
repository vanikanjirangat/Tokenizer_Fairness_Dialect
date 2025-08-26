This repo consists the scripts and notebooks related to our EMNLP 2025 main paper: 
## Tokenization and Representation Biases in Multilingual Models on Dialectal NLP Tasks 

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

* The folder consists of the scripts used to fine-tune the models for downstream tasks
  * You can use a shell script or run for instance, in dialect classification/, python llama3_classification_hyper.py --lora_r 8 --epochs 8 --dropout 0.1
  * For inference: python llama3_classification_inference.py --experiment OUTPUT-PATH
 
Intrinsic Analysis
------------

To evaluate the Tokenization Parity (TP), we adapted the code from https://github.com/AleksandarPetrov/tokenization-fairness.
To evaluate the Information Parity (IP), we adapted codes from https://github.com/tsalex1992/Information-Parity


