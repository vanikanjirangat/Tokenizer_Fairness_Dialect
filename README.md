## This repo consists the scripts and notebooks related to our ACL 2025 main paper: 
# Script Biases and Language Disparities in LLMs 
* The LLM_FT folder consists of the scripts used to fine-tune the decoder-only (Llama 3.2, Phi 3.5, SILMA, etc.) models and encoder-only models
  * You can use a shell script or run for instance, python llama3_classification_hyper.py --lora_r 8 --epochs 8 --dropout 0.1
  * For inference: python llama3_classification_inference.py --experiment $OUT_PATH$
* The colab notebooks consists of the tokenizer analysis including vocabulary, parity etc. as mentioned in the paper.

### Requirements

