# Supplementary Code for [SERL]

This repository contains the **supplementary code** used to implement the experiments described in the paper:
In this project, we provide the data, prompts, and running scripts for training the General QA task. 
The complete data and prompts(Summarization and Open writing) will be made available after the results are released.

> **[SERL]**   
> **Conference/Journal:** [AAAI2026]  

The implementation is based on the [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) library by Hugging Face, which provides state-of-the-art methods for post-training foundation models using techniques such as DPO, PPO, SFT, and more.

---

## 📦 Requirements

Before running the code, make sure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

All training scripts are located in the `scripts/` directory. Here is our running step our pipeline:

### Training with SERL

```bash
python -m SERL.examples.scripts.SERL \
```

---

## 📁 Directory Structure

```
.
├── README.md                   <- This file
├── requirements.txt            <- Required packages
├── examples/
│   ├── scripts/
│   │   ├── SERL.py             <- SERL training script
│   │   └── ...                 <- Additional scripts
└── data/                       <- Optional: Preprocessed datasets
```

---

## 📚 References

Our implementation builds on top of the TRL library:

```bibtex
@misc{vonwerra2022trl,
  author = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallouédec},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/trl}}
}
```

---

## ✅ Notes for Reproducibility

- All random seeds are fixed in the training scripts.
- We use deterministic versions of PyTorch and Transformers where possible.
- The exact version of the libraries used is listed in `requirements.txt`.

---

## 📝 License

This code is released under the **Apache 2.0 License**, same as the TRL library.

---
