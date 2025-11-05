# DDPCFMD
Dual-driven physically consistent fusion for multimodal dehazing — code, data pointers, and eval scripts.

# Dual-Driven Physically Consistent Fusion for Multimodal Dehazing

*(This section will be made public after the paper is officially accepted.)*

## Overview
This repository contains the official implementation, dataset pointers, and evaluation scripts for our paper:
**"Dual-Driven Physically Consistent Fusion for Multimodal Dehazing"**  
We introduce a dual-driven (physics + data) multimodal dehazing framework that tightly couples imaging physics with deep learning for robust visibility restoration under dense haze.  
Code and datasets will be publicly released upon paper acceptance.

---


## Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/Unconventional-Vision-Lab-ZZU/DDPCFMD.git
cd DDPCFMD
conda create -n ddpcfmd python=3.8
conda activate ddpcfmd
pip install -r requirements.txt
```

---

*(This section will be made public after the paper is officially accepted.)*

## Dataset
We provide both synthetic and real-scene multimodal datasets:
- **syn-MH**: A physically aligned synthetic multimodal haze dataset.  
- **real-MH**: A real-scene multimodal dataset collected with a custom imaging system.  

Dataset links and instructions will be released after paper acceptance.  
Please contact us for access during the review process.

---

*(This section will be made public after the paper is officially accepted.)*

## Training
To train the model from scratch:
```bash
python train.py --config configs/train_ddpcfmd.yaml
```
To evaluate on the test split:
```bash
python eval.py --config configs/test_ddpcfmd.yaml
```

Training and inference share consistent normalization, optimizer, and resolution settings across all modalities.

---



## Results
Our method achieves state-of-the-art performance on both synthetic and real-scene haze datasets.  
Metrics include **PSNR**, **SSIM**, and **NIQE**, along with **Params**, **MACs**, and **runtime** for efficiency comparison.  
Full result tables, model checkpoints, and visual examples will be provided after publication.

---

*(This section will be made public after the paper is officially accepted.)*

## Citation
If you find this work useful, please cite:
bibtex
@article{DDPCFMD2025,
  title={Dual-Driven Physically Consistent Fusion for Multimodal Dehazing},
  author={Author names},
  journal={IEEE Transactions on Multimedia},
  year={2025}
}
```

---

*(This section will be made public after the paper is officially accepted.)*

## Contact
For any inquiries, please contact:
**Unconventional Vision Lab, Zhengzhou University (ZZU)**  
📧 Contact: [zzu_xzq@zzu.edu.cn, iexumingliang@zzu.edu.cn, iemyjiu@zzu.edu.cn]  
🌐 Project Page: [https://github.com/Unconventional-Vision-Lab-ZZU/DDPCFMD]

