# Towards Generalizable Scene Change Detection

This repository represents the official implementation of the paper titled "Towards Generalizable Scene Change Detection".

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2409.06214)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
[![Website](https://img.shields.io/badge/Website-CVPR_2025-blue)](https://cvpr.thecvf.com/virtual/2025/poster/34711)

<p align="center">
  <a href=""><strong>Jaewoo Kim</strong></a>
  ·  
  <a href=""><strong>Uehwan Kim</strong></a>
  <br>
  <strong>CVPR 2025</strong>
</p>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-generalizable-scene-change-detection/scene-change-detection-on-changevpr)](https://paperswithcode.com/sota/scene-change-detection-on-changevpr?p=towards-generalizable-scene-change-detection)

<div align='center'>
  <br><img src="image/title_img-1.png" width=100%>
  <br>Comparative results of the current state-of-the-art model and our GeSCF on various unseen images.
</div>

## :bulb: Problem Formulation
We formulate the research problem by casting a fundamental question: **"Can contemporary SCD models detect arbitrary real-world changes beyond the scope of research data?"**
Our findings, as shown in the figure above, indicate that their reported effectiveness does not hold in real-world applications. Specifically, we observe that they (1) <ins>produce inconsistent change masks</ins> when the input order is reversed, and (2) <ins>exhibit significant performance drops</ins> when deployed to unseen domains with different visual features. 

In this work, we address these two pivotal SCD problems by proposing a novel framework (GeSCF) and a novel benchmark (GeSCD) to foster SCD research in generalizability. 

## 🗂 Datasets
For a comprehensive evaluation of SCD performance, we consider three standard SCD datasets with different characteristics and our proposed ChangeVPR dataset. 

Please follow this <a href="https://huggingface.co/datasets/Flourish/VL-CMU-CD/blob/main/VL-CMU-CD-binary255.zip"><strong>page</strong></a> to download the VL-CMU-CD dataset.

Please follow this <a href="https://kensakurada.github.io/pcd_dataset.html"><strong>page</strong></a> to download the TSUNAMI dataset.

Please follow this <a href="https://github.com/SAMMiCA/ChangeSim"><strong>page</strong></a> to download the ChangeSim dataset.

To download our **ChangeVPR** dataset, go <a href="https://docs.google.com/forms/d/e/1FAIpQLSeTYO0D5p1jEc5NlYbKR9xWkd8NSXzLCCCLR3OTlQ2LPCZk2Q/viewform?usp=sharing"><strong>here</strong></a> to download it. 

## 🛠 Requirements 


## 📃 Citation
If you find the work useful for your research, please cite:
```bibtex
@inproceedings{Kim2024TowardsGS,
  title={Towards Generalizable Scene Change Detection},
  author={Jaewoo Kim and Uehwan Kim},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

