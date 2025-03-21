# Towards Generalizable Scene Change Detection

This repository represents the official implementation of the paper titled "Towards Generalizable Scene Change Detection".


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
We formulate the research problem by casting a fundamental question: *"Can contemporary SCD models detect arbitrary real-world changes beyond the scope of research data?"*
As shown in the above figure, their reported effectiveness does not hold in real-world applications. Specifically, they produce inconsistent change masks when the input order is reversed and exhibit significant performance drops when deployed to unseen domains with different visual features.
