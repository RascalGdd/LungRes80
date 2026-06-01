# LungRes80: Towards Tangled Surgical Workflow Recognition in Video-Assisted Thoracoscopic Surgery

## Abstract

Video-Assisted Thoracoscopic Surgery (VATS) is a minimally invasive procedure developed to remove specific lung segments for the treatment of early-stage lung diseases. The surgical procedure involves intricate vascular and bronchial anatomy to preserve as much lung tissue as possible, minimizing impact on the pulmonary function. To assist in monitoring and early warning of this high-risk surgical workflow, we build a new dataset, LungRes80, including 269,806 video frames with phase annotations sampled from 80 VATS cases.

LungRes80 presents unique challenges for hierarchical temporal modeling due to diverse short-term transitions between segmentectomy phases and latent long-term causal relations. To this end, we introduce an online baseline model termed LungReco. This framework employs Masked Causal Reasoning (MCR) to perform causal reasoning with semantic modeling from continuously updated memories along with pre-trained Large Language Models (LLMs), and combines it with Concurrent Spatial-Temporal encoding (CoST) for holistic bi-modal co-spatial-temporal aggregation across short- and long-term memories.

Furthermore, a new metric, called the Attentional Distraction Coefficient (ADC), is proposed to quantify the costs of intraoperative distraction and postoperative corrections by wrong predictions. We establish a comprehensive benchmark for surgical workflow recognition by evaluating representative models on LungRes80, AutoLaparo, and Cholec80, where our method consistently achieves state-of-the-art performance. Code and data will be available.

## Main Figure

![LungRes80 main figure](assets/teaser.png)

## Coming Soon

- Code
- Dataset access
- Pretrained models
- Evaluation scripts
- Documentation
