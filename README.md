# LungRes80: Towards Tangled Surgical Workflow Recognition in Video-Assisted Thoracoscopic Surgery

## Abstract

Video-Assisted Thoracoscopic Surgery (VATS) is a minimally invasive procedure developed to remove specific lung segments for the treatment of early-stage lung diseases. The surgical procedure involves intricate vascular and bronchial anatomy to preserve as much lung tissue as possible, minimizing impact on the pulmonary function. To assist in monitoring and early warning of this high-risk surgical workflow, we build a new dataset, LungRes80, including 269,806 video frames with phase annotations sampled from 80 VATS cases.

LungRes80 presents unique challenges for hierarchical temporal modeling due to diverse short-term transitions between segmentectomy phases and latent long-term causal relations. To this end, we introduce an online baseline model termed LungReco. This framework employs Masked Causal Reasoning (MCR) to perform causal reasoning with semantic modeling from continuously updated memories along with pre-trained Large Language Models (LLMs), and combines it with Concurrent Spatial-Temporal encoding (CoST) for holistic bi-modal co-spatial-temporal aggregation across short- and long-term memories.

Furthermore, a new metric, called the Attentional Distraction Coefficient (ADC), is proposed to quantify the costs of intraoperative distraction and postoperative corrections by wrong predictions. We establish a comprehensive benchmark for surgical workflow recognition by evaluating representative models on LungRes80, AutoLaparo, and Cholec80, where our method consistently achieves state-of-the-art performance. Code and data will be available.

## Main Figure

![LungRes80 main figure](assets/teaser.png)

## Coming Soon

- Dataset access
- Pretrained models


## Citation

If this helps you, please cite this work:

```bibtex
@article{guo2026lungres80,
  title={LungRes80: Towards tangled surgical workflow recognition in video-assisted thoracoscopic surgery},
  author={Guo, Diandian and Yang, Shu and Pei, Jialun and Li, Jiaao and Wan, Yanhui and Chen, Hao and Heng, Pheng-Ann},
  journal={Medical Image Analysis},
  pages={104237},
  year={2026},
  publisher={Elsevier}
}
```

## Code

本仓库提供 LungRes80 数据集上的训练、在线推理与评估代码。

### Installation

```bash
conda env create -f environment.yml
conda activate LungReco
pip install -r requirements_lungreco.txt
```

运行前请准备：

- LungRes80 数据集；
- Kinetics-400 预训练的 `TimeSformer_divST_8x32_224_K400.pyth`；
- LLaVA v1.5 7B，默认使用 `liuhaotian/llava-v1.5-7b`。

### Data Preparation

```text
LungRes80/
├── frames/
│   ├── 01/
│   ├── 02/
│   └── ...
└── phase_annotations/
    ├── 01.txt
    ├── 02.txt
    └── ...
```

生成训练、验证和测试所需的 pickle 文件：

```bash
python tools/rebuild_lungres80_splits.py /path/to/LungRes80
```

数据划分为 50 个训练视频、15 个验证视频和 15 个测试视频。生成结果位于：

```text
LungRes80/labels/train/1fpstrain.pickle
LungRes80/labels/val/1fpsval.pickle
LungRes80/labels/test/1fpstest.pickle
```

### Training and Testing

```bash
PROJECT_ROOT=$PWD \
DATA_ROOT=/path/to/LungRes80 \
PRETRAINED_PATH=/path/to/TimeSformer_divST_8x32_224_K400.pyth \
LLAVA_MODEL_PATH=liuhaotian/llava-v1.5-7b \
TORCHRUN_BIN=$(command -v torchrun) \
RUN_NAME=lungreco_run1 \
bash scripts/train_lungres80.sh
```

默认使用两张 GPU、每卡 batch size 8、BF16 和 DeepSpeed。训练过程根据验证集保存最佳 checkpoint，完成后自动执行在线测试。在线验证和测试按视频时间顺序进行，每个视频开始时重置时序状态。

### Evaluation

测试完成后会生成：

```text
unrelaxed.jsonl
metrics_strict.json
metrics_relaxed_k5.json
```

也可以单独运行评估：

```bash
python evaluation_lungreco.py /path/to/unrelaxed.jsonl
python evaluation_lungreco.py /path/to/unrelaxed.jsonl --relaxed --relax-k 5
```

评估结果包括 Accuracy、Precision、Recall、Jaccard、ADC、Edit、F1@10、F1@25 和 F1@50。

## License

许可证见 [LICENSE](LICENSE)。
