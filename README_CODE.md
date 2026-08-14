# LungReco

LungReco 是面向肺切除手术视频的在线阶段识别模型。本仓库提供 LungRes80 数据集上的训练、推理与评估代码。

## 安装

```bash
conda env create -f environment.yml
conda activate LungReco
pip install -r requirements_lungreco.txt
```

运行前请准备：

- LungRes80 数据集；
- Kinetics-400 预训练的 `TimeSformer_divST_8x32_224_K400.pyth`；
- LLaVA v1.5 7B，默认使用 `liuhaotian/llava-v1.5-7b`。

## 数据准备

数据目录应包含逐帧图像和阶段标注：

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

数据划分为 50 个训练视频、15 个验证视频和 15 个测试视频。脚本会检查视频 ID、标注和图像路径，并将生成结果写入：

```text
LungRes80/labels/train/1fpstrain.pickle
LungRes80/labels/val/1fpsval.pickle
LungRes80/labels/test/1fpstest.pickle
```

## 训练与测试

使用以下命令启动完整实验：

```bash
PROJECT_ROOT=$PWD \
DATA_ROOT=/path/to/LungRes80 \
PRETRAINED_PATH=/path/to/TimeSformer_divST_8x32_224_K400.pyth \
LLAVA_MODEL_PATH=liuhaotian/llava-v1.5-7b \
TORCHRUN_BIN=$(command -v torchrun) \
RUN_NAME=lungreco_run1 \
bash scripts/train_lungres80.sh
```

默认使用两张 GPU、每卡 batch size 8、BF16 和 DeepSpeed。训练过程根据验证集保存最佳 checkpoint，完成后自动执行在线测试。

在线验证和测试按视频时间顺序进行。模型只使用当前帧及其之前的信息，每个视频开始时会重置时序状态。

## 评估

测试完成后会保存逐帧预测和指标：

```text
unrelaxed.jsonl
metrics_strict.json
metrics_relaxed_k5.json
```

`unrelaxed.jsonl` 包含逐帧 logits、原始预测、原始概率和平滑概率。严格指标和 relaxed k=5 指标均由同一份预测文件计算。

也可以单独运行评估：

```bash
python evaluation_lungreco.py /path/to/unrelaxed.jsonl
python evaluation_lungreco.py /path/to/unrelaxed.jsonl --relaxed --relax-k 5
```

评估结果包括 Accuracy、Precision、Recall、Jaccard、ADC、Edit、F1@10、F1@25 和 F1@50。

## 许可

许可证见 [LICENSE](LICENSE)。
