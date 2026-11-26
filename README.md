# 4DJM

4DJM 基于 Qwen3.5-9B 核验动态场景中的文本断言。模型从同步多视角视频均匀采样 16 个时间点，将每个时刻的视角拼成 672×672 图像。语义单元查询视觉特征，形成随时间变化的证据场。分类头结合存在性得分和多尺度时间模式匹配得分，输出断言的真假判断。

## 环境

使用 Linux、NVIDIA CUDA GPU 和 Python 3.11。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-cuda.txt
```

CUDA扩展需要与本机驱动、CUDA和PyTorch匹配。依赖文件记录训练环境的版本。

## 数据与配置

- 将 Qwen3.5-9B 基座、processor 和 tokenizer 放在 `models/Qwen3.5-9B`，或修改配置中的 `paths.model`。
- 将训练、验证和测试标注放在 `annotations/train.json`、`annotations/validation.json`、`annotations/test.json`。
- 标注为JSON数组，样本包含 `sample_id`、`source`、`scene_id`、`capability`、`assertion`、`answer` 和 `reason`。
- 按 `configs/scene_mappings.yaml` 准备原始视频或图像序列，或者修改其中的媒体路径。
- `configs/language.yaml`、`configs/evidence.yaml`、`configs/temporal.json` 分别配置语言适配、语义证据学习和时间配置学习。路径以仓库根目录为基准。

## 从头训练

先使用断言分类与解释生成监督训练语言LoRA，获得后续模型的初始化权重：

```bash
CUDA_VISIBLE_DEVICES=0 python train_language.py --config configs/language.yaml --formal-training
```

随后加载验证集选出的语言适配器，随机初始化语义单元提取器、视觉证据场及初始分类头，联合更新这些模块与语言LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 python train_evidence.py --config configs/evidence.yaml --formal-training
```

最后加载验证集选出的语义证据模型，冻结其自定义模块，随机初始化时间模式生成器与最终分类头，联合更新新增模块和语言LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 python train_temporal.py --config configs/temporal.json --run-dir outputs/temporal/train --formal-training
```

| 训练内容 | Epochs | LoRA学习率 | 自定义模块学习率 | Batch / 梯度累积 |
|---|---:|---:|---:|---:|
| 语言适配初始化 | 3 | 2e-5 | 分类头：1e-4 | 2 / 4 |
| 语义证据学习 | 3 | 2e-5 | 1e-4 | 1 / 8 |
| 时间配置学习 | 2 | 1e-6 | 5e-4 | 1 / 8 |

Qwen基座与视觉编码器始终冻结。训练目标为二分类损失加0.2倍解释生成损失。参考解释用于监督语言模型的生成。

## 评估

使用最终训练输出的验证集最优checkpoint：

```bash
CHECKPOINT=$(python -c 'import json; print(json.load(open("outputs/temporal/train/best_checkpoint.json"))["checkpoint"])')
CUDA_VISIBLE_DEVICES=0 python eval.py --checkpoint-dir "$CHECKPOINT" --output-dir outputs/evaluation --split test --allow-test
```
