# ALARM MAMI 实验复现

作者代码：https://github.com/Jian-Lang/ALARM

复现代码：https://github.com/lw1912190101/ALARM-MAMI-Reproduction

本仓库包含 ALARM 的 MAMI 复现脚本及结果汇总。原始算法、提示词和检测依据管理代码来自作者仓库；本次增加环境准备、MAMI 数据整理、单卡执行、分阶段续跑、50 示例对照和评估程序。原作者的项目说明见 [UPSTREAM_README.md](UPSTREAM_README.md)。

## 实验设置与结果

MAMI 官方训练集 10,000 条、测试集 1,000 条。使用 Qwen2.5-VL-72B-Instruct-AWQ 和 jina-clip-v2，在一张 96GB 显卡上依次执行各阶段及对照实验。

| 方法 | Acc | Macro-F1 |
| --- | --- | --- |
| ALARM AWQ | 81.60% | 81.56% |
| 50 示例对照 | 80.40% | 80.33% |

筛选 2,000 条高置信度训练样本，组成 833 对，生成经验并更新检测依据。测试时使用 15 条依据。结果汇总见 [results](results)。本次采用 AWQ 量化及固定种子，50 示例由训练集抽取并生成描述；相关实验设置见 [PROTOCOL.txt](reproduction/PROTOCOL.txt)。

## 运行

1. 根据 [RUN_INSTRUCTIONS.txt](reproduction/RUN_INSTRUCTIONS.txt) 准备环境、模型和数据。所列服务器路径为已完成实验的运行路径，在其他机器上须修改为实际路径。
2. 使用 `prepare_mami.py` 转换官方数据，学习输入与真实标签分别保存。
3. 使用 `run_pipeline.py` 依次执行伪标签、嵌入、配对、经验生成、依据更新及预测。
4. 使用 `run_fewshot.py` 运行 50 示例对照，使用 `evaluate.py` 和 `summarize_main.py` 评估。

`run_main_job.py` 是原实验服务器的调度脚本，固定使用物理 GPU 4，其他机器应调整设备选择及 Python 路径后运行。各阶段与对照顺序执行。

原始模因图片、文字转录、模型权重、逐条实验输出和服务器内部运行记录不包含在此公开仓库中。MAMI 数据请向[数据集发布方](https://github.com/MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI-)获取。已验证依赖版本见 `reproduction/pip_freeze.txt`。

## 来源

论文：From Shallow Humor to Metaphor: Towards Label-Free Harmful Meme Detection via LMM Agent Self-Improvement。

作者：Jian Lang、Rongpei Hong、Ting Zhong、Leiting Chen、Qiang Gao、Fan Zhou。

原论文：https://arxiv.org/abs/2512.21598

原代码及修改文件保留其原有权利归属，本仓库不另行声明上游代码许可。
