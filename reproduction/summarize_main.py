"""Summarize completed, paired MAMI predictions without tuning to test results."""
import json
from pathlib import Path
import sys
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

root = Path(sys.argv[1])
def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
gold = json.loads((root / 'data/gold_labels.json').read_text())
alarm = {x['id']: x['pred'] for x in read(root / 'runs/main_awq/predictions.jsonl')}
baseline = {x['id']: x['pred'] for x in read(root / 'runs/main_fewshot_awq/predictions.jsonl')}
ids = sorted(x['id'] for x in read(root / 'data/full/data.jsonl') if x['split'] == 'test')
assert set(alarm) == set(baseline) == set(ids) and len(ids) == 1000
y = [gold[i] for i in ids]
metrics = {}
for name, predictions in [('ALARM', alarm), ('Qwen_50shot_reconstructed', baseline)]:
    yp = [predictions[i] for i in ids]
    metrics[name] = {'accuracy': accuracy_score(y, yp), 'macro_f1': f1_score(y, yp, average='macro')}
fixed = sum(alarm[i] == gold[i] and baseline[i] != gold[i] for i in ids)
broken = sum(alarm[i] != gold[i] and baseline[i] == gold[i] for i in ids)
report = {'samples': 1000, 'model': 'Qwen/Qwen2.5-VL-72B-Instruct-AWQ',
          'hardware_scope': 'One physical GPU, sequential stages and baseline; explicitly requested by user.',
          'results': metrics, 'paired_fixes': fixed, 'paired_regressions': broken,
          'mcnemar_exact_p': binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0,
          'paper_targets_percent': {'ALARM': {'accuracy': 81.28, 'macro_f1': 81.25},
                                    'Qwen_50shot': {'accuracy': 78.70, 'macro_f1': 78.65}},
          'limitations': ['Single-GPU AWQ quantization; this is not a full-precision table reproduction.',
                         'Released greedy pairing differs from paper bidirectional description.',
                         'Author precision/averaging details and 50-shot demo IDs/captions are not fully specified.',
                         'One fixed run; no official-test tuning; baseline captions are reconstructed locally.']}
(root / 'runs/main_comparison.json').write_text(json.dumps(report, indent=2))
a, b = metrics['ALARM'], metrics['Qwen_50shot_reconstructed']
lines = [
    'ALARM：MAMI 完整主实验记录',
    '范围：官方训练集 10,000 条、测试集 1,000 条；Qwen2.5-VL-72B-Instruct-AWQ；单卡串行运行。',
    f"ALARM：准确率 {a['accuracy']:.2%}，Macro-F1 {a['macro_f1']:.2%}。",
    f"50 示例对照：准确率 {b['accuracy']:.2%}，Macro-F1 {b['macro_f1']:.2%}。",
    f"准确率差值 {(a['accuracy']-b['accuracy'])*100:+.2f} 个百分点；Macro-F1 差值 {(a['macro_f1']-b['macro_f1'])*100:+.2f} 个百分点。",
    f"同一测试集上，ALARM 纠正 {fixed} 条对照错误，同时把 {broken} 条对照正确预测改错。配对精确检验 p={report['mcnemar_exact_p']:.4g}。",
    '论文报告 ALARM 准确率 81.28%、Macro-F1 81.25%；本记录直接报告上述完整测试集的实际结果。',
    '解释边界：按用户单卡要求使用 AWQ 量化权重，不作为全精度结果。采用官方代码的贪心一对一配对，论文文字描述为双向最近邻。推理采用论文的确定性设置和 15 条规则。',
    '50 个训练示例的编号及图像描述由本次固定抽样与本地模型生成，原作者未提供相同示例清单；因此是按论文设置重建的对照。',
    '本次结果来自一次固定设置运行，不能据此宣称所有随机种子下均有提升。详见 PROTOCOL.txt、逐条预测与中间经验。',
]
(root / 'runs/主实验结果与分析.txt').write_text('\n'.join(lines) + '\n')
print(json.dumps(report, indent=2))
