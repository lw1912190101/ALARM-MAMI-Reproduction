"""The only experiment stage that reads gold labels; never used to tune the run."""
import argparse
import json
import math
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def evaluate(data, gold_path, run):
    records = rows(data)
    train_ids = {x['id'] for x in records if x['split'] == 'train'}
    eval_ids = {x['id'] for x in records if x['split'] == 'test'}
    assert train_ids.isdisjoint(eval_ids)
    labels = rows(run / 'labels.jsonl')
    assert len(labels) == len(train_ids) and {x['id'] for x in labels} == train_ids
    assert all(math.isfinite(x['prob0']) and math.isfinite(x['prob1']) and
               abs(x['prob0'] + x['prob1'] - 1) < 1e-5 for x in labels)
    pairs = json.loads((run / 'pairs.json').read_text())
    assert pairs and all(x['id1'] in train_ids and x['id2'] in train_ids for x in pairs)
    assert len(rows(run / 'experiences.jsonl')) == len(pairs)
    assert len(rows(run / 'reference_history.jsonl')) == len(pairs)
    refs = json.loads((run / 'references.json').read_text())
    assert any(x['reference'] != 'placeholder' for x in refs)
    gold = json.loads(gold_path.read_text())
    report = {'train_samples': len(train_ids), 'evaluation_samples': len(eval_ids), 'pairs': len(pairs),
              'real_references': sum(x['reference'] != 'placeholder' for x in refs),
              'config': json.loads((run / 'config.json').read_text()), 'results': {}}
    selected = json.loads((run / 'selected.json').read_text())
    report['pseudo_label_diagnostics'] = {
        'all_training_accuracy': sum(x['pred'] == gold[x['id']] for x in labels) / len(labels),
        'selected_accuracy': sum(x['pred'] == gold[x['id']] for x in selected) / len(selected),
        'selected_samples': len(selected),
        'note': 'Computed after the completed run; never used for selection or tuning.'}
    for name, filename in [('ALARM', 'predictions.jsonl'), ('zero_shot_control', 'direct_predictions.jsonl')]:
        if name == 'zero_shot_control' and not (run / filename).exists():
            continue
        predictions = rows(run / filename)
        assert len(predictions) == len(eval_ids) and {x['id'] for x in predictions} == eval_ids
        assert all(x['pred'] in [0, 1] for x in predictions)
        if name == 'ALARM':
            assert all(x['rules'] > 0 for x in predictions)
        y_true = [gold[x['id']] for x in predictions]
        y_pred = [x['pred'] for x in predictions]
        report['results'][name] = {
            'accuracy': accuracy_score(y_true, y_pred), 'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'class_f1': f1_score(y_true, y_pred, labels=[0, 1], average=None).tolist(),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            'fallback_count': sum(x['fallback'] for x in predictions),
            'inference_seconds': sum(x['seconds'] for x in predictions),
        }
    report['technical_gate_passed'] = all(x['fallback_count'] / len(eval_ids) <= 0.05 for x in report['results'].values())
    report['scope_note'] = 'Integrity checks do not establish an improvement. Zero-shot control is not the paper 50-shot baseline.'
    if 'AWQ' in report['config']['model']:
        report['scope_note'] += ' This run uses quantized AWQ weights.'
    (run / 'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'config'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    evaluate(a.data, a.gold, a.run)
