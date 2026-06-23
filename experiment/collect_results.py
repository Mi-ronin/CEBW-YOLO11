import os
import json
import pandas as pd
from glob import glob

def collect_metrics(exp_dir_pattern='runs/*/metrics.json'):
    metrics_files = glob(exp_dir_pattern)
    results = []
    for mf in metrics_files:
        exp_name = mf.split(os.sep)[1]   # 假设结构 runs/exp_name/metrics.json
        with open(mf, 'r') as f:
            data = json.load(f)
        row = {
            'Model': exp_name,
            'Precision': data.get('macro_precision', 0),
            'Recall': data.get('macro_recall', 0),
            'F1': data.get('macro_f1', 0),
            'mAP@0.5': data.get('mAP', 0),
            'AP_small': data.get('ap_per_class', [0,0,0,0])[0],
            'AP_medium': data.get('ap_per_class', [0,0,0,0])[1],
            'AP_large': data.get('ap_per_class', [0,0,0,0])[2],
            'MR': data.get('MR', 0),
            'FDR': data.get('FDR', 0),
            'Params': data.get('params', 0),
            'FLOPs': data.get('flops', 0),
            'FPS': data.get('fps', 0),
        }
        results.append(row)
    df = pd.DataFrame(results)
    df.to_csv('collected_results.csv', index=False)
    return df

if __name__ == '__main__':
    collect_metrics()