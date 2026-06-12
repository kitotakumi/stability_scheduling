#!/usr/bin/env python3
"""
PR 実験結果の追加分析スクリプト

run_pr_experiment.py が生成した JSON から anytime HV / 領域別 anytime HV を生成する。

使い方:
  python analyze_pr_experiment.py results/pr_experiment_<timestamp>/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')

from run_pr_experiment import (
    ILS_METHODS, METHOD_LABELS, METHOD_COLORS,
    pareto_front_2d, hv_2d,
    plot_anytime_hv_pr, plot_anytime_region_hv_pr,
)


def load_and_plot(result_dir):
    for prob_dir_name in sorted(os.listdir(result_dir)):
        prob_dir = os.path.join(result_dir, prob_dir_name)
        if not os.path.isdir(prob_dir):
            continue

        for fname in sorted(os.listdir(prob_dir)):
            if not fname.startswith('results_') or not fname.endswith('.json'):
                continue

            json_path = os.path.join(prob_dir, fname)
            print(f"処理中: {json_path}")

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            methods = [mk for mk in ILS_METHODS if f'{mk}_histories' in data]
            w_label = fname[len('results_'):-len('.json')]
            prob_label = prob_dir_name
            init_ms = data.get('init_makespan', 1)

            all_stab = []
            for mk in methods:
                for hist in data.get(f'{mk}_histories', []):
                    if hist:
                        all_stab.extend(h[1] for h in hist if h[2])
            stab_ref = (max(all_stab) * 1.05) if all_stab else 30.0
            ref_point = (init_ms * 1.05, stab_ref)

            plot_anytime_hv_pr(data, methods, w_label, prob_dir, prob_label, ref_point)
            plot_anytime_region_hv_pr(data, methods, w_label, prob_dir, prob_label, init_ms)
            print(f"  → anytime_hv_{w_label}.png, anytime_region_hv_{w_label}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', help='pr_experiment の出力ディレクトリ')
    args = parser.parse_args()
    load_and_plot(args.result_dir)
    print("完了")


if __name__ == '__main__':
    main()
