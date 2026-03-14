#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
threshold_scan.py
概念压缩阈值扫描实验脚本（修复np.savez错误）
对91概念规模的自然涌现模型，扫描检测阈值（compression_synergy）从0.5到0.9，
记录每个阈值下检测到的压缩次数，并绘制曲线。
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径（根据实际项目结构调整）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.universe_enhanced import CognitiveUniverseEnhanced


def run_threshold_scan(
    thresholds: list,
    num_concepts: int = 91,
    iterations: int = 10000,
    network_seed: int = 42,
    detection_interval: int = 200,
    temp_save: bool = True
) -> dict:
    """
    对给定的阈值列表执行扫描实验。

    Args:
        thresholds (list): 待扫描的压缩检测阈值列表。
        num_concepts (int): 概念节点数量。
        iterations (int): 演化迭代次数。
        network_seed (int): 随机种子。
        detection_interval (int): 涌现检测间隔。
        temp_save (bool): 是否在每次迭代后保存临时结果（防止中断丢失）。

    Returns:
        dict: 阈值 -> 压缩次数的映射。
    """
    results = {}

    for i, thresh in enumerate(tqdm(thresholds, desc="阈值扫描进度")):
        # 创建宇宙实例（每次重新初始化，保证随机序列一致）
        universe = CognitiveUniverseEnhanced(
            individual_params=None,
            network_seed=network_seed,
            num_concepts=num_concepts
        )

        # 修改检测器的压缩协同性阈值
        universe.emergence_detector.thresholds['compression_synergy'] = thresh

        # 初始化语义网络（节点和边基于语义相似度生成）
        universe.initialize_semantic_network()

        # 运行演化并检测涌现
        observations = universe.evolve_with_emergence_detection(
            iterations=iterations,
            detection_interval=detection_interval
        )

        # 记录压缩事件数量
        comp_count = len(observations.get('natural_compressions', []))
        results[thresh] = comp_count

        # 可选：保存临时结果到JSON文件（覆盖保存最新状态）
        if temp_save:
            # 将浮点数键转换为字符串，以便JSON序列化
            str_key_results = {str(k): v for k, v in results.items()}
            with open('threshold_scan_temp.json', 'w', encoding='utf-8') as f:
                json.dump(str_key_results, f, indent=2)

    return results


def plot_scan_results(results: dict, save_path: str = None):
    """
    绘制阈值-压缩次数曲线。

    Args:
        results (dict): 阈值 -> 压缩次数字典（键为浮点数）。
        save_path (str, optional): 图片保存路径。
    """
    thresholds = sorted(results.keys())
    counts = [results[t] for t in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, counts, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Compression Synergy Threshold', fontsize=12)
    plt.ylabel('Number of Detected Compressions', fontsize=12)
    plt.title('Effect of Detection Threshold on Observed Compression Events\n(91 Concepts, 10,000 Iterations)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(thresholds)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 定义阈值扫描范围（0.5 到 0.9，步长 0.05）
    scan_thresholds = np.arange(0.5, 0.91, 0.05).round(2).tolist()
    print(f"将扫描以下阈值: {scan_thresholds}")

    # 运行扫描实验
    results = run_threshold_scan(
        thresholds=scan_thresholds,
        num_concepts=91,
        iterations=10000,
        network_seed=42,
        detection_interval=200,
        temp_save=True
    )

    # 打印结果
    print("\n阈值扫描结果:")
    for thresh in sorted(results.keys()):
        print(f" 阈值 {thresh:.2f} -> 压缩次数 {results[thresh]}")

    # 保存结果到JSON文件（键转为字符串）
    str_key_results = {str(k): v for k, v in results.items()}
    with open('../../results/analysis/add_scan/threshold_scan_results.json', 'w', encoding='utf-8') as f:
        json.dump(str_key_results, f, indent=2)
    print("结果已保存至 threshold_scan_results.json")

    # 也可以保存为npz格式（但需要字符串键，这里为兼容性跳过）
    # 绘制曲线
    plot_scan_results(results, save_path='../../results/analysis/add_scan/threshold_scan_plot.png')