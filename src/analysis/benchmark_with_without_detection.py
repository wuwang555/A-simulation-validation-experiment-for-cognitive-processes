#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_time.py

对比纯公设演化（无检测）与完整演化（含检测）的运行时间。
"""

import sys
import os
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emergence.universe import CognitiveUniverse
from emergence.universe_enhanced import CognitiveUniverseEnhanced
from utils.individual_variation import IndividualVariation, create_enhanced_individual_params
from config import BASE_PARAMETERS, VARIATION_RANGES

random.seed(42)
np.random.seed(42)


# ============================================================
# 1. 纯公设模式（无检测）
# ============================================================
class CoreOnlyUniverse(CognitiveUniverse):
    """
    只保留时空图与能量最小化，移除所有观测、记录、检测。
    """

    def __init__(self, individual_params=None, network_seed=42, num_concepts=None):
        super().__init__(individual_params, network_seed)
        self.num_concepts = num_concepts

    def evolve_core_only(self, iterations=10000):
        """
        执行纯公设演化，不记录任何观测。
        """
        # 初始化网络（重要！）
        self.initialize_semantic_network()

        for i in range(iterations):
            self.iteration_count += 1

            # 能量优化（学习）
            self.basic_energy_optimization()

            # 随机遍历
            if random.random() < 0.3:
                self._random_traversal()

            # 遗忘（每10步）
            if i % 10 == 0:
                self.apply_basic_forgetting()

    def initialize_semantic_network(self):
        from core.semantic_network import SemanticConceptNetwork

        semantic_net = SemanticConceptNetwork()
        semantic_net.build_comprehensive_network(num_concepts=self.num_concepts)

        nodes = list(semantic_net.concept_definitions.keys())
        self.G.add_nodes_from(nodes)

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                similarity = semantic_net.calculate_semantic_similarity(node1, node2)
                if similarity > 0.1:
                    energy = 2.0 - similarity * 1.5
                    energy = max(0.3, min(2.0, energy))
                    self.G.add_edge(node1, node2,
                                    weight=energy,
                                    traversal_count=0,
                                    original_weight=energy,
                                    similarity=similarity)
                    self.last_activation_time[(node1, node2)] = 0


def run_core_only_benchmark(scales, num_individuals=3, iterations=10000):
    """
    运行纯公设模式，返回结果列表。
    """
    results = []
    for scale in scales:
        for ind in range(num_individuals):
            var_sim = IndividualVariation(BASE_PARAMETERS, VARIATION_RANGES)
            base_params = var_sim.generate_individual(f"scale{scale}_ind{ind}")
            ind_params = create_enhanced_individual_params(base_params)

            universe = CoreOnlyUniverse(
                individual_params=ind_params,
                network_seed=42 + ind,
                num_concepts=scale
            )

            start = time.time()
            universe.evolve_core_only(iterations=iterations)
            elapsed = time.time() - start

            results.append({
                'mode': 'core_only',
                'scale': scale,
                'individual': ind,
                'time_sec': elapsed
            })

            print(f"[core_only] 规模 {scale} 个体 {ind+1} 完成，耗时 {elapsed:.2f} 秒")
    return results


# ============================================================
# 2. 完整模式（含检测）
# ============================================================
def run_full_benchmark(scales, num_individuals=3, iterations=10000):
    """
    运行完整模式（CognitiveUniverseEnhanced + 检测），返回结果列表。
    """
    results = []
    for scale in scales:
        for ind in range(num_individuals):
            var_sim = IndividualVariation(BASE_PARAMETERS, VARIATION_RANGES)
            base_params = var_sim.generate_individual(f"scale{scale}_ind{ind}")
            ind_params = create_enhanced_individual_params(base_params)

            universe = CognitiveUniverseEnhanced(
                individual_params=ind_params,
                network_seed=42 + ind,
                num_concepts=scale
            )

            # 重要：初始化网络（完整模式不会自动初始化）
            universe.initialize_semantic_network()

            start = time.time()
            universe.evolve_with_emergence_detection(
                iterations=iterations,
                detection_interval=100
            )
            elapsed = time.time() - start

            results.append({
                'mode': 'full_with_detection',
                'scale': scale,
                'individual': ind,
                'time_sec': elapsed
            })

            print(f"[full] 规模 {scale} 个体 {ind+1} 完成，耗时 {elapsed:.2f} 秒")
    return results


# ============================================================
# 3. 主函数：依次运行并输出对比
# ============================================================
def main():
    scales = [51, 71, 91, 111]
    num_individuals = 3
    iterations = 10000

    print("=" * 70)
    print("时间对比实验：纯公设 vs 完整检测")
    print(f"规模列表: {scales}")
    print(f"每个规模个体数: {num_individuals}")
    print(f"迭代次数: {iterations}")
    print("=" * 70)

    # 1. 运行纯公设模式
    print("\n--- 开始测量纯公设模式（无检测）---")
    core_results = run_core_only_benchmark(scales, num_individuals, iterations)

    # 2. 运行完整模式
    print("\n--- 开始测量完整模式（含检测）---")
    full_results = run_full_benchmark(scales, num_individuals, iterations)

    # 3. 汇总统计
    core_by_scale = {}
    full_by_scale = {}
    for r in core_results:
        s = r['scale']
        core_by_scale.setdefault(s, []).append(r['time_sec'])
    for r in full_results:
        s = r['scale']
        full_by_scale.setdefault(s, []).append(r['time_sec'])

    print("\n" + "=" * 70)
    print("各规模平均耗时（秒）")
    print(f"{'规模':>6} {'纯公设':>12} {'完整检测':>12} {'差异':>12} {'比例':>10}")
    print("-" * 70)
    for scale in scales:
        core_avg = np.mean(core_by_scale[scale])
        full_avg = np.mean(full_by_scale[scale])
        diff = full_avg - core_avg
        ratio = full_avg / core_avg if core_avg > 0 else float('inf')
        print(f"{scale:>6} {core_avg:>12.2f} {full_avg:>12.2f} {diff:>+12.2f} {ratio:>9.2f}x")

    total_core = sum(r['time_sec'] for r in core_results)
    total_full = sum(r['time_sec'] for r in full_results)
    print("-" * 70)
    print(f"{'总计':>6} {total_core:>12.2f} {total_full:>12.2f} {total_full - total_core:>+12.2f} {total_full/total_core:>9.2f}x")

    # 4. 保存结果到 CSV
    import pandas as pd
    df = pd.DataFrame(core_results + full_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"../../results/analysis/time/time_comparison_{timestamp}.csv"
    df.to_csv(out_file, index=False)
    print(f"\n详细结果已保存至 {out_file}")


if __name__ == "__main__":
    main()