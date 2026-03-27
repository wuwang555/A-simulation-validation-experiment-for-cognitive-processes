#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
认知图客观指标分析脚本（中心节点 + 节点总频次 Zipf）

功能：
1. 运行涌现实验，保存能耗历史、压缩事件、迁移事件。
2. 指标1：能耗下降曲线的幂律/指数拟合，计算衰减指数和拟合优度。
3. 指标2：中心概念出现次数的 Zipf 检验（幂律拟合），绘制 log-log 图。
4. 指标3：概念节点在压缩集群中出现的总次数（中心+成员）的 Zipf 检验。
输出目录: results/analysis/objective_metrics/
"""

import os
import sys
import json
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import stats
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergence.universe_enhanced import CognitiveUniverseEnhanced
from utils.individual_variation import IndividualVariation, create_enhanced_individual_params
from config import BASE_PARAMETERS, VARIATION_RANGES

# 设置中文字体（如果系统有）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ObjectiveMetricsAnalysis:
    """
    认知图客观指标分析类（中心节点 + 节点总频次 Zipf）
    """

    def __init__(self, output_dir="results/analysis/objective_metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.output_dir / "figures"
        self.fig_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_data = []                 # 每个个体的汇总信息
        self.center_counter = Counter()         # 中心节点出现次数
        self.node_total_counter = Counter()     # 节点在压缩事件中出现总次数（去重）

    # ---------- 拟合函数定义 ----------
    @staticmethod
    def power_law(x, a, b, c):
        """幂律函数 a * x^(-b) + c"""
        return a * np.power(x, -b) + c

    @staticmethod
    def exp_decay(x, a, b, c):
        """指数衰减 a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c

    def fit_energy_curve(self, x, y):
        """
        对能耗历史拟合两种模型，返回最佳模型的参数和指标。
        返回字典：{'model': 'power'/'exp', 'params': [a,b,c], 'r2': float, 'rmse': float, 'fluctuation': float}
        """
        # 幂律拟合
        try:
            popt_power, _ = curve_fit(self.power_law, x, y, maxfev=5000,
                                      p0=[y[0]-y[-1], 0.5, y[-1]])
            residuals_power = y - self.power_law(x, *popt_power)
            ss_res = np.sum(residuals_power**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2_power = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            rmse_power = np.sqrt(np.mean(residuals_power**2))
            fluctuation_power = np.mean(np.abs(residuals_power))
        except:
            r2_power = -np.inf
            rmse_power = np.inf
            fluctuation_power = np.inf
            popt_power = None

        # 指数拟合
        try:
            popt_exp, _ = curve_fit(self.exp_decay, x, y, maxfev=5000,
                                    p0=[y[0]-y[-1], 0.001, y[-1]])
            residuals_exp = y - self.exp_decay(x, *popt_exp)
            ss_res = np.sum(residuals_exp**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2_exp = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            rmse_exp = np.sqrt(np.mean(residuals_exp**2))
            fluctuation_exp = np.mean(np.abs(residuals_exp))
        except:
            r2_exp = -np.inf
            rmse_exp = np.inf
            fluctuation_exp = np.inf
            popt_exp = None

        # 选择 R² 较高的模型
        if r2_power > r2_exp and popt_power is not None:
            return {
                'model': 'power',
                'params': popt_power.tolist(),
                'r2': r2_power,
                'rmse': rmse_power,
                'fluctuation': fluctuation_power
            }
        elif r2_exp > r2_power and popt_exp is not None:
            return {
                'model': 'exp',
                'params': popt_exp.tolist(),
                'r2': r2_exp,
                'rmse': rmse_exp,
                'fluctuation': fluctuation_exp
            }
        else:
            # 都失败或相等，返回空
            return None

    # ---------- 实验运行 ----------
    def run_experiments(self, scales=None, num_individuals=3, max_iterations=10000,
                        detection_interval=100, window=100):
        """
        对指定规模和个体数量运行涌现实验，保存原始数据并计算客观指标。
        """
        if scales is None:
            scales = [51, 71, 91, 111]

        print("=" * 60)
        print("认知图客观指标分析实验（中心节点 + 节点总频次 Zipf）")
        print("=" * 60)
        print(f"规模列表: {scales}")
        print(f"个体数: {num_individuals}")
        print(f"迭代次数: {max_iterations}")
        print(f"检测间隔: {detection_interval}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)

        variation_simulator = IndividualVariation(BASE_PARAMETERS, VARIATION_RANGES)

        for scale in scales:
            print(f"\n--- 开始处理规模 {scale} 概念 ---")
            for ind in range(num_individuals):
                ind_id = f"ind{ind}"
                print(f"  个体 {ind_id} ...")

                # 生成个体参数
                base_params = variation_simulator.generate_individual(f"scale{scale}_{ind_id}")
                individual_params = create_enhanced_individual_params(base_params)

                # 创建宇宙实例
                universe = CognitiveUniverseEnhanced(individual_params,
                                                     network_seed=42+ind,
                                                     num_concepts=scale)

                # 初始化网络
                universe.initialize_semantic_network()

                # 运行演化，同时记录观测
                start_time = time.time()
                observations = universe.evolve_with_emergence_detection(
                    iterations=max_iterations,
                    detection_interval=detection_interval
                )
                elapsed = time.time() - start_time

                # 获取能耗历史
                energy_history = universe.energy_history

                # 提取压缩和迁移事件
                compressions = observations.get('natural_compressions', [])
                migrations = observations.get('natural_migrations', [])

                # --- 保存原始数据 ---
                self._save_energy_history(scale, ind, energy_history)
                self._save_compressions(scale, ind, compressions)
                self._save_migrations(scale, ind, migrations)

                # --- 指标1：能耗曲线拟合 ---
                x = np.arange(len(energy_history))
                y = np.array(energy_history)
                fit_result = self.fit_energy_curve(x, y)

                # --- 基础统计 ---
                initial_energy = energy_history[0]
                final_energy = energy_history[-1]
                energy_improvement = (initial_energy - final_energy) / initial_energy * 100

                # --- 收集压缩事件中的节点频次 ---
                for comp in compressions:
                    center = comp.get('center')
                    related = comp.get('related_nodes', [])
                    if center:
                        # 中心节点计数
                        self.center_counter[center] += 1
                        # 节点总频次：所有涉及的节点（去重）
                        nodes_in_event = set([center] + related)
                        for node in nodes_in_event:
                            self.node_total_counter[node] += 1

                # --- 个体汇总数据 ---
                entry = {
                    'scale': scale,
                    'individual': ind,
                    'initial_energy': initial_energy,
                    'final_energy': final_energy,
                    'energy_improvement': energy_improvement,
                    'total_compressions': len(compressions),
                    'total_migrations': len(migrations),
                    'elapsed_time': elapsed,
                }
                if fit_result:
                    entry.update({
                        'fit_model': fit_result['model'],
                        'fit_r2': fit_result['r2'],
                        'fit_rmse': fit_result['rmse'],
                        'fit_fluctuation': fit_result['fluctuation'],
                        'fit_params': json.dumps(fit_result['params'])  # 转为字符串存储
                    })
                else:
                    entry.update({
                        'fit_model': 'none',
                        'fit_r2': None,
                        'fit_rmse': None,
                        'fit_fluctuation': None,
                        'fit_params': None
                    })

                self.summary_data.append(entry)

                print(f"      能耗改善: {energy_improvement:.2f}%")
                print(f"      拟合模型: {entry.get('fit_model', 'none')}, R²: {entry.get('fit_r2', 0):.3f}")
                print(f"      压缩事件: {len(compressions)}")
                print(f"      迁移事件: {len(migrations)}")
                print(f"      耗时: {elapsed:.1f}s")

                # --- 绘制个体能耗下降速率图（可选，简化版）---
                self.plot_energy_rate(scale, ind, energy_history, fit_result, window)

        # 保存汇总结果
        self._save_summary()

        # --- 指标2：中心节点出现次数的 Zipf 分析 ---
        self.zipf_center_analysis()

        # --- 指标3：节点总频次的 Zipf 分析 ---
        self.zipf_node_total_analysis()

    def plot_energy_rate(self, scale, ind, energy_history, fit_result, window=100):
        """绘制能耗下降速率图，并叠加拟合曲线的理论速率（虚线）"""
        rates = []
        indices = []
        for i in range(0, len(energy_history) - window, window):
            rates.append(energy_history[i] - energy_history[i + window])
            indices.append(i)

        plt.figure(figsize=(8, 5))
        # 实际速率曲线
        plt.plot(indices, rates, 'b-', linewidth=1.5, label='实际下降速率')

        # 如果有拟合结果，添加理论速率虚线
        if fit_result and fit_result['model'] != 'none':
            model = fit_result['model']
            params = fit_result['params']
            # 计算理论能量值（在每个时间点）
            x_full = np.arange(len(energy_history))
            if model == 'power':
                y_fit = self.power_law(x_full, *params)
            else:  # exp
                y_fit = self.exp_decay(x_full, *params)
            # 计算理论速率（同样按window步长）
            fit_rates = []
            for i in indices:
                if i + window < len(y_fit):
                    fit_rates.append(y_fit[i] - y_fit[i + window])
                else:
                    fit_rates.append(0)
            plt.plot(indices, fit_rates, 'r--', linewidth=1.5, label=f'拟合理论速率 ({model})')

        plt.xlabel('迭代次数')
        plt.ylabel('能耗下降速率')
        plt.title(f'规模 {scale} 个体 {ind}：能耗下降速率与拟合曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = self.fig_dir / f"energy_rate_scale{scale}_ind{ind}_{self.timestamp}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

    def zipf_center_analysis(self):
        """
        对中心节点出现次数进行幂律拟合（Zipf 检验），绘制 log-log 图。
        """
        if not self.center_counter:
            print("警告：没有压缩事件，无法进行中心节点 Zipf 分析")
            return

        # 按出现次数排序
        items = sorted(self.center_counter.items(), key=lambda x: x[1], reverse=True)
        centers = [item[0] for item in items]
        freqs = [item[1] for item in items]

        # 转换为对数坐标
        log_ranks = np.log(np.arange(1, len(freqs)+1))
        log_freqs = np.log(freqs)

        # 线性拟合（幂律）
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
        r2 = r_value**2

        print("\n" + "=" * 60)
        print("指标2：中心节点出现次数的 Zipf 分布检验")
        print("=" * 60)
        print(f"幂律指数（斜率）: {slope:.3f}")
        print(f"拟合优度 R²: {r2:.3f}")
        print(f"p 值: {p_value:.3e}")
        print(f"总压缩事件数: {sum(freqs)}")
        print(f"不同中心节点数: {len(centers)}")
        print("=" * 60)

        # 绘制 log-log 图
        plt.figure(figsize=(8, 6))
        plt.scatter(log_ranks, log_freqs, alpha=0.7, label='Observed data')
        x_fit = np.linspace(min(log_ranks), max(log_ranks), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, 'r--', label=f'Fitted line (slope={slope:.2f})')
        plt.xlabel('log(rank)')
        plt.ylabel('log(frequency of center nodes)')
        plt.title('Power-law test for center node frequencies (Zipf)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = self.fig_dir / f"zipf_center_{self.timestamp}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"中心节点 Zipf 图已保存: {fig_path}")

        # 保存 Zipf 结果到 JSON
        zipf_result = {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'p_value': p_value,
            'total_events': sum(freqs),
            'unique_centers': len(centers),
            'top_centers': {centers[i]: freqs[i] for i in range(min(10, len(centers)))}
        }
        with open(self.output_dir / f"zipf_center_result_{self.timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(zipf_result, f, ensure_ascii=False, indent=2)

    def zipf_node_total_analysis(self):
        """
        对节点在压缩事件中出现的总次数（中心+成员，去重）进行幂律拟合（Zipf 检验），绘制 log-log 图。
        """
        if not self.node_total_counter:
            print("警告：没有压缩事件，无法进行节点总频次 Zipf 分析")
            return

        # 按出现次数排序
        items = sorted(self.node_total_counter.items(), key=lambda x: x[1], reverse=True)
        nodes = [item[0] for item in items]
        freqs = [item[1] for item in items]

        # 转换为对数坐标
        log_ranks = np.log(np.arange(1, len(freqs)+1))
        log_freqs = np.log(freqs)

        # 线性拟合（幂律）
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
        r2 = r_value**2

        print("\n" + "=" * 60)
        print("指标3：节点在压缩事件中出现总次数的 Zipf 分布检验")
        print("=" * 60)
        print(f"幂律指数（斜率）: {slope:.3f}")
        print(f"拟合优度 R²: {r2:.3f}")
        print(f"p 值: {p_value:.3e}")
        print(f"总出现次数: {sum(freqs)}")
        print(f"不同节点数: {len(nodes)}")
        print("=" * 60)

        # 绘制 log-log 图
        plt.figure(figsize=(8, 6))
        plt.scatter(log_ranks, log_freqs, alpha=0.7, label='Observed data')
        x_fit = np.linspace(min(log_ranks), max(log_ranks), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, 'r--', label=f'Fitted line (slope={slope:.2f})')
        plt.xlabel('log(rank)')
        plt.ylabel('log(total frequency of nodes)')
        plt.title('Power-law test for total node frequencies in compressions (Zipf)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = self.fig_dir / f"zipf_node_total_{self.timestamp}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"节点总频次 Zipf 图已保存: {fig_path}")

        # 保存 Zipf 结果到 JSON
        zipf_result = {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'p_value': p_value,
            'total_occurrences': sum(freqs),
            'unique_nodes': len(nodes),
            'top_nodes': {nodes[i]: freqs[i] for i in range(min(10, len(nodes)))}
        }
        with open(self.output_dir / f"zipf_node_total_result_{self.timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(zipf_result, f, ensure_ascii=False, indent=2)

    # ---------- 数据保存辅助函数 ----------
    def _save_energy_history(self, scale, ind, energy_history):
        filename = self.output_dir / f"energy_history_scale{scale}_ind{ind}_{self.timestamp}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'energy'])
            for it, e in enumerate(energy_history):
                writer.writerow([it, e])

    def _save_compressions(self, scale, ind, compressions):
        if not compressions:
            return
        filename = self.output_dir / f"compressions_scale{scale}_ind{ind}_{self.timestamp}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['detection_iteration', 'center', 'related_nodes', 'cluster_size',
                          'energy_synergy', 'cohesion', 'emergence_strength']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in compressions:
                row = {k: c.get(k, '') for k in fieldnames}
                if 'related_nodes' in row and isinstance(row['related_nodes'], list):
                    row['related_nodes'] = '|'.join(row['related_nodes'])
                writer.writerow(row)

    def _save_migrations(self, scale, ind, migrations):
        if not migrations:
            return
        filename = self.output_dir / f"migrations_scale{scale}_ind{ind}_{self.timestamp}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['detection_iteration', 'principle_node', 'from_node', 'to_node',
                          'efficiency_gain', 'domain_span', 'path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in migrations:
                row = {k: m.get(k, '') for k in fieldnames}
                if 'path' in row and isinstance(row['path'], list):
                    row['path'] = ' -> '.join(row['path'])
                writer.writerow(row)

    def _save_summary(self):
        """保存个体汇总数据到 CSV 和 JSON"""
        df = pd.DataFrame(self.summary_data)
        csv_path = self.output_dir / f"individual_summary_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        json_path = self.output_dir / f"individual_summary_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary_data, f, ensure_ascii=False, indent=2)

        print(f"\n汇总文件已保存:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")

        # 打印按规模的平均拟合指标
        print("\n按规模平均拟合指标:")
        grouped = df.groupby('scale').agg({
            'energy_improvement': 'mean',
            'fit_r2': 'mean',
            'fit_rmse': 'mean',
            'fit_fluctuation': 'mean',
            'total_compressions': 'mean',
            'total_migrations': 'mean'
        }).round(3)
        print(grouped.to_string())


if __name__ == "__main__":
    analyzer = ObjectiveMetricsAnalysis()
    analyzer.run_experiments(
        scales=[51, 71, 91, 111],
        num_individuals=3,
        max_iterations=10000,
        detection_interval=100,
        window=100
    )
    print("\n所有实验完成，数据已保存。")