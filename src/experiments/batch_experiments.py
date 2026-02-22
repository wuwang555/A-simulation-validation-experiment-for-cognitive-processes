"""
batch_experiments.py
认知图模型批处理实验脚本
用于系统性地运行不同概念规模下的所有模型对比实验
"""

import sys
import os
import time
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
# 导入实验管理器
try:
    from main import CognitiveGraphExperimentManager
    print("✅ 成功导入实验管理器")
except ImportError as e:
    print(f"❌ 导入实验管理器失败: {e}")
    sys.exit(1)


class BatchExperimentRunner:
    """批处理实验运行器，管理多规模多模型的对比实验。

    该类负责配置、运行和保存所有模型的实验数据，并生成对比图表。

    Attributes:
        manager (CognitiveGraphExperimentManager): 实验管理器实例。
        output_dir (Path): 结果输出目录。
        config (dict): 实验配置参数。
        results (list): 所有实验结果的列表。
        summary (dict): 按规模汇总的实验结果。
    """

    def __init__(self, output_dir="../../results/batch_experiments"):
        """初始化批处理运行器。

        Args:
            output_dir (str): 结果保存目录，默认为 "../../results/batch_experiments"。
        """
        self.manager = CognitiveGraphExperimentManager()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 实验配置
        self.config = {
            "iterations": 10000,  # 每次迭代次数
            "repetitions": 1,  # 每个配置重复次数（减少随机性）
            "models": ["random", "qlearning", "traditional", "emergence"],
            "scales": [51, 71, 91, 111],  # 概念规模
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        # 结果存储
        self.results = []
        self.summary = {}

    def run_single_experiment(self, model_type, scale):
        """运行单个实验。

        Args:
            model_type (str): 模型类型，可选 "random", "qlearning", "traditional", "emergence"。
            scale (int): 概念规模（节点数）。

        Returns:
            dict or None: 实验结果的指标字典，失败返回None。
        """
        print(f"\n{'=' * 60}")
        print(f"运行实验: {model_type.upper()}模型 | 概念规模: {scale}")
        print('=' * 60)

        start_time = time.time()

        try:
            if model_type == "random":
                result = self.manager.run_random_network_model(
                    num_nodes=scale,
                    max_iterations=self.config["iterations"]
                )
            elif model_type == "qlearning":
                result = self.manager.run_qlearning_model(
                    num_nodes=scale,
                    max_iterations=self.config["iterations"]
                )
            elif model_type == "traditional":
                result = self.manager.run_preset_algorithm_model(
                    num_concepts=scale
                )
                # 处理列表格式的结果
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
            elif model_type == "emergence":
                result = self.manager.run_natural_emergence_model(
                    num_individuals=1,  # 为了速度，只运行1个个体
                    max_iterations=self.config["iterations"],
                    num_concepts=scale
                )
                # 处理列表格式的结果
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
            else:
                print(f"❌ 未知模型类型: {model_type}")
                return None

            elapsed_time = time.time() - start_time

            # 提取关键指标
            metrics = self._extract_metrics(model_type, result, scale, elapsed_time)

            print(f"✅ 实验完成 | 耗时: {elapsed_time:.1f}秒 | 能耗改善: {metrics.get('improvement', 'N/A')}%")

            return metrics

        except Exception as e:
            print(f"❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_metrics(self, model_type, result, scale, elapsed_time):
        """从结果中提取关键指标。

        Args:
            model_type (str): 模型类型。
            result (dict): 实验返回的结果字典。
            scale (int): 概念规模。
            elapsed_time (float): 运行耗时（秒）。

        Returns:
            dict: 提取的指标字典。
        """
        metrics = {
            "model": model_type,
            "scale": scale,
            "elapsed_time": elapsed_time,
            "iterations": self.config["iterations"]
        }

        if not result:
            return metrics

        # 通用指标提取
        if isinstance(result, dict):
            # 能耗改善指标
            if 'improvement' in result:
                metrics["improvement"] = result['improvement']
            elif 'energy_improvement' in result:
                metrics["improvement"] = result['energy_improvement']
            elif 'improvement_percent' in result:
                metrics["improvement"] = result['improvement_percent']

            # 概念压缩
            if 'compression_centers' in result:
                metrics["compression_centers"] = result['compression_centers']
            elif 'compression_count' in result:
                metrics["compression_centers"] = result['compression_count']

            # 原理迁移
            if 'migration_bridges' in result:
                metrics["migration_bridges"] = result['migration_bridges']
            elif 'migration_count' in result:
                metrics["migration_bridges"] = result['migration_count']

            # 网络统计
            if 'network_stats' in result:
                stats = result['network_stats']
                metrics.update({
                    "num_nodes": stats.get('num_nodes', scale),
                    "num_edges": stats.get('num_edges', 0),
                    "avg_energy": stats.get('avg_energy', 0)
                })

        # 特定模型指标
        if model_type == "qlearning" and isinstance(result, dict):
            if 'q_table_stats' in result:
                q_stats = result['q_table_stats']
                metrics.update({
                    "q_table_sparsity": q_stats.get('sparsity', 0),
                    "q_table_non_zero": q_stats.get('non_zero_entries', 0)
                })
        elif model_type == "traditional" and isinstance(result, dict):
            # 传统模型可能有认知状态统计
            if 'state_stats' in result:
                state_stats = result['state_stats']
                metrics["exploration_ratio"] = state_stats.get('exploration', 0)
                metrics["inspiration_ratio"] = state_stats.get('inspiration', 0)
        elif model_type == "emergence" and isinstance(result, dict):
            # 涌现模型特有指标
            metrics.update({
                "compression_frequency": result.get('compression_frequency', 0),
                "migration_frequency": result.get('migration_frequency', 0)
            })

        return metrics

    def run_full_batch(self):
        """运行完整批处理实验（所有规模 × 所有模型）。"""
        print("\n" + "=" * 80)
        print("开始运行完整批处理实验")
        print(f"配置: {self.config['scales']}个规模 × {self.config['models']}个模型")
        print(f"迭代次数: {self.config['iterations']}")
        print("=" * 80)

        total_experiments = len(self.config["scales"]) * len(self.config["models"])
        completed = 0

        for scale in self.config["scales"]:
            scale_results = {}

            for model in self.config["models"]:
                # 运行实验
                metrics = self.run_single_experiment(model, scale)

                if metrics:
                    # 保存结果
                    self.results.append(metrics)
                    scale_results[model] = metrics

                    # 更新进度
                    completed += 1
                    progress = (completed / total_experiments) * 100
                    print(f"\n📊 进度: {completed}/{total_experiments} ({progress:.1f}%)")

            # 保存该规模的结果摘要
            self.summary[scale] = scale_results

        # 保存所有结果
        self.save_results()

        print("\n" + "=" * 80)
        print("🎉 所有批处理实验完成!")
        print("=" * 80)

        # 显示摘要
        self.display_summary()

    def save_results(self):
        """保存实验结果到CSV和JSON文件。"""
        timestamp = self.config["timestamp"]

        # 保存详细结果到CSV
        df_results = pd.DataFrame(self.results)
        csv_path = self.output_dir / f"detailed_results_{timestamp}.csv"
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 保存摘要到JSON
        summary_path = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            # 转换不能序列化的对象
            json_ready = {}
            for scale, models in self.summary.items():
                json_ready[str(scale)] = {}
                for model, metrics in models.items():
                    json_ready[str(scale)][model] = {
                        k: (float(v) if isinstance(v, (int, float)) else str(v))
                        for k, v in metrics.items()
                    }

            json.dump(json_ready, f, ensure_ascii=False, indent=2)

        # 保存配置
        config_path = self.output_dir / f"config_{timestamp}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        print(f"\n📁 结果已保存:")
        print(f"  详细结果: {csv_path}")
        print(f"  实验摘要: {summary_path}")
        print(f"  配置文件: {config_path}")

        return csv_path, summary_path

    def display_summary(self):
        """在控制台显示实验结果摘要表格。"""
        print("\n" + "=" * 80)
        print("实验结果摘要")
        print("=" * 80)

        # 创建摘要表格
        summary_data = []

        for scale in self.config["scales"]:
            scale_row = {"概念规模": scale}

            if scale in self.summary:
                for model in self.config["models"]:
                    if model in self.summary[scale]:
                        metrics = self.summary[scale][model]
                        improvement = metrics.get('improvement', 0)

                        # 添加性能指标
                        scale_row[f"{model}能耗改善(%)"] = f"{improvement:.1f}%" if isinstance(improvement, (int, float)) else improvement

                        # 添加特殊指标
                        if model == "emergence":
                            compression = metrics.get('compression_centers', 0)
                            migration = metrics.get('migration_bridges', 0)
                            scale_row["涌现压缩"] = compression
                            scale_row["涌现迁移"] = migration

            summary_data.append(scale_row)

        # 显示表格
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))

        # 计算统计数据
        print("\n" + "=" * 80)
        print("关键统计数据")
        print("=" * 80)

        # 1. 各模型在不同规模下的平均性能
        print("\n📈 各模型性能表现:")
        for model in self.config["models"]:
            improvements = []
            for scale in self.config["scales"]:
                if scale in self.summary and model in self.summary[scale]:
                    imp = self.summary[scale][model].get('improvement', 0)
                    if isinstance(imp, (int, float)):
                        improvements.append(imp)

            if improvements:
                avg_imp = sum(improvements) / len(improvements)
                print(
                    f"  {model:15s}: 平均改善 {avg_imp:.1f}% (范围: {min(improvements):.1f}%-{max(improvements):.1f}%)")

        # 2. 规模效应分析
        print("\n📊 规模效应分析:")
        for model in ["traditional", "emergence"]:
            improvements_by_scale = {}
            for scale in self.config["scales"]:
                if scale in self.summary and model in self.summary[scale]:
                    imp = self.summary[scale][model].get('improvement', 0)
                    if isinstance(imp, (int, float)):
                        improvements_by_scale[scale] = imp

            if improvements_by_scale:
                print(f"  {model}模型:")
                for scale, imp in sorted(improvements_by_scale.items()):
                    print(f"    {scale}概念: {imp:.1f}%")

    def create_comparison_charts(self):
        """创建性能对比和规模效应图表（使用matplotlib）。"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            print("\n📊 正在生成对比图表...")

            # 准备数据
            scales = self.config["scales"]
            models = self.config["models"]

            # 性能对比柱状图
            fig1, ax1 = plt.subplots(figsize=(12, 8))

            bar_width = 0.2
            x = np.arange(len(scales))

            model_colors = {
                "random": "#999999",
                "qlearning": "#4C72B0",
                "traditional": "#55A868",
                "emergence": "#C44E52"
            }

            model_labels = {
                "random": "随机网络",
                "qlearning": "增强Q-learning",
                "traditional": "传统机制设计",
                "emergence": "自然涌现"
            }

            for i, model in enumerate(models):
                improvements = []
                for scale in scales:
                    if scale in self.summary and model in self.summary[scale]:
                        imp = self.summary[scale][model].get('improvement', 0)
                        improvements.append(float(imp) if isinstance(imp, (int, float)) else 0)
                    else:
                        improvements.append(0)

                ax1.bar(x + i * bar_width, improvements, bar_width,
                        label=model_labels[model], color=model_colors[model], alpha=0.8)

            ax1.set_xlabel('概念规模', fontsize=12)
            ax1.set_ylabel('能耗改善 (%)', fontsize=12)
            ax1.set_title('不同模型在不同概念规模下的性能对比', fontsize=14, fontweight='bold')
            ax1.set_xticks(x + bar_width * 1.5)
            ax1.set_xticklabels([f"{s}概念" for s in scales])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            plt.tight_layout()
            chart_path1 = self.output_dir / f"performance_comparison_{self.config['timestamp']}.png"
            plt.savefig(chart_path1, dpi=300)
            print(f"✅ 性能对比图已保存: {chart_path1}")

            # 规模效应折线图
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            for model in ["traditional", "emergence"]:
                scales_list = []
                improvements_list = []

                for scale in scales:
                    if scale in self.summary and model in self.summary[scale]:
                        imp = self.summary[scale][model].get('improvement', 0)
                        if isinstance(imp, (int, float)):
                            scales_list.append(scale)
                            improvements_list.append(imp)

                if scales_list and improvements_list:
                    ax2.plot(scales_list, improvements_list,
                             marker='o', linewidth=2, markersize=8,
                             label=model_labels[model])

            ax2.set_xlabel('概念规模', fontsize=12)
            ax2.set_ylabel('能耗改善 (%)', fontsize=12)
            ax2.set_title('认知模型规模效应分析', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            chart_path2 = self.output_dir / f"scale_effect_{self.config['timestamp']}.png"
            plt.savefig(chart_path2, dpi=300)
            print(f"✅ 规模效应图已保存: {chart_path2}")

            plt.show()

        except ImportError:
            print("⚠️  Matplotlib未安装，跳过图表生成")
            print("   请运行: pip install matplotlib")
        except Exception as e:
            print(f"❌ 生成图表时出错: {e}")


def run_specific_combination():
    """运行特定组合的实验（调试用）。"""
    runner = BatchExperimentRunner()

    # 测试单个组合
    print("测试单个实验组合...")
    metrics = runner.run_single_experiment("emergence", 91)

    if metrics:
        print("\n测试结果:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


def main():
    """主函数，提供交互式菜单选择运行模式。"""
    print("\n" + "=" * 80)
    print("认知图模型批处理实验平台")
    print("=" * 80)

    print("\n选择运行模式:")
    print("1. 运行完整批处理实验 (所有规模 × 所有模型)")
    print("2. 运行特定规模实验")
    print("3. 运行特定模型实验")
    print("4. 只运行涌现模型多规模实验")
    print("5. 测试单个实验组合")

    choice = input("\n请选择模式 (1-5): ").strip()

    runner = BatchExperimentRunner()

    if choice == "1":
        # 完整批处理
        runner.run_full_batch()
        runner.create_comparison_charts()

    elif choice == "2":
        # 特定规模
        print("\n选择概念规模:")
        print("可用规模: 51, 71, 91, 111")
        scale_input = input("输入规模（用逗号分隔，如 51,91）: ").strip()

        try:
            scales = [int(s.strip()) for s in scale_input.split(",")]
            runner.config["scales"] = scales
            runner.run_full_batch()
            runner.create_comparison_charts()
        except ValueError:
            print("❌ 输入无效，请使用数字")

    elif choice == "3":
        # 特定模型
        print("\n选择模型:")
        print("1. 随机网络")
        print("2. 增强Q-learning")
        print("3. 传统机制设计")
        print("4. 自然涌现")
        print("5. 所有模型")

        model_choice = input("请选择 (1-5): ").strip()

        model_map = {
            "1": ["random"],
            "2": ["qlearning"],
            "3": ["traditional"],
            "4": ["emergence"],
            "5": ["random", "qlearning", "traditional", "emergence"]
        }

        if model_choice in model_map:
            runner.config["models"] = model_map[model_choice]
            runner.run_full_batch()
            runner.create_comparison_charts()
        else:
            print("❌ 选择无效")

    elif choice == "4":
        # 只运行涌现模型多规模
        runner.config["models"] = ["emergence"]
        runner.run_full_batch()
        runner.create_comparison_charts()

    elif choice == "5":
        # 测试单个组合
        run_specific_combination()

    else:
        print("⚠️ 无效选择，运行完整批处理")
        runner.run_full_batch()
        runner.create_comparison_charts()

    print("\n" + "=" * 80)
    print("批处理实验完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()