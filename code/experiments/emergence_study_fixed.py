# emergence_study_fixed.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

from emergence.universe import CognitiveUniverse
from emergence.observer import EmergenceObserver
from emergence.detector import EmergenceDetector
from emergence.metrics import NaturalEmergenceMetrics
from utils.individual_variation import IndividualVariation, create_enhanced_individual_params
from config import *


class EmergenceStudyFixed:
    """修复后的涌现现象研究实验"""

    def __init__(self):
        self.results = {}
        self.comparison_data = {}

    def run_pure_emergence_experiment(self, num_individuals=3, max_iterations=None):
        """运行纯粹的涌现观察实验 - 修复版本"""
        if max_iterations is None:
            max_iterations = EXPERIMENT_CONFIG['default_iterations']

        print("=== 纯粹涌现观察实验 ===")
        print("目标：观察从两个公设中自然涌现的认知现象")
        print(f"配置：{num_individuals}个个体，{max_iterations}次迭代")
        print("=" * 50)

        variation_simulator = IndividualVariation(BASE_PARAMETERS, VARIATION_RANGES)
        emergence_results = []

        for i in range(num_individuals):
            individual_id = f"涌现个体_{i + 1}"
            print(f"\n--- 观察 {individual_id} ---")

            # 生成个体参数
            base_params = variation_simulator.generate_individual(individual_id)
            individual_params = create_enhanced_individual_params(base_params)

            # 创建纯粹能量宇宙
            universe = CognitiveUniverse(individual_params)
            observer = EmergenceObserver()
            detector = EmergenceDetector()

            # 初始化宇宙网络
            self._initialize_universe_network(universe)

            # 运行宇宙演化
            start_time = time.time()
            observations = universe.evolve(iterations=max_iterations)
            end_time = time.time()

            # 收集个体结果
            individual_result = {
                'individual_id': individual_id,
                'parameters': individual_params,
                'observations': universe.observations,
                'final_energy': universe.calculate_network_energy(),
                'initial_energy': universe.energy_history[0] if universe.energy_history else 1.0,
                'energy_improvement': self._calculate_energy_improvement(universe.energy_history),
                'compression_count': len(universe.observations['spontaneous_compressions']),
                'migration_count': len(universe.observations['emergent_migrations']),
                'computation_time': end_time - start_time,
                'universe': universe  # 保存universe对象以便后续使用
            }

            emergence_results.append(individual_result)

            print(f"{individual_id} 完成:")
            print(f"  计算时间: {individual_result['computation_time']:.1f}秒")
            print(f"  能耗改善: {individual_result['energy_improvement']:.1f}%")
            print(f"  观察到压缩: {individual_result['compression_count']}次")
            print(f"  观察到迁移: {individual_result['migration_count']}次")

        self.results['pure_emergence'] = emergence_results
        self._analyze_emergence_results(emergence_results)

        return emergence_results

    def _initialize_universe_network(self, universe):
        """初始化宇宙网络 - 简化版本"""
        try:
            # 使用内置的语义网络初始化
            universe.initialize_semantic_network()
        except Exception as e:
            print(f"语义网络初始化失败: {e}，使用测试网络")
            self._create_test_network(universe)

    def _create_test_network(self, universe):
        """创建测试网络"""
        test_nodes = ["算法", "数据结构", "优化", "递归", "迭代", "抽象", "模式识别",
                      "能量", "学习", "记忆", "思考", "创造", "理解", "应用"]
        universe.G.add_nodes_from(test_nodes)

        import random
        # 创建更有意义的连接
        connections = [
            ("算法", "数据结构"), ("算法", "优化"), ("递归", "迭代"),
            ("抽象", "模式识别"), ("学习", "记忆"), ("思考", "创造"),
            ("理解", "应用"), ("能量", "优化"), ("算法", "递归")
        ]

        for u, v in connections:
            energy = random.uniform(0.5, 1.5)
            universe.G.add_edge(u, v, weight=energy)

        # 添加一些随机连接
        for i in range(10):
            u, v = random.sample(test_nodes, 2)
            if not universe.G.has_edge(u, v):
                energy = random.uniform(0.8, 2.0)
                universe.G.add_edge(u, v, weight=energy)

        print(f"测试网络: {len(test_nodes)}个节点, {universe.G.number_of_edges()}条边")

    def _calculate_energy_improvement(self, energy_history):
        """计算能耗改善百分比"""
        if len(energy_history) < 2:
            return 0.0
        initial = energy_history[0]
        final = energy_history[-1]
        if initial == 0:
            return 0.0
        return ((initial - final) / initial) * 100

    # 其他方法保持不变...
    def _analyze_emergence_results(self, results):
        """分析涌现实验结果"""
        print("\n" + "=" * 50)
        print("自然涌现实验结果分析")
        print("=" * 50)

        # 基本统计
        improvements = [r['energy_improvement'] for r in results]
        compressions = [r['compression_count'] for r in results]
        migrations = [r['migration_count'] for r in results]

        print(f"能耗改善统计:")
        print(f"  平均: {np.mean(improvements):.1f}%")
        print(f"  标准差: {np.std(improvements):.1f}%")
        print(f"  范围: {min(improvements):.1f}% - {max(improvements):.1f}%")

        print(f"概念压缩涌现:")
        print(f"  平均: {np.mean(compressions):.1f}次")
        print(f"  总次数: {sum(compressions)}次")

        print(f"原理迁移涌现:")
        print(f"  平均: {np.mean(migrations):.1f}次")
        print(f"  总次数: {sum(migrations)}次")

    def visualize_emergence_results(self):
        """可视化涌现实验结果"""
        if 'pure_emergence' not in self.results:
            print("请先运行实验!")
            return

        results = self.results['pure_emergence']

        # 创建对比图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 能耗改善对比
        improvements = [r['energy_improvement'] for r in results]
        individuals = [r['individual_id'] for r in results]

        bars = ax1.bar(individuals, improvements, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('自然涌现组能耗改善对比')
        ax1.set_ylabel('能耗改善 (%)')
        ax1.set_xlabel('个体')

        # 在柱状图上添加数值标签
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{improvement:.1f}%', ha='center', va='bottom')

        # 2. 涌现现象数量
        compressions = [r['compression_count'] for r in results]
        migrations = [r['migration_count'] for r in results]

        x = np.arange(len(individuals))
        width = 0.35

        ax2.bar(x - width / 2, compressions, width, label='概念压缩', color='orange')
        ax2.bar(x + width / 2, migrations, width, label='原理迁移', color='purple')
        ax2.set_title('涌现现象数量对比')
        ax2.set_ylabel('现象数量')
        ax2.set_xlabel('个体')
        ax2.set_xticks(x)
        ax2.set_xticklabels(individuals)
        ax2.legend()

        # 3. 能量收敛曲线示例
        if results and len(results) > 0 and hasattr(results[0]['universe'], 'energy_history'):
            sample_energy_history = results[0]['universe'].energy_history
            ax3.plot(sample_energy_history, 'b-', alpha=0.7)
            ax3.set_title('典型能量收敛过程')
            ax3.set_xlabel('迭代次数')
            ax3.set_ylabel('平均认知能耗')
            ax3.grid(True, alpha=0.3)

        # 4. 网络统计
        if results and len(results) > 0:
            node_counts = [r['universe'].G.number_of_nodes() for r in results]
            edge_counts = [r['universe'].G.number_of_edges() for r in results]

            ax4.bar(individuals, node_counts, alpha=0.6, label='节点数')
            ax4.bar(individuals, edge_counts, alpha=0.6, label='边数')
            ax4.set_title('网络规模统计')
            ax4.set_ylabel('数量')
            ax4.legend()

        plt.tight_layout()
        plt.show()



def main_fixed():
    """修复版本的主函数"""
    study = EmergenceStudyFixed()

    # 运行纯粹涌现实验
    emergence_results = study.run_pure_emergence_experiment(
        num_individuals=2,
        max_iterations=3000  # 减少迭代次数以便快速测试
    )

    # 可视化结果
    study.visualize_emergence_results()

    return study


if __name__ == "__main__":
    main_fixed()