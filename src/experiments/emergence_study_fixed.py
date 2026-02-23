# emergence_study_fixed.py
"""
涌现现象研究实验（修复版本）
用于观察从两个公设（认知时空和能量动力学）中自然涌现的概念压缩和原理迁移。
"""

import numpy as np
import time
import pandas as pd
from datetime import datetime
from emergence.universe_enhanced import CognitiveUniverseEnhanced
from emergence.observer import EmergenceObserver
from emergence.detector_fixed import EmergenceDetectorFixed
from utils.individual_variation import IndividualVariation, create_enhanced_individual_params
from config import *
import os
import csv
import networkx as nx


class EmergenceStudyFixed:
    """修复后的涌现现象研究实验类。

    该类负责运行纯粹的涌现实验，记录概念压缩和原理迁移事件，
    并保存结果到Excel文件。
    """

    def __init__(self):
        """初始化研究实例，创建结果容器和统一时间戳。"""
        self.results = {}
        self.comparison_data = {}
        self.excel_data = {'compressions': [], 'migrations': []}
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # 统一时间戳

    def save_to_excel(self, filename=None):
        """保存涌现数据到Excel文件。

        Args:
            filename (str, optional): 保存的文件名。如果为None，自动生成。

        Returns:
            str: 保存的文件路径。
        """
        if filename is None:
            # 修改：确保目录存在
            emergence_dir = "results/emergence"
            os.makedirs(emergence_dir, exist_ok=True)

            filename = os.path.join(emergence_dir, f"emergence_results_{self.timestamp}.xlsx")

        # 创建DataFrame
        df_compressions = pd.DataFrame(self.excel_data['compressions'])
        df_migrations = pd.DataFrame(self.excel_data['migrations'])

        # 保存到Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_compressions.to_excel(writer, sheet_name='概念压缩', index=False)
            df_migrations.to_excel(writer, sheet_name='第一性原理迁移', index=False)

        print(f"数据已保存到: {filename}")
        print(f"概念压缩记录: {len(df_compressions)} 条")
        print(f"原理迁移记录: {len(df_migrations)} 条")

        return filename

    def run_pure_emergence_experiment(self, num_individuals=3, max_iterations=None, num_concepts=None):
        """运行纯粹的涌现观察实验。

        基于两个公设（认知时空和能量动力学）运行宇宙演化，观察自然涌现的概念压缩和原理迁移。

        Args:
            num_individuals (int): 个体数量。
            max_iterations (int, optional): 每个个体的最大迭代次数。
            num_concepts (int, optional): 概念节点数量。如果为None，使用默认值。

        Returns:
            list: 每个个体的结果字典列表。
        """
        if max_iterations is None:
            max_iterations = EXPERIMENT_CONFIG['default_iterations']

        print("=== 纯粹涌现观察实验 ===")
        print("目标：观察从两个公设中自然涌现的认知现象")
        print(f"配置：{num_individuals}个个体，{max_iterations}次迭代")
        if num_concepts:
            print(f"概念数量：{num_concepts}")
        print("=" * 50)

        variation_simulator = IndividualVariation(BASE_PARAMETERS, VARIATION_RANGES)
        emergence_results = []

        for i in range(num_individuals):
            individual_id = f"涌现个体_{i + 1}"
            print(f"\n--- 观察 {individual_id} ---")

            # 生成个体参数
            base_params = variation_simulator.generate_individual(individual_id)
            individual_params = create_enhanced_individual_params(base_params)

            # 直接创建增强宇宙实例，传入num_concepts参数
            universe_enhanced = CognitiveUniverseEnhanced(individual_params, num_concepts=num_concepts)
            observer = EmergenceObserver()
            detector = EmergenceDetectorFixed()

            # 初始化宇宙网络
            self._initialize_universe_network(universe_enhanced)

            # 运行宇宙演化
            start_time = time.time()
            observations = universe_enhanced.evolve_with_emergence_detection(
                iterations=max_iterations,
                detection_interval=100
            )
            end_time = time.time()

            # 保存能量历史
            energy_history = universe_enhanced.energy_history
            energy_file = os.path.join("results/emergence", f"energy_history_{individual_id}_{self.timestamp}.csv")
            with open(energy_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'energy'])
                for idx, e in enumerate(energy_history):
                    writer.writerow([idx, e])
            print(f"能量历史已保存: {energy_file}")

            # 可选：保存最终网络结构（GraphML）
            graphml_file = os.path.join("results/emergence", f"network_{individual_id}_{self.timestamp}.graphml")
            nx.write_graphml(universe_enhanced.G, graphml_file)
            print(f"网络结构已保存: {graphml_file}")

            # 记录到Excel数据结构
            self._record_excel_data(individual_id, observations, universe_enhanced)

            # 收集个体结果
            individual_result = {
                'individual_id': individual_id,
                'parameters': individual_params,
                'observations': observations,  # 使用返回的observations
                'final_energy': universe_enhanced.calculate_network_energy(),
                'initial_energy': universe_enhanced.energy_history[0] if universe_enhanced.energy_history else 1.0,
                'energy_improvement': self._calculate_energy_improvement(universe_enhanced.energy_history),
                'compression_count': len(observations['natural_compressions']),
                'migration_count': len(observations['natural_migrations']),
                'computation_time': end_time - start_time,
                'universe': universe_enhanced  # 保存universe对象以便后续使用
            }

            emergence_results.append(individual_result)

            print(f"{individual_id} 完成:")
            print(f"  计算时间: {individual_result['computation_time']:.1f}秒")
            print(f"  能耗改善: {individual_result['energy_improvement']:.1f}%")
            print(f"  观察到压缩: {individual_result['compression_count']}次")
            print(f"  观察到迁移: {individual_result['migration_count']}次")

        self.results['pure_emergence'] = emergence_results
        self._analyze_emergence_results(emergence_results)
        excel_file = self.save_to_excel()
        return emergence_results

    def _record_excel_data(self, individual_id, observations, universe):
        """记录压缩和迁移数据到excel_data容器。

        Args:
            individual_id (str): 个体标识。
            observations (dict): 观察到的涌现现象字典。
            universe (CognitiveUniverseEnhanced): 宇宙对象。
        """
        # 记录概念压缩
        for compression in observations['natural_compressions']:
            self.excel_data['compressions'].append({
                '个体ID': individual_id,
                '中心节点': compression['center'],
                '相关节点数': len(compression['related_nodes']),
                '相关节点': ', '.join(compression['related_nodes']),
                '能量协同性': compression.get('energy_synergy', 0),
                '集群内聚性': compression.get('cohesion', 0),
                '涌现强度': compression.get('emergence_strength', 0),
                '检测迭代': compression.get('detection_iteration', 0),
                '当前网络能耗': universe.calculate_network_energy(),
                '时间戳': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # 记录原理迁移
        for migration in observations['natural_migrations']:
            self.excel_data['migrations'].append({
                '个体ID': individual_id,
                '原理节点': migration['principle_node'],
                '起始节点': migration['from_node'],
                '目标节点': migration['to_node'],
                '迁移路径': ' -> '.join(migration.get('path', [])),
                '效率增益': migration.get('efficiency_gain', 0),
                '领域跨度': migration.get('domain_span', 0),
                '检测迭代': migration.get('detection_iteration', 0),
                '当前网络能耗': universe.calculate_network_energy(),
                '时间戳': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    def _initialize_universe_network(self, universe):
        """初始化宇宙网络。

        尝试使用语义网络初始化，失败时回退到测试网络。

        Args:
            universe (CognitiveUniverseEnhanced): 宇宙对象。
        """
        try:
            # 使用增强的语义网络初始化，支持num_concepts参数
            universe.initialize_semantic_network()
        except Exception as e:
            print(f"语义网络初始化失败: {e}，使用测试网络")
            self._create_test_network(universe)

    def _create_test_network(self, universe):
        """创建测试网络用于快速调试。

        Args:
            universe (CognitiveUniverseEnhanced): 宇宙对象。
        """
        test_nodes = ["算法", "数据结构", "优化", "递归", "迭代", "抽象", "模式识别",
                      "能量", "学习", "记忆", "思考", "创造", "理解", "应用"]
        universe.G.add_nodes_from(test_nodes)

        import random
        np.random.seed(42)
        random.seed(42)
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
        """计算能耗改善百分比。

        Args:
            energy_history (list): 能耗历史列表。

        Returns:
            float: 改善百分比。
        """
        if len(energy_history) < 2:
            return 0.0
        initial = energy_history[0]
        final = energy_history[-1]
        if initial == 0:
            return 0.0
        return ((initial - final) / initial) * 100

    def _analyze_emergence_results(self, results):
        """分析涌现实验结果，打印统计信息。

        Args:
            results (list): 个体结果列表。
        """
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
        """可视化涌现实验结果，生成能耗改善、涌现数量等图表。"""
        import matplotlib.pyplot as plt
        import numpy as np

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

        fig_dir = "results/emergence/figures"
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f"emergence_visualization_{self.timestamp}.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()


def main_fixed():
    """修复版本的主函数，运行涌现实验并可视化。"""
    study = EmergenceStudyFixed()

    # 运行纯粹涌现实验
    emergence_results = study.run_pure_emergence_experiment(
        num_individuals=2,
        max_iterations=8000  # 减少迭代次数以便快速测试
    )

    # 可视化结果
    study.visualize_emergence_results()

    return study


if __name__ == "__main__":
    main_fixed()