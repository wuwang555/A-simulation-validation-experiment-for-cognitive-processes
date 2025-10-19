import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

from emergence.universe import *
from emergence.observer import *
from emergence.detector import *
from emergence.metrics import *
from utils.individual_variation import IndividualVariation, create_enhanced_individual_params
from config import *


class EmergenceStudy:
    """涌现现象研究实验"""

    def __init__(self):
        self.results = {}
        self.comparison_data = {}

    def run_pure_emergence_experiment(self, num_individuals=3, max_iterations=None):
        """运行纯粹的涌现观察实验"""
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

            # 初始化能量历史
            universe.energy_history = [self._calculate_network_energy(universe)]

            start_time = time.time()

            # 主观察循环
            for iteration in range(max_iterations):
                universe.iteration_count += 1

                # 第一步：纯粹能量优化
                optimization_occurred = universe.basic_energy_optimization()

                # 第二步：记录当前能量
                current_energy = self._calculate_network_energy(universe)
                universe.energy_history.append(current_energy)

                # 第三步：定期观察涌现现象（每200次迭代检查一次）
                if iteration % 200 == 0:
                    # 使用正确的网络属性
                    network = universe.G

                    # 观察概念压缩涌现 - 修复参数匹配
                    compressions = observer.observe_compression_emergence(
                        network, universe.energy_history, iteration
                    )

                    # 观察原理迁移涌现 - 修复参数匹配
                    migrations = observer.observe_migration_emergence(
                        network, universe.traversal_history, None, iteration  # semantic_network设为None
                    )

                    # 使用检测器检测涌现现象
                    detected_compressions = detector.detect_spontaneous_compression(
                        universe.G, universe.energy_history, universe.traversal_history
                    )

                    # 观察原理迁移涌现
                    detected_migrations = detector.detect_emergent_migration(
                        universe.G, universe.traversal_history, iteration
                    )

                    # 记录确认的涌现现象
                    all_compressions = compressions + detected_compressions
                    all_migrations = migrations + detected_migrations

                    if all_compressions:
                        print(f"  迭代 {iteration}: 观察到 {len(all_compressions)} 个概念压缩涌现!")
                        for compression in all_compressions:
                            universe.observations['spontaneous_compressions'].append({
                                **compression,
                                'individual_id': individual_id,
                                'iteration': iteration
                            })

                    if all_migrations:
                        print(f"  迭代 {iteration}: 观察到 {len(all_migrations)} 个原理迁移涌现!")
                        for migration in all_migrations:
                            universe.observations['emergent_migrations'].append({
                                **migration,
                                'individual_id': individual_id,
                                'iteration': iteration
                            })

                # 进度显示
                if iteration % 1000 == 0:
                    print(f"  进度: {iteration}/{max_iterations}, 当前能耗: {current_energy:.3f}")

            end_time = time.time()

            # 收集个体结果
            individual_result = {
                'individual_id': individual_id,
                'parameters': individual_params,
                'observations': universe.observations,
                'final_energy': current_energy,
                'initial_energy': universe.energy_history[0],
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
        """初始化宇宙网络"""
        try:
            # 尝试从语义网络初始化
            from core.semantic_network import SemanticConceptNetwork
            semantic_net = SemanticConceptNetwork()
            semantic_net.build_comprehensive_network()

            if hasattr(universe, 'initialize_from_semantic_network'):
                universe.initialize_from_semantic_network(semantic_net)
                print(f"语义网络初始化: {universe.G.number_of_nodes()}节点, {universe.G.number_of_edges()}条边")
            else:
                # 手动初始化
                self._manual_network_initialization(universe, semantic_net)

        except Exception as e:
            print(f"语义网络初始化失败: {e}，使用测试网络")
            self._create_test_network(universe)

    def _manual_network_initialization(self, universe, semantic_net):
        """手动网络初始化"""
        nodes = list(semantic_net.concept_definitions.keys())[:20]  # 使用前20个概念
        universe.G.add_nodes_from(nodes)

        # 添加基于语义相似度的边
        edge_count = 0
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1:], i + 1):
                try:
                    similarity = semantic_net.calculate_semantic_similarity(node1, node2)
                    if similarity > 0.1:
                        energy = 2.0 - similarity * 1.5
                        energy = max(0.3, min(2.0, energy))
                        universe.G.add_edge(node1, node2, weight=energy)
                        edge_count += 1
                except:
                    continue

        print(f"手动网络初始化: {len(nodes)}个节点, {edge_count}条边")

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

    def _calculate_network_energy(self, universe):
        """计算网络能量"""
        network = universe.G
        if network is None or network.number_of_edges() == 0:
            return 1.0  # 默认能量值
        energies = [network[u][v]['weight'] for u, v in network.edges()]
        return np.mean(energies)

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
    def run_comparison_experiment(self, num_individuals=2, max_iterations=8000):
        """对比实验：人工机制 vs 自然涌现"""
        print("\n" + "=" * 60)
        print("对比实验：人工干预机制 vs 自然涌现现象")
        print("=" * 60)

        # 运行自然涌现实验
        print("\n--- 自然涌现组 ---")
        emergence_group = self.run_pure_emergence_experiment(
            num_individuals=num_individuals,
            max_iterations=max_iterations
        )

        # 运行人工机制实验（使用现有代码）
        print("\n--- 人工干预组 ---")
        from experiments.population_study import run_semantic_enhanced_experiment
        manual_group = run_semantic_enhanced_experiment(
            num_individuals=num_individuals,
            max_iterations=max_iterations
        )

        # 对比分析
        self._compare_emergence_vs_manual(emergence_group, manual_group)

        return {
            'emergence_group': emergence_group,
            'manual_group': manual_group
        }

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

        # 涌现时间分析
        emergence_timings = self._analyze_emergence_timing(results)
        print(f"\n涌现时间分析:")
        print(f"  首次压缩平均迭代: {emergence_timings['first_compression']:.0f}")
        print(f"  首次迁移平均迭代: {emergence_timings['first_migration']:.0f}")

    def _analyze_emergence_timing(self, results):
        """分析涌现现象出现的时间"""
        first_compressions = []
        first_migrations = []

        for result in results:
            compressions = result['observations']['spontaneous_compressions']
            migrations = result['observations']['emergent_migrations']

            if compressions:
                first_compressions.append(min(c['iteration'] for c in compressions))
            if migrations:
                first_migrations.append(min(m['iteration'] for m in migrations))

        return {
            'first_compression': np.mean(first_compressions) if first_compressions else 0,
            'first_migration': np.mean(first_migrations) if first_migrations else 0
        }

    def _compare_emergence_vs_manual(self, emergence_group, manual_group):
        """对比自然涌现与人工干预的结果"""
        print("\n" + "=" * 50)
        print("自然涌现 vs 人工干预 对比分析")
        print("=" * 50)

        # 能耗改善对比
        emergence_improvements = [r['energy_improvement'] for r in emergence_group]
        manual_improvements = [r.get('improvement', 0) for r in manual_group]  # 修复键名

        print(f"能耗改善对比:")
        print(f"  自然涌现组: {np.mean(emergence_improvements):.1f}% ± {np.std(emergence_improvements):.1f}%")
        print(f"  人工干预组: {np.mean(manual_improvements):.1f}% ± {np.std(manual_improvements):.1f}%")

        # 现象数量对比
        emergence_compressions = [r['compression_count'] for r in emergence_group]
        manual_compressions = [r.get('compression_centers', 0) for r in manual_group]  # 修复键名

        print(f"\n概念压缩对比:")
        print(f"  自然涌现组: {np.mean(emergence_compressions):.1f}次")
        print(f"  人工干预组: {np.mean(manual_compressions):.1f}次")

        # 质量分析
        emergence_quality = self._assess_emergence_quality(emergence_group)
        manual_quality = self._assess_manual_quality(manual_group)

        print(f"\n现象质量对比:")
        print(f"  自然涌现平均质量: {emergence_quality:.3f}")
        print(f"  人工干预平均质量: {manual_quality:.3f}")

    def _assess_emergence_quality(self, emergence_group):
        """评估自然涌现现象的质量"""
        quality_scores = []
        for result in emergence_group:
            # 基于涌现现象的稳定性和能耗改善评估质量
            compressions = result['observations']['spontaneous_compressions']
            migrations = result['observations']['emergent_migrations']

            if compressions or migrations:
                # 简化的质量评估：基于现象数量和能耗改善
                quality = min(1.0, (len(compressions) + len(migrations)) * 0.1 + result['energy_improvement'] * 0.01)
                quality_scores.append(quality)

        return np.mean(quality_scores) if quality_scores else 0.0

    def _assess_manual_quality(self, manual_group):
        """评估人工干预现象的质量"""
        quality_scores = []
        for result in manual_group:
            # 基于压缩中心和迁移桥梁数量评估质量
            compressions = result.get('compression_centers', 0)
            migrations = result.get('migration_bridges', 0)
            improvement = result.get('improvement', 0)

            quality = min(1.0, (compressions + migrations) * 0.1 + improvement * 0.01)
            quality_scores.append(quality)

        return np.mean(quality_scores) if quality_scores else 0.0

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

        # 3. 涌现时间分布
        emergence_times = self._get_emergence_timing_data(results)
        if emergence_times['compression_times']:
            ax3.hist(emergence_times['compression_times'], bins=10, alpha=0.7,
                     color='orange', label='概念压缩')
        if emergence_times['migration_times']:
            ax3.hist(emergence_times['migration_times'], bins=10, alpha=0.7,
                     color='purple', label='原理迁移')
        ax3.set_title('涌现现象时间分布')
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('现象数量')
        ax3.legend()

        # 4. 能量收敛曲线示例
        if results and len(results) > 0 and hasattr(results[0]['universe'], 'energy_history'):
            sample_energy_history = results[0]['universe'].energy_history
            ax4.plot(sample_energy_history, 'b-', alpha=0.7)
            ax4.set_title('典型能量收敛过程')
            ax4.set_xlabel('迭代次数')
            ax4.set_ylabel('平均认知能耗')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _get_emergence_timing_data(self, results):
        """获取涌现现象的时间数据"""
        compression_times = []
        migration_times = []

        for result in results:
            for compression in result['observations']['spontaneous_compressions']:
                compression_times.append(compression['iteration'])
            for migration in result['observations']['emergent_migrations']:
                migration_times.append(migration['iteration'])

        return {
            'compression_times': compression_times,
            'migration_times': migration_times
        }

    def generate_emergence_report(self):
        """生成详细的涌现观察报告"""
        if 'pure_emergence' not in self.results:
            print("请先运行实验!")
            return

        results = self.results['pure_emergence']

        print("\n" + "=" * 60)
        print("自然涌现现象详细观察报告")
        print("=" * 60)

        for result in results:
            print(f"\n--- {result['individual_id']} ---")
            print(f"计算时间: {result['computation_time']:.1f}秒")
            print(f"能耗改善: {result['energy_improvement']:.1f}%")

            # 详细现象报告
            compressions = result['observations']['spontaneous_compressions']
            migrations = result['observations']['emergent_migrations']

            print(f"\n概念压缩涌现 ({len(compressions)}次):")
            for i, compression in enumerate(compressions[:3]):  # 显示前3个
                print(f"  {i + 1}. 中心: {compression.get('center_candidate', compression.get('center', 'N/A'))}")
                nodes = compression.get('nodes', [])
                print(f"     相关节点: {nodes[:3]}...")  # 显示前3个节点
                print(f"     涌现迭代: {compression['iteration']}")
                print(f"     能量协同性: {compression.get('energy_synergy', 0):.3f}")

            print(f"\n原理迁移涌现 ({len(migrations)}次):")
            for i, migration in enumerate(migrations[:3]):
                print(f"  {i + 1}. 原理节点: {migration.get('principle_node', migration.get('mediator_node', 'N/A'))}")
                print(f"     效率提升: {migration.get('efficiency_gain', 0):.3f}")
                print(f"     涌现迭代: {migration['iteration']}")


def main():
    """主函数：运行涌现研究实验"""
    study = EmergenceStudy()

    # 运行纯粹涌现实验
    emergence_results = study.run_pure_emergence_experiment(
        num_individuals=3,
        max_iterations=5000  # 减少迭代次数以便快速测试
    )

    # 可视化结果
    study.visualize_emergence_results()

    # 生成详细报告
    study.generate_emergence_report()

    # 运行对比实验（可选）
    run_comparison = input("\n是否运行对比实验? (y/n): ").lower().strip()
    if run_comparison == 'y':
        comparison_results = study.run_comparison_experiment(
            num_individuals=2,
            max_iterations=4000
        )

    return study


if __name__ == "__main__":
    main()