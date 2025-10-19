import networkx as nx
import numpy as np
import random
import math
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

from core.cognitive_graph import BaseCognitiveGraph
from core.cognitive_states import CognitiveState, CognitiveStateManager
from emergence.detector import EmergenceDetector
from emergence.metrics import NaturalEmergenceMetrics
from emergence.observer import EmergenceObserver


class PureEnergyCognitiveGraph(BaseCognitiveGraph):
    """
    纯粹能量认知图 - 只实现两个公设，观察一切自然涌现
    移除所有人工机制，只保留最基本的能量优化
    """

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        super().__init__(individual_params, network_seed)

        # 移除人工机制相关的参数
        self.compression_bias = 0
        self.migration_bias = 0

        # 涌现观察系统
        self.emergence_observer = EmergenceObserver()
        self.emergence_detector = EmergenceDetector()
        self.emergence_metrics = NaturalEmergenceMetrics()

        # 纯粹的能量优化记录
        self.pure_energy_history = []
        self.natural_compressions = []
        self.natural_migrations = []

        # 网络历史用于分析涌现
        self.network_history = []

        print("初始化纯粹能量认知图 - 只实现两个公设")

    def monte_carlo_iteration(self, max_iterations: int = 5000):
        """纯粹能量优化的蒙特卡洛模拟 - 移除所有人工机制"""
        print(f"=== 纯粹能量优化模拟开始 ===")
        print(f"初始状态: {self.current_state.value}, 主观能耗: {self.subjective_energy:.2f}")
        print("注意: 所有概念压缩和原理迁移都将是自然涌现的!")

        # 记录初始网络状态
        self._record_network_snapshot()

        for iteration in range(max_iterations):
            self.iteration_count += 1

            # 基本的状态更新
            if iteration % 100 == 0:
                self.update_cognitive_state()

            # 应用遗忘机制
            self._apply_forgetting()

            # 纯粹的能量优化 - 只做最基本的操作
            current_energy = self.calculate_network_energy()
            self.energy_history.append(current_energy)
            self.pure_energy_history.append(current_energy)

            # 选择基本操作（移除压缩和迁移）
            operation = self._select_basic_operation()

            if operation == "hard_traversal":
                self._basic_hard_traversal()
            elif operation == "soft_traversal":
                self._basic_soft_traversal()
            # 移除 compression 和 migration 操作

            # 观察自然涌现
            if iteration % 200 == 0:  # 每200次迭代观察一次
                self._observe_natural_emergence(iteration)

            if iteration % 500 == 0:
                stats = self.get_network_stats()
                print(f"迭代 {iteration}, 状态: {self.current_state.value}, "
                      f"网络能耗: {current_energy:.3f}")

        print(f"=== 纯粹能量优化模拟完成 ===")
        self._report_natural_emergence()

    def _select_basic_operation(self) -> str:
        """选择基本操作 - 移除压缩和迁移"""
        state_operations = {
            CognitiveState.FOCUSED: {
                "hard_traversal": 0.7,
                "soft_traversal": 0.3
            },
            CognitiveState.EXPLORATORY: {
                "hard_traversal": 0.4,
                "soft_traversal": 0.6
            },
            CognitiveState.FATIGUED: {
                "hard_traversal": 0.3,
                "soft_traversal": 0.7
            },
            CognitiveState.INSPIRED: {
                "hard_traversal": 0.5,
                "soft_traversal": 0.5
            }
        }

        probs = state_operations[self.current_state]
        rand_val = random.random()
        cumulative = 0

        for op, prob in probs.items():
            cumulative += prob
            if rand_val <= cumulative:
                return op

        return "hard_traversal"

    def _basic_hard_traversal(self):
        """基本的硬遍历 - 没有人工优化"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return

        start_node = random.choice(available_nodes)
        path = self._find_basic_path(start_node, 3, "hard")

        if path and len(path) >= 2:
            self.traverse_path(path, "hard")

    def _basic_soft_traversal(self):
        """基本的软遍历 - 没有人工优化"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return

        start_node = random.choice(available_nodes)
        path = self._find_basic_path(start_node, 2, "soft")

        if path and len(path) >= 2:
            self.traverse_path(path, "soft")

    def _find_basic_path(self, start_node: str, max_length: int, traversal_type: str) -> Optional[List[str]]:
        """寻找基本路径 - 没有人工优化策略"""
        path = [start_node]
        current_node = start_node

        for _ in range(max_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break

            if traversal_type == "hard":
                # 硬遍历：选择能耗较低的邻居
                neighbors.sort(key=lambda n: self.G[current_node][n]['weight'])
            else:
                # 软遍历：随机选择
                random.shuffle(neighbors)

            found_next = False
            for neighbor in neighbors:
                if neighbor not in path:
                    edge_energy = self.G[current_node][neighbor]['weight']
                    can_traverse, _ = self.can_traverse_edge(edge_energy, traversal_type)
                    if can_traverse:
                        path.append(neighbor)
                        current_node = neighbor
                        found_next = True
                        break

            if not found_next:
                break

        return path if len(path) >= 2 else None

    def _observe_natural_emergence(self, iteration: int):
        """观察自然涌现现象"""
        # 记录当前网络状态
        self._record_network_snapshot()

        # 检测自然概念压缩
        compressions = self.emergence_detector.detect_spontaneous_compression(
            self.G, self.energy_history, self.traversal_history
        )

        for compression in compressions:
            if not self._is_compression_already_recorded(compression):
                compression['emergence_iteration'] = iteration
                compression['type'] = 'natural_compression'
                self.natural_compressions.append(compression)
                print(f"迭代 {iteration}: 观察到自然概念压缩! "
                      f"中心: {compression['center']}, 相关节点: {len(compression['related_nodes'])}")

        # 检测自然原理迁移
        migrations = self.emergence_detector.detect_emergent_migration(
            self.G, self.traversal_history, iteration
        )

        for migration in migrations:
            if not self._is_migration_already_recorded(migration):
                migration['emergence_iteration'] = iteration
                migration['type'] = 'natural_migration'
                self.natural_migrations.append(migration)
                print(f"迭代 {iteration}: 观察到自然原理迁移! "
                      f"原理: {migration['principle_node']}, 效率提升: {migration['efficiency_gain']:.3f}")

        # 记录观察结果
        self.emergence_observer.record_observations(
            compressions, migrations, iteration
        )

    def _record_network_snapshot(self):
        """记录网络快照用于分析涌现"""
        snapshot = {
            'iteration': self.iteration_count,
            'nodes': list(self.G.nodes()),
            'edges': [(u, v, self.G[u][v]['weight']) for u, v in self.G.edges()],
            'avg_energy': self.calculate_network_energy(),
            'clustering_coef': nx.average_clustering(self.G) if self.G.number_of_nodes() > 0 else 0
        }
        self.network_history.append(snapshot)

    def _is_compression_already_recorded(self, new_compression: Dict) -> bool:
        """检查压缩是否已经记录过"""
        for existing in self.natural_compressions:
            if (existing['center'] == new_compression['center'] and
                    set(existing['related_nodes']) == set(new_compression['related_nodes'])):
                return True
        return False

    def _is_migration_already_recorded(self, new_migration: Dict) -> bool:
        """检查迁移是否已经记录过"""
        for existing in self.natural_migrations:
            if (existing['principle_node'] == new_migration['principle_node'] and
                    existing['from_node'] == new_migration['from_node'] and
                    existing['to_node'] == new_migration['to_node']):
                return True
        return False

    def _report_natural_emergence(self):
        """报告自然涌现结果"""
        print("\n" + "=" * 60)
        print("纯粹能量模型 - 自然涌现观察报告")
        print("=" * 60)

        print(f"\n📊 模拟统计:")
        print(f"  总迭代次数: {self.iteration_count}")
        print(f"  最终网络能耗: {self.calculate_network_energy():.3f}")
        print(f"  能耗降低: {((self.energy_history[0] - self.energy_history[-1]) / self.energy_history[0] * 100):.1f}%")

        print(f"\n🎯 自然概念压缩:")
        print(f"  观察到的压缩事件: {len(self.natural_compressions)}")
        for i, compression in enumerate(self.natural_compressions[:5]):  # 显示前5个
            print(f"  {i + 1}. 中心: {compression['center']}")
            print(f"     相关节点: {compression['related_nodes']}")
            print(f"     涌现迭代: {compression['emergence_iteration']}")
            print(f"     能量协同性: {compression.get('energy_synergy', 0):.3f}")

        print(f"\n🌉 自然原理迁移:")
        print(f"  观察到的迁移事件: {len(self.natural_migrations)}")
        for i, migration in enumerate(self.natural_migrations[:5]):  # 显示前5个
            print(f"  {i + 1}. 原理节点: {migration['principle_node']}")
            print(f"     连接: {migration['from_node']} -> {migration['to_node']}")
            print(f"     效率提升: {migration.get('efficiency_gain', 0):.3f}")
            print(f"     涌现迭代: {migration['emergence_iteration']}")

        # 计算涌现指标
        metrics = self.emergence_metrics.calculate_emergence_metrics(
            self.network_history, self.traversal_history
        )

        print(f"\n📈 涌现强度指标:")
        print(f"  概念压缩涌现强度: {metrics.get('compression_emergence_strength', 0):.3f}")
        print(f"  原理迁移涌现强度: {metrics.get('migration_emergence_strength', 0):.3f}")
        print(f"  网络结构创新率: {metrics.get('network_innovation_rate', 0):.3f}")

    def get_emergence_statistics(self) -> Dict[str, Any]:
        """获取涌现统计信息"""
        return {
            'total_iterations': self.iteration_count,
            'final_energy': self.calculate_network_energy(),
            'energy_reduction_percent': (
                        (self.energy_history[0] - self.energy_history[-1]) / self.energy_history[0] * 100),
            'natural_compressions_count': len(self.natural_compressions),
            'natural_migrations_count': len(self.natural_migrations),
            'compression_centers': [comp['center'] for comp in self.natural_compressions],
            'migration_principles': [mig['principle_node'] for mig in self.natural_migrations],
            'emergence_metrics': self.emergence_metrics.calculate_emergence_metrics(
                self.network_history, self.traversal_history
            )
        }

    def visualize_natural_emergence(self):
        """可视化自然涌现过程"""
        import matplotlib.pyplot as plt

        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 能量收敛过程
        ax1.plot(self.pure_energy_history, 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('平均认知能耗')
        ax1.set_title('纯粹能量优化收敛过程')
        ax1.grid(True, alpha=0.3)

        # 标记自然涌现事件
        for compression in self.natural_compressions:
            iter_num = compression['emergence_iteration']
            if iter_num < len(self.pure_energy_history):
                ax1.axvline(x=iter_num, color='red', alpha=0.5, linestyle='--')

        for migration in self.natural_migrations:
            iter_num = migration['emergence_iteration']
            if iter_num < len(self.pure_energy_history):
                ax1.axvline(x=iter_num, color='green', alpha=0.5, linestyle=':')

        # 2. 涌现事件时间线
        emergence_timeline = []
        for comp in self.natural_compressions:
            emergence_timeline.append((comp['emergence_iteration'], '压缩', comp['center']))
        for mig in self.natural_migrations:
            emergence_timeline.append((mig['emergence_iteration'], '迁移', mig['principle_node']))

        emergence_timeline.sort()

        if emergence_timeline:
            iterations, types, nodes = zip(*emergence_timeline)
            colors = ['red' if t == '压缩' else 'green' for t in types]
            ax2.scatter(iterations, range(len(iterations)), c=colors, alpha=0.6)
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('涌现事件序号')
            ax2.set_title('自然涌现事件时间线')
            ax2.grid(True, alpha=0.3)

        # 3. 网络结构演化
        if len(self.network_history) > 10:
            sample_indices = np.linspace(0, len(self.network_history) - 1, 10, dtype=int)
            clustering_evolution = [self.network_history[i]['clustering_coef'] for i in sample_indices]
            ax3.plot(sample_indices, clustering_evolution, 'o-', color='purple')
            ax3.set_xlabel('迭代次数')
            ax3.set_ylabel('平均聚类系数')
            ax3.set_title('网络结构演化')
            ax3.grid(True, alpha=0.3)

        # 4. 涌现强度指标
        metrics = self.emergence_metrics.calculate_emergence_metrics(
            self.network_history, self.traversal_history
        )

        metric_names = ['压缩涌现', '迁移涌现', '结构创新']
        metric_values = [
            metrics.get('compression_emergence_strength', 0),
            metrics.get('migration_emergence_strength', 0),
            metrics.get('network_innovation_rate', 0)
        ]

        bars = ax4.bar(metric_names, metric_values, color=['red', 'green', 'blue'], alpha=0.7)
        ax4.set_ylabel('涌现强度')
        ax4.set_title('自然涌现强度指标')

        # 在柱状图上添加数值
        for bar, value in zip(bars, metric_values):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def calculate_semantic_similarity(self, node1: str, node2: str) -> float:
        """计算语义相似度 - 简化版本，用于基本功能"""
        # 在纯粹能量模型中，我们使用简单的相似度计算
        # 实际应用中可以从语义网络获取
        if hasattr(self, 'semantic_network'):
            return self.semantic_network.calculate_semantic_similarity(node1, node2)
        else:
            # 默认实现 - 基于节点名称的简单相似度
            words1 = set(node1)
            words2 = set(node2)
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0