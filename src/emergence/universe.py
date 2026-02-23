"""
认知宇宙模块（基于两个公设）
-----------------------------
定义 PureEnergyDynamics 和 CognitiveUniverse 类，仅依赖能量最小化公设和遗忘机制，
观察自然涌现现象。
"""

import networkx as nx
import numpy as np
import random
import math
from typing import Dict, Any, List, Optional

from config import BASE_PARAMETERS

np.random.seed(42)
random.seed(42)

class PureEnergyDynamics:
    """纯粹的能量动力学，只实现两个公设：全局能量计算和局部变化尝试。"""

    def __init__(self, individual_params: Dict[str, Any]):
        """
        :param individual_params: 个体参数，包含 forgetting_rate 等
        """
        self.individual_params = individual_params
        self.energy_state = {}
        self.global_energy_history = []
        self.local_energy_changes = []
        self.forgetting_rate = individual_params.get('forgetting_rate', 0.002)

    def compute_global_energy(self, network: nx.Graph) -> float:
        """计算网络全局能量（平均边权重）。"""
        if network.number_of_edges() == 0:
            return 0
        energies = [network[u][v]['weight'] for u, v in network.edges()]
        return np.mean(energies)

    def compute_local_energy(self, node: str, network: nx.Graph) -> float:
        """计算节点局部能量（邻居边权重的均值）。"""
        if node not in network:
            return 0
        neighbors = list(network.neighbors(node))
        if not neighbors:
            return 0
        local_energy = 0
        for neighbor in neighbors:
            local_energy += network[node][neighbor]['weight']
        return local_energy / len(neighbors)

    def generate_random_changes(self, network: nx.Graph, num_changes: int = 5) -> List[Dict]:
        """生成随机变化尝试，包括边权重调整、节点激活、局部优化。"""
        changes = []
        nodes = list(network.nodes())

        if len(nodes) < 2:
            return changes

        for _ in range(num_changes):
            change_type = random.choice([
                'edge_weight_adjustment',
                'node_activation',
                'local_optimization'
            ])

            if change_type == 'edge_weight_adjustment':
                if network.number_of_edges() > 0:
                    u, v = random.choice(list(network.edges()))
                    current_weight = network[u][v]['weight']
                    new_weight = max(0.05, current_weight * random.uniform(0.8, 1.2))
                    changes.append({
                        'type': 'edge_weight_adjustment',
                        'edge': (u, v),
                        'old_weight': current_weight,
                        'new_weight': new_weight
                    })

            elif change_type == 'node_activation':
                node = random.choice(nodes)
                changes.append({
                    'type': 'node_activation',
                    'node': node,
                    'effect': random.uniform(0.9, 1.1)
                })

            elif change_type == 'local_optimization':
                if len(nodes) >= 3:
                    center = random.choice(nodes)
                    neighbors = list(network.neighbors(center))[:3]
                    if len(neighbors) >= 2:
                        changes.append({
                            'type': 'local_optimization',
                            'center': center,
                            'neighbors': neighbors,
                            'optimization_strength': random.uniform(0.7, 0.95)
                        })

        return changes

    def apply_change_and_compute(self, network: nx.Graph, change: Dict) -> float:
        """应用变化并计算新能量（不永久保留变化）。"""
        original_state = {}

        if change['type'] == 'edge_weight_adjustment':
            u, v = change['edge']
            original_state[(u, v)] = network[u][v]['weight']
            network[u][v]['weight'] = change['new_weight']

        elif change['type'] == 'node_activation':
            node = change['node']
            for neighbor in network.neighbors(node):
                original_state[(node, neighbor)] = network[node][neighbor]['weight']
                network[node][neighbor]['weight'] *= change['effect']

        elif change['type'] == 'local_optimization':
            center = change['center']
            for neighbor in change['neighbors']:
                if network.has_edge(center, neighbor):
                    original_state[(center, neighbor)] = network[center][neighbor]['weight']
                    network[center][neighbor]['weight'] *= change['optimization_strength']

        new_energy = self.compute_global_energy(network)

        # 恢复原始状态
        for key, value in original_state.items():
            if isinstance(key, tuple) and len(key) == 2:
                u, v = key
                if network.has_edge(u, v):
                    network[u][v]['weight'] = value

        return new_energy

    def keep_change(self, network: nx.Graph, change: Dict) -> None:
        """永久保留变化（更新网络）。"""
        if change['type'] == 'edge_weight_adjustment':
            u, v = change['edge']
            network[u][v]['weight'] = change['new_weight']

        elif change['type'] == 'node_activation':
            node = change['node']
            for neighbor in network.neighbors(node):
                network[node][neighbor]['weight'] *= change['effect']

        elif change['type'] == 'local_optimization':
            center = change['center']
            for neighbor in change['neighbors']:
                if network.has_edge(center, neighbor):
                    network[center][neighbor]['weight'] *= change['optimization_strength']


class CognitiveUniverse:
    """认知宇宙，基于两个公设：认知时空和能量动力学，观察自然涌现。

    主要机制：
        - 学习：通过遍历降低边权重
        - 遗忘：长时间未激活的边向原始权重恢复
        - 随机遍历：探索网络
    不包含任何预设的压缩或迁移算法，仅记录观察到的涌现现象。
    """

    def __init__(self, individual_params: Optional[Dict[str, Any]] = None, network_seed: int = 42):
        """
        :param individual_params: 个体参数，默认使用 BASE_PARAMETERS
        :param network_seed: 随机种子
        """
        if individual_params is None:
            individual_params = BASE_PARAMETERS.copy()

        self.G = nx.Graph()
        self.individual_params = individual_params
        self.network_seed = network_seed
        self.iteration_count = 0

        self.energy_dynamics = PureEnergyDynamics(individual_params)

        self.state_history = []
        self.current_energy_level = 1.0

        self.energy_history = []

        self.traversal_history = []

        self.last_activation_time = {}

        self.observations = {
            'spontaneous_compressions': [],
            'emergent_migrations': [],
            'traversal_patterns': [],
            'energy_minimization_traces': [],
            'network_evolution_snapshots': []
        }

        self.learning_rate = individual_params.get('base_learning_rate', 0.85)
        self.forgetting_rate = individual_params.get('forgetting_rate', 0.002)
        self.base_forgetting_strength = 0.002

        random.seed(network_seed)
        np.random.seed(network_seed)
        print("认知宇宙初始化完成: 遗忘机制已激活")

    def initialize_semantic_network(self) -> None:
        """初始化语义网络，从 SemanticConceptNetwork 构建节点和边。"""
        from core.semantic_network import SemanticConceptNetwork

        semantic_net = SemanticConceptNetwork()
        semantic_net.build_comprehensive_network()

        nodes = list(semantic_net.concept_definitions.keys())
        self.G.add_nodes_from(nodes)

        initial_edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1:], i + 1):
                similarity = semantic_net.calculate_semantic_similarity(node1, node2)

                if similarity > 0.1:
                    energy = 2.0 - similarity * 1.5
                    energy = max(0.3, min(2.0, energy))

                    initial_edges.append((node1, node2, {
                        'weight': energy,
                        'traversal_count': 0,
                        'original_weight': energy,
                        'similarity': similarity
                    }))
                    self.last_activation_time[(node1, node2)] = 0

        for u, v, attr in initial_edges:
            self.G.add_edge(u, v, **attr)

        print(f"语义网络初始化: {len(nodes)}节点, {len(initial_edges)}条边")
        print(f"初始全局能量: {self.calculate_network_energy():.3f}")

    def calculate_network_energy(self) -> float:
        """计算当前网络平均能耗。"""
        if self.G.number_of_edges() == 0:
            return 0
        energies = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        return np.mean(energies)

    def basic_energy_optimization(self) -> bool:
        """基本的能量优化：随机选择一条边进行学习（降低能耗），并可能扩散到邻居。"""
        if self.G.number_of_edges() == 0:
            return False

        edges = list(self.G.edges())
        u, v = random.choice(edges)

        self.record_edge_activation(u, v)

        if random.random() < 0.5:
            neighbors = list(self.G.neighbors(u))
            if neighbors:
                neighbor = random.choice(neighbors)
                if self.G.has_edge(u, neighbor):
                    self.record_edge_activation(u, neighbor)

        return True

    def record_edge_activation(self, u: str, v: str) -> None:
        """记录边的激活，更新激活时间并应用学习效应（降低权重）。"""
        self.last_activation_time[(u, v)] = self.iteration_count

        current_energy = self.G[u][v]['weight']
        learning_rate = 0.1
        new_energy = current_energy * (1 - learning_rate)
        self.G[u][v]['weight'] = max(0.05, new_energy)

    def apply_basic_forgetting(self) -> None:
        """应用基础遗忘机制：长时间未激活的边向原始能量恢复。"""
        current_time = self.iteration_count

        for u, v in self.G.edges():
            time_since_activation = current_time - self.last_activation_time.get((u, v), 0)

            if time_since_activation > 0:
                current_energy = self.G[u][v]['weight']
                original_energy = self.G[u][v].get('original_weight', 2.0)

                forget_factor = self._compute_forget_factor(time_since_activation)

                new_energy = current_energy + (original_energy - current_energy) * forget_factor
                new_energy = min(new_energy, original_energy)

                self.G[u][v]['weight'] = max(0.1, new_energy)

    def _compute_forget_factor(self, time_gap: int) -> float:
        """计算遗忘因子，基于指数衰减。"""
        base_rate = self.forgetting_rate
        time_factor = 1 - math.exp(-time_gap / 800)
        forget_factor = base_rate * time_factor
        return min(forget_factor, 0.15)

    def _random_traversal(self) -> None:
        """随机遍历：从随机节点出发随机行走2-4步，记录路径并激活沿途边。"""
        nodes = list(self.G.nodes())
        if len(nodes) < 2:
            return

        start_node = random.choice(nodes)
        path = [start_node]
        current_node = start_node
        path_length = random.randint(2, 4)

        for step in range(path_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            if next_node not in path:
                path.append(next_node)
                current_node = next_node
            else:
                break

        if len(path) >= 2:
            self.traversal_history.append({
                'path': path.copy(),
                'iteration': self.iteration_count,
                'type': 'random_traversal'
            })

            current_time = self.iteration_count
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.G.has_edge(u, v):
                    self.record_edge_activation(u, v)

    def evolve(self, iterations: int = 1000, observation_interval: int = 100) -> Dict[str, List]:
        """让宇宙自然演化，观察涌现现象。

        :param iterations: 迭代次数
        :param observation_interval: 记录网络快照的间隔
        :return: observations 字典
        """
        print(f"开始认知宇宙演化: {iterations}次迭代")
        print("只执行基本能量优化，观察自然涌现...")

        initial_energy = self.calculate_network_energy()
        self.energy_history.append(initial_energy)
        print(f"初始全局能量: {initial_energy:.3f}")

        for i in range(iterations):
            self.iteration_count += 1

            self.basic_energy_optimization()

            if random.random() < 0.3:
                self._random_traversal()

            if i % 10 == 0:
                self.apply_basic_forgetting()

            current_energy = self.calculate_network_energy()
            self.energy_history.append(current_energy)

            if i % observation_interval == 0:
                self._take_network_snapshot()

            if i % 500 == 0:
                current_energy = self.calculate_network_energy()
                improvement = ((initial_energy - current_energy) / initial_energy * 100) if initial_energy > 0 else 0
                print(f"迭代 {i}: 全局能量 = {current_energy:.3f} (改善: {improvement:.1f}%)")

        final_energy = self.calculate_network_energy()
        total_improvement = ((initial_energy - final_energy) / initial_energy * 100) if initial_energy > 0 else 0

        print(f"\n演化完成!")
        print(f"最终全局能量: {final_energy:.3f}")
        print(f"总改善: {total_improvement:.1f}%")
        print(f"网络规模: {self.G.number_of_nodes()}节点, {self.G.number_of_edges()}边")
        print(f"遍历次数: {len(self.traversal_history)}")

        return self.observations

    def _take_network_snapshot(self) -> None:
        """记录当前网络快照（统计信息）。"""
        snapshot = {
            'iteration': self.iteration_count,
            'global_energy': self.calculate_network_energy(),
            'node_count': self.G.number_of_nodes(),
            'edge_count': self.G.number_of_edges(),
            'average_degree': np.mean([d for n, d in self.G.degree()]) if self.G.number_of_nodes() > 0 else 0,
            'clustering_coefficient': nx.average_clustering(self.G) if self.G.number_of_nodes() > 0 else 0
        }
        self.observations['network_evolution_snapshots'].append(snapshot)

    def get_emergence_metrics(self) -> Dict[str, float]:
        """获取涌现现象的量化指标。"""
        metrics = {
            'energy_minimization_efficiency': 0,
            'structural_emergence_index': 0,
            'cognitive_complexity': 0,
            'adaptation_rate': 0
        }

        if len(self.energy_dynamics.global_energy_history) > 1:
            initial_energy = self.energy_dynamics.global_energy_history[0]
            final_energy = self.energy_dynamics.global_energy_history[-1]
            metrics['energy_minimization_efficiency'] = (
                initial_energy - final_energy) / initial_energy if initial_energy > 0 else 0

            energy_changes = np.diff(self.energy_dynamics.global_energy_history)
            negative_changes = [change for change in energy_changes if change < 0]
            metrics['adaptation_rate'] = len(negative_changes) / len(energy_changes) if energy_changes else 0

        if len(self.observations['network_evolution_snapshots']) > 1:
            initial_clustering = self.observations['network_evolution_snapshots'][0]['clustering_coefficient']
            final_clustering = self.observations['network_evolution_snapshots'][-1]['clustering_coefficient']
            metrics['structural_emergence_index'] = final_clustering - initial_clustering

            density = self.G.number_of_edges() / (self.G.number_of_nodes() *
                                                  (self.G.number_of_nodes() - 1) / 2) if self.G.number_of_nodes() > 1 else 0
            metrics['cognitive_complexity'] = density * final_clustering

        return metrics

    def report_emergence_findings(self) -> Dict[str, float]:
        """打印并返回涌现观察报告。"""
        print("\n" + "=" * 60)
        print("纯粹涌现观察报告")
        print("=" * 60)

        metrics = self.get_emergence_metrics()

        print(f"能量最小化效率: {metrics['energy_minimization_efficiency']:.3f}")
        print(f"结构涌现指数: {metrics['structural_emergence_index']:.3f}")
        print(f"认知复杂性: {metrics['cognitive_complexity']:.3f}")
        print(f"适应率: {metrics['adaptation_rate']:.3f}")

        print(f"\n网络演化快照: {len(self.observations['network_evolution_snapshots'])}次记录")
        print(f"遍历模式: {len(self.traversal_history)}次遍历")

        if self.energy_dynamics.global_energy_history:
            energy_reduction = self.energy_dynamics.global_energy_history[0] - \
                               self.energy_dynamics.global_energy_history[-1]
            print(f"能量降低总量: {energy_reduction:.3f}")

        print(f"\n观察结论:")
        if metrics['energy_minimization_efficiency'] > 0.1:
            print("✓ 观察到显著的能量最小化趋势")
        if metrics['structural_emergence_index'] > 0:
            print("✓ 观察到网络结构的自组织")
        if metrics['adaptation_rate'] > 0.3:
            print("✓ 观察到良好的环境适应性")

        return metrics

    def get_network_stats(self) -> Dict[str, Any]:
        """返回当前网络统计信息。"""
        return {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'iterations': self.iteration_count,
            'global_energy': self.calculate_network_energy(),
            'traversal_count': len(self.traversal_history),
            'snapshots_count': len(self.observations['network_evolution_snapshots'])
        }


if __name__ == "__main__":
    universe = CognitiveUniverse()
    universe.initialize_semantic_network()
    universe.evolve(iterations=10000)
    print("CognitiveUniverse 演化测试完成")