import networkx as nx
import numpy as np
import random
import math
from typing import Dict, Any, List, Tuple

from core.cognitive_states import CognitiveState, CognitiveStateManager
from config import BASE_PARAMETERS, ENERGY_CONFIG


class PureEnergyDynamics:
    """纯粹的能量动力学 - 只实现两个公设"""

    def __init__(self, individual_params: Dict[str, Any]):
        self.individual_params = individual_params
        self.energy_state = {}
        self.global_energy_history = []
        self.local_energy_changes = []
        self.forgetting_rate = individual_params.get('forgetting_rate', 0.002)

    def compute_global_energy(self, network) -> float:
        """计算网络全局能量"""
        if network.number_of_edges() == 0:
            return 0
        energies = [network[u][v]['weight'] for u, v in network.edges()]
        return np.mean(energies)

    def compute_local_energy(self, node, network) -> float:
        """计算节点局部能量"""
        if node not in network:
            return 0

        neighbors = list(network.neighbors(node))
        if not neighbors:
            return 0

        local_energy = 0
        for neighbor in neighbors:
            local_energy += network[node][neighbor]['weight']
        return local_energy / len(neighbors)

    def generate_random_changes(self, network, num_changes=5) -> List[Dict]:
        """生成随机变化尝试"""
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

    def apply_change_and_compute(self, network, change) -> float:
        """应用变化并计算新能量"""
        # 保存原始状态
        original_state = {}

        if change['type'] == 'edge_weight_adjustment':
            u, v = change['edge']
            original_state[(u, v)] = network[u][v]['weight']
            network[u][v]['weight'] = change['new_weight']

        elif change['type'] == 'node_activation':
            node = change['node']
            # 节点激活会影响所有连接的边
            for neighbor in network.neighbors(node):
                original_state[(node, neighbor)] = network[node][neighbor]['weight']
                network[node][neighbor]['weight'] *= change['effect']

        elif change['type'] == 'local_optimization':
            center = change['center']
            for neighbor in change['neighbors']:
                if network.has_edge(center, neighbor):
                    original_state[(center, neighbor)] = network[center][neighbor]['weight']
                    network[center][neighbor]['weight'] *= change['optimization_strength']

        # 计算新能量
        new_energy = self.compute_global_energy(network)

        # 恢复原始状态
        for key, value in original_state.items():
            if isinstance(key, tuple) and len(key) == 2:
                u, v = key
                if network.has_edge(u, v):
                    network[u][v]['weight'] = value

        return new_energy

    def keep_change(self, network, change):
        """永久保留变化"""
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
    """认知宇宙 - 只实现两个公设，观察一切涌现"""

    def __init__(self, individual_params: Dict[str, Any] = None, network_seed: int = 42):
        if individual_params is None:
            individual_params = BASE_PARAMETERS.copy()

        self.G = nx.Graph()
        self.individual_params = individual_params
        self.network_seed = network_seed
        self.iteration_count = 0

        # 纯粹的能量动力学
        self.energy_dynamics = PureEnergyDynamics(individual_params)

        # 简化的状态管理（只记录，不主动控制）
        self.state_history = []
        self.current_energy_level = 1.0

        # 添加能量历史记录
        self.energy_history = []

        # 遍历历史记录
        self.traversal_history = []

        # 遗忘机制关键：记录每条边的最后激活时间
        self.last_activation_time = {}

        # 纯粹的观察记录（不是预设机制！）
        self.observations = {
            'spontaneous_compressions': [],  # 自发概念压缩
            'emergent_migrations': [],  # 涌现的原理迁移
            'traversal_patterns': [],  # 遍历模式分化
            'energy_minimization_traces': [],  # 能量最小化轨迹
            'network_evolution_snapshots': []  # 网络演化快照
        }

        # 基础参数
        self.learning_rate = individual_params.get('base_learning_rate', 0.85)
        self.forgetting_rate = individual_params.get('forgetting_rate', 0.002)
        self.base_forgetting_strength = 0.002

        # 设置随机种子
        random.seed(network_seed)
        np.random.seed(network_seed)
        print("认知宇宙初始化完成: 遗忘机制已激活")

    def initialize_semantic_network(self):
        """初始化语义网络"""
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
                    # 基于相似度设置初始能量（相似度越高，能量越低）
                    energy = 2.0 - similarity * 1.5
                    energy = max(0.3, min(2.0, energy))

                    initial_edges.append((node1, node2, {
                        'weight': energy,
                        'traversal_count': 0,
                        'original_weight': energy,
                        'similarity': similarity
                    }))
                    # 初始化激活时间为0
                    self.last_activation_time[(node1, node2)] = 0

        for u, v, attr in initial_edges:
            self.G.add_edge(u, v, **attr)

        print(f"语义网络初始化: {len(nodes)}节点, {len(initial_edges)}条边")
        print(f"初始全局能量: {self.calculate_network_energy():.3f}")

    def calculate_network_energy(self):
        """计算网络平均能耗"""
        if self.G.number_of_edges() == 0:
            return 0
        energies = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        return np.mean(energies)

    def basic_energy_optimization(self):
        """基本的能量优化 - 包含学习效应"""
        if self.G.number_of_edges() == 0:
            return False

        # 随机选择一条边进行"学习"
        edges = list(self.G.edges())
        u, v = random.choice(edges)

        # 记录激活并应用学习
        self.record_edge_activation(u, v)

        # 50%的概率也激活相邻的边（模拟概念扩散）
        if random.random() < 0.5:
            neighbors = list(self.G.neighbors(u))
            if neighbors:
                neighbor = random.choice(neighbors)
                if self.G.has_edge(u, neighbor):
                    self.record_edge_activation(u, neighbor)

        return True

    def record_edge_activation(self, u, v):
        """记录边的激活并应用学习效应"""
        # 更新激活时间
        self.last_activation_time[(u, v)] = self.iteration_count

        # 应用学习效应：降低这条边的能耗
        current_energy = self.G[u][v]['weight']
        learning_rate = 0.1  # 学习率
        new_energy = current_energy * (1 - learning_rate)
        self.G[u][v]['weight'] = max(0.05, new_energy)  # 最低能耗阈值

    def apply_basic_forgetting(self):
        """应用基础遗忘机制 - 关键修复部分"""
        current_time = self.iteration_count

        for u, v in self.G.edges():
            time_since_activation = current_time - self.last_activation_time.get((u, v), 0)

            if time_since_activation > 0:  # 只有长时间未激活的边才会遗忘
                current_energy = self.G[u][v]['weight']
                original_energy = self.G[u][v].get('original_weight', 2.0)

                # 遗忘函数：基于时间的指数衰减
                forget_factor = self._compute_forget_factor(time_since_activation)

                # 应用遗忘：能耗向原始值恢复
                new_energy = current_energy + (original_energy - current_energy) * forget_factor
                new_energy = min(new_energy, original_energy)  # 不超过原始值

                self.G[u][v]['weight'] = max(0.1, new_energy)  # 设置最低能耗阈值

    def _compute_forget_factor(self, time_gap):
        """计算遗忘因子"""
        # 基于指数衰减的遗忘函数
        base_rate = self.forgetting_rate

        # 时间间隔越长，遗忘越强
        time_factor = 1 - math.exp(-time_gap / 800)  # 调整分母改变遗忘速度

        # 综合遗忘因子
        forget_factor = base_rate * time_factor
        return min(forget_factor, 0.15)  # 最大遗忘率15%

    def _random_traversal(self):
        """随机遍历 - 最基本的认知活动"""
        nodes = list(self.G.nodes())
        if len(nodes) < 2:
            return

        start_node = random.choice(nodes)
        path = [start_node]
        current_node = start_node

        # 随机走2-4步
        path_length = random.randint(2, 4)

        for step in range(path_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break

            next_node = random.choice(neighbors)
            if next_node not in path:  # 避免循环
                path.append(next_node)
                current_node = next_node
            else:
                break

        if len(path) >= 2:
            # 记录遍历
            self.traversal_history.append({
                'path': path.copy(),
                'iteration': self.iteration_count,
                'type': 'random_traversal'
            })

            # 更新激活时间并应用学习
            current_time = self.iteration_count
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.G.has_edge(u, v):
                    self.record_edge_activation(u, v)

    def evolve(self, iterations: int = 1000, observation_interval: int = 100):
        """让宇宙自然演化，观察涌现现象"""
        print(f"开始认知宇宙演化: {iterations}次迭代")
        print("只执行基本能量优化，观察自然涌现...")

        initial_energy = self.calculate_network_energy()
        self.energy_history.append(initial_energy)  # 记录初始能量
        print(f"初始全局能量: {initial_energy:.3f}")

        for i in range(iterations):
            self.iteration_count += 1

            # 第一步：基本的能量优化（学习）
            self.basic_energy_optimization()

            # 第二步：随机遍历（另一种学习形式）
            if random.random() < 0.3:  # 30%的概率进行随机遍历
                self._random_traversal()

            # 第三步：应用遗忘机制
            if i % 10 == 0:  # 每10次迭代应用一次遗忘
                self.apply_basic_forgetting()

            # 记录当前能量
            current_energy = self.calculate_network_energy()
            self.energy_history.append(current_energy)

            # 第四步：定期记录网络快照
            if i % observation_interval == 0:
                self._take_network_snapshot()

            # 第五步：定期报告进度
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

    def _take_network_snapshot(self):
        """记录网络演化快照"""
        snapshot = {
            'iteration': self.iteration_count,
            'global_energy': self.calculate_network_energy(),
            'node_count': self.G.number_of_nodes(),
            'edge_count': self.G.number_of_edges(),
            'average_degree': np.mean([d for n, d in self.G.degree()]) if self.G.number_of_nodes() > 0 else 0,
            'clustering_coefficient': nx.average_clustering(self.G) if self.G.number_of_nodes() > 0 else 0
        }

        self.observations['network_evolution_snapshots'].append(snapshot)

    def get_emergence_metrics(self):
        """获取涌现现象的量化指标"""
        metrics = {
            'energy_minimization_efficiency': 0,
            'structural_emergence_index': 0,
            'cognitive_complexity': 0,
            'adaptation_rate': 0
        }

        if len(self.energy_dynamics.global_energy_history) > 1:
            # 能量最小化效率
            initial_energy = self.energy_dynamics.global_energy_history[0]
            final_energy = self.energy_dynamics.global_energy_history[-1]
            metrics['energy_minimization_efficiency'] = (
                                                                    initial_energy - final_energy) / initial_energy if initial_energy > 0 else 0

            # 适应率（能量下降的速率）
            energy_changes = np.diff(self.energy_dynamics.global_energy_history)
            negative_changes = [change for change in energy_changes if change < 0]
            metrics['adaptation_rate'] = len(negative_changes) / len(energy_changes) if energy_changes else 0

        # 结构涌现指数（基于网络统计）
        if len(self.observations['network_evolution_snapshots']) > 1:
            initial_clustering = self.observations['network_evolution_snapshots'][0]['clustering_coefficient']
            final_clustering = self.observations['network_evolution_snapshots'][-1]['clustering_coefficient']
            metrics['structural_emergence_index'] = final_clustering - initial_clustering

            # 认知复杂性（基于网络密度和聚类）
            density = self.G.number_of_edges() / (self.G.number_of_nodes() * (
                    self.G.number_of_nodes() - 1) / 2) if self.G.number_of_nodes() > 1 else 0
            metrics['cognitive_complexity'] = density * final_clustering

        return metrics

    def report_emergence_findings(self):
        """报告涌现观察结果"""
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

        # 分析能量最小化轨迹
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

    def get_network_stats(self):
        """获取网络统计信息"""
        return {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'iterations': self.iteration_count,
            'global_energy': self.calculate_network_energy(),
            'traversal_count': len(self.traversal_history),
            'snapshots_count': len(self.observations['network_evolution_snapshots'])
        }