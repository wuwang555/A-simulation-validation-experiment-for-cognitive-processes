"""
认知图基础模块
----------------
定义 BaseCognitiveGraph 类，实现认知网络的动态演化，包括遍历、遗忘、压缩、迁移等核心操作，
并整合认知状态管理器以模拟主观能耗变化。
"""

import networkx as nx
import numpy as np
import random
import math
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

from core.cognitive_states import CognitiveState, CognitiveStateManager

np.random.seed(42)
random.seed(42)

class BaseCognitiveGraph:
    """基础认知图类，实现能量动力学驱动的认知网络演化。

    该类封装了认知图的核心数据结构（无向图）及操作方法。网络中的边权重表示认知能耗，
    通过遍历、遗忘、压缩、迁移等操作实现全局能耗的最小化。同时集成了认知状态管理器，
    模拟主观认知状态（专注、探索、疲劳、灵感）对行为策略的影响。

    :param individual_params: 个体参数字典，包含遗忘率、学习率、各类偏置等。
    :param network_seed: 随机种子，用于结果可重复。
    """

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        self.G = nx.Graph()
        self.traversal_history = []          # 记录遍历路径及类型
        self.concept_centers = {}             # 记录已压缩的概念中心
        self.iteration_count = 0
        self.energy_history = []              # 网络平均能耗历史

        # 状态管理
        self.state_manager = CognitiveStateManager()

        # 个体参数
        self.individual_params = individual_params
        self._setup_parameters(individual_params)

        self.last_activation_time = {}        # 记录每条边最后被激活的时刻
        self.network_seed = network_seed

    def _setup_parameters(self, individual_params: Dict[str, Any]) -> None:
        """从参数字典中解析个体认知参数。

        :param individual_params: 包含以下键值的字典：
            - forgetting_rate: 遗忘率
            - base_learning_rate: 基础学习率
            - hard_traversal_bias: 硬遍历偏置
            - soft_traversal_bias: 软遍历偏置
            - compression_bias: 压缩偏置
            - migration_bias: 迁移偏置
            - learning_rate_variation: 学习率变异系数
        """
        self.forgetting_rate = individual_params.get('forgetting_rate', 0.002)
        self.base_learning_rate = individual_params.get('base_learning_rate', 0.85)
        self.hard_traversal_bias = individual_params.get('hard_traversal_bias', 0.0)
        self.soft_traversal_bias = individual_params.get('soft_traversal_bias', 0.0)
        self.compression_bias = individual_params.get('compression_bias', 0.0)
        self.migration_bias = individual_params.get('migration_bias', 0.0)
        self.learning_rate_variation = individual_params.get('learning_rate_variation', 0.1)

        # 硬遍历和软遍历的能耗分配策略
        self.hard_traversal_energy_ratio = 0.6
        self.soft_traversal_energy_ratio = 0.4

    @property
    def current_state(self) -> CognitiveState:
        """当前认知状态（枚举值）。"""
        return self.state_manager.current_state

    @property
    def subjective_energy(self) -> float:
        """主观认知能耗，反映个体当前的认知资源水平。"""
        return self.state_manager.subjective_energy

    @property
    def cognitive_energy_history(self) -> List[Dict]:
        """认知状态变化历史记录。"""
        return self.state_manager.cognitive_energy_history

    def update_cognitive_state(self) -> None:
        """根据状态转移矩阵更新认知状态，并记录历史。"""
        self.state_manager.update_cognitive_state()

    def _update_subjective_energy(self) -> None:
        """根据当前状态更新主观能耗值。"""
        self.state_manager._update_subjective_energy()

    def can_traverse_edge(self, edge_energy: float, traversal_type: str) -> Tuple[bool, float]:
        """判断在当前主观能耗下是否能够遍历某条边。

        :param edge_energy: 边当前的能耗（权重）
        :param traversal_type: 遍历类型，'hard' 或 'soft'
        :return: (是否可遍历, 剩余能量余额)
        """
        if traversal_type == "hard":
            required_energy = edge_energy * 0.8
            available_energy = self.subjective_energy * self.hard_traversal_energy_ratio
        else:
            required_energy = edge_energy * 0.6
            available_energy = self.subjective_energy * self.soft_traversal_energy_ratio

        return available_energy >= required_energy, available_energy - required_energy

    def traverse_path(self, path: List[str], traversal_type: str = "hard") -> None:
        """沿着指定路径执行遍历，更新边权重（学习效应），并记录历史。

        遍历会降低路径上各边的能耗，学习率受遍历类型、节点相似度及个体差异共同影响。
        遍历后根据能量余额调整认知状态。

        :param path: 节点列表，表示遍历路径
        :param traversal_type: 遍历类型，'hard' 或 'soft'
        """
        # 随机更新认知状态
        if random.random() < 0.1:
            self.update_cognitive_state()

        total_required_energy = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                total_required_energy += self.G[u][v]['weight']

        can_traverse, energy_balance = self.can_traverse_edge(total_required_energy, traversal_type)

        # 小概率强行遍历（模拟认知资源透支）
        if not can_traverse:
            if random.random() < 0.2 and self.current_state != CognitiveState.FATIGUED:
                can_traverse = True
                energy_balance = -0.5

        if not can_traverse:
            if random.random() < 0.3:
                self.state_manager.current_state = CognitiveState.FATIGUED
                self._update_subjective_energy()
            return

        self.traversal_history.append((path, traversal_type, self.iteration_count))
        current_time = self.iteration_count

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                self.last_activation_time[(u, v)] = current_time
                if 'traversal_count' not in self.G[u][v]:
                    self.G[u][v]['traversal_count'] = 0
                self.G[u][v]['traversal_count'] += 1

                similarity = 0.5   # 默认相似度，实际应由语义网络提供
                base_rate = self.base_learning_rate

                individual_learning_variation = np.random.uniform(
                    1 - self.learning_rate_variation,
                    1 + self.learning_rate_variation
                )

                if traversal_type == "hard":
                    learning_rate = base_rate * (0.7 + 0.3 * similarity) * individual_learning_variation
                else:
                    learning_rate = base_rate * 0.9 * individual_learning_variation

                current_weight = self.G[u][v]['weight']
                learning_effect = learning_rate * (current_weight / 2.0)

                new_weight = max(0.05, current_weight * (1 - learning_effect))
                self.G[u][v]['weight'] = new_weight

        self._post_traversal_state_update(traversal_type, energy_balance)

    def _post_traversal_state_update(self, traversal_type: str, energy_balance: float) -> None:
        """遍历后根据能量余额更新认知状态。

        :param traversal_type: 遍历类型
        :param energy_balance: 遍历后剩余能量（可正可负）
        """
        if energy_balance > 0.3:
            if traversal_type == "hard" and random.random() < 0.4:
                self.state_manager.current_state = CognitiveState.FOCUSED
            elif traversal_type == "soft" and random.random() < 0.3:
                self.state_manager.current_state = CognitiveState.EXPLORATORY
        elif energy_balance < -0.2:
            if random.random() < 0.5:
                self.state_manager.current_state = CognitiveState.FATIGUED

        self._update_subjective_energy()

    def _apply_forgetting(self) -> None:
        """应用遗忘机制，增加长时间未激活边的能耗（向原始权重恢复）。"""
        current_time = self.iteration_count

        for u, v in self.G.edges():
            time_since_last_activation = current_time - self.last_activation_time.get((u, v), 0)
            if time_since_last_activation > 0:
                current_energy = self.G[u][v]['weight']
                similarity = 0.5

                forget_factor = self.forgetting_function(
                    current_time,
                    self.last_activation_time.get((u, v), 0),
                    current_energy,
                    similarity
                )

                new_weight = self.G[u][v]['weight'] * (1 + forget_factor)
                original = self.G[u][v].get('original_weight', 2.0)
                self.G[u][v]['weight'] = min(new_weight, original)

    def forgetting_function(self, current_time: int, last_activation_time: int,
                            current_energy: float, similarity: float) -> float:
        """计算遗忘因子，基于指数衰减模型。

        .. math::
            forget\_factor = (1 - e^{-\\Delta t / 500}) \\times
            (0.5 + 0.5 \\cdot \\frac{E}{2.0}) \\times (1 - 0.5 \\cdot sim) \\times forgetting\_rate

        :param current_time: 当前迭代次数
        :param last_activation_time: 上次激活时间
        :param current_energy: 边当前能耗
        :param similarity: 两端节点的语义相似度
        :return: 遗忘因子（0~0.1之间）
        """
        time_gap = current_time - last_activation_time

        base_forgetting = 1 - math.exp(-time_gap / 500)
        energy_factor = 0.5 + 0.5 * (current_energy / 2.0)
        similarity_protection = 1 - (similarity * 0.5)

        forgetting_factor = (base_forgetting * energy_factor *
                             similarity_protection * self.forgetting_rate)

        return min(forgetting_factor, 0.1)

    def monte_carlo_iteration(self, max_iterations: int = 10000) -> None:
        """执行蒙特卡洛模拟，迭代演化认知网络。

        每步迭代包括：
            1. 定期更新认知状态
            2. 应用遗忘机制
            3. 根据当前状态选择操作（硬遍历、软遍历、压缩、迁移）
            4. 记录网络能耗历史

        :param max_iterations: 最大迭代次数
        """
        print(f"初始认知状态: {self.current_state.value}, 主观能耗: {self.subjective_energy:.2f}")

        for iteration in range(max_iterations):
            self.iteration_count += 1

            if iteration % 100 == 0:
                self.update_cognitive_state()

            self._apply_forgetting()

            current_avg_energy = self.calculate_network_energy()
            self.energy_history.append(current_avg_energy)

            operation = self._select_operation_based_on_state()

            if operation == "hard_traversal":
                self._state_based_hard_traversal()
            elif operation == "soft_traversal":
                self._state_based_soft_traversal()
            elif operation == "compression":
                self._random_compression()
            elif operation == "migration":
                self._random_migration()

            if iteration % 500 == 0:
                stats = self.get_network_stats()
                print(f"迭代 {iteration}, 状态: {self.current_state.value}, "
                      f"主观能耗: {self.subjective_energy:.2f}, 网络能耗: {current_avg_energy:.3f}")

    def _select_operation_based_on_state(self) -> str:
        """根据当前认知状态选择下一步操作类型。

        状态-操作概率映射由预定义字典决定，不同状态下各操作的概率不同。

        :return: 操作名称字符串
        """
        state_operations = {
            CognitiveState.FOCUSED: {
                "hard_traversal": 0.5,
                "soft_traversal": 0.3,
                "compression": 0.1,
                "migration": 0.1
            },
            CognitiveState.EXPLORATORY: {
                "hard_traversal": 0.3,
                "soft_traversal": 0.4,
                "compression": 0.1,
                "migration": 0.2
            },
            CognitiveState.FATIGUED: {
                "hard_traversal": 0.2,
                "soft_traversal": 0.4,
                "compression": 0.2,
                "migration": 0.2
            },
            CognitiveState.INSPIRED: {
                "hard_traversal": 0.3,
                "soft_traversal": 0.4,
                "compression": 0.1,
                "migration": 0.2
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

    def _state_based_hard_traversal(self) -> None:
        """基于当前状态发起硬遍历：寻找并遍历一条低能耗路径。"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return
        start_node = random.choice(available_nodes)
        path = self._find_hard_traversal_path(start_node, 3)
        if path and len(path) >= 2:
            self.traverse_path(path, "hard")

    def _state_based_soft_traversal(self) -> None:
        """基于当前状态发起软遍历：随机游走探索新路径。"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return
        start_node = random.choice(available_nodes)
        path = self._find_soft_traversal_path(start_node, 2)
        if path and len(path) >= 2:
            self.traverse_path(path, "soft")

    def _find_hard_traversal_path(self, start_node: str, max_length: int) -> Optional[List[str]]:
        """寻找硬遍历路径：优先选择能耗较低的边，且要求能量允许。

        :param start_node: 起始节点
        :param max_length: 最大路径长度（节点数）
        :return: 节点列表（路径）或 None
        """
        path = [start_node]
        current_node = start_node

        for _ in range(max_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break

            # 按能耗升序排列（低能耗优先）
            neighbors.sort(key=lambda n: self.G[current_node][n]['weight'])

            found_next = False
            for neighbor in neighbors[:3]:
                if neighbor not in path:
                    edge_energy = self.G[current_node][neighbor]['weight']
                    can_traverse, _ = self.can_traverse_edge(edge_energy, "hard")
                    if can_traverse:
                        path.append(neighbor)
                        current_node = neighbor
                        found_next = True
                        break

            if not found_next:
                break

        return path if len(path) >= 2 else None

    def _find_soft_traversal_path(self, start_node: str, max_length: int) -> Optional[List[str]]:
        """寻找软遍历路径：随机选择邻居，但受能量约束。

        :param start_node: 起始节点
        :param max_length: 最大路径长度
        :return: 节点列表（路径）或 None
        """
        path = [start_node]
        current_node = start_node

        for _ in range(max_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break

            random.shuffle(neighbors)

            found_next = False
            for neighbor in neighbors:
                if neighbor not in path:
                    edge_energy = self.G[current_node][neighbor]['weight']
                    can_traverse, _ = self.can_traverse_edge(edge_energy, "soft")
                    if can_traverse:
                        path.append(neighbor)
                        current_node = neighbor
                        found_next = True
                        break

            if not found_next:
                break

        return path if len(path) >= 2 else None

    def _random_compression(self) -> None:
        """随机尝试概念压缩：选择中心节点，若其强连接邻居足够则进行压缩。"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 3:
            return

        # 低概率触发
        if random.random() > 0.10:
            return

        if self.iteration_count < 2000:
            return

        center_candidate = random.choice(available_nodes)

        good_neighbors = []
        for neighbor in self.G.neighbors(center_candidate):
            if (self.G[center_candidate][neighbor]['weight'] < 1.0 and
                self.calculate_semantic_similarity(center_candidate, neighbor) > 0.4):
                good_neighbors.append(neighbor)

        if len(good_neighbors) >= 3:
            num_to_compress = random.randint(2, min(3, len(good_neighbors)))
            nodes_to_compress = random.sample(good_neighbors, num_to_compress)

            compression_strength = random.uniform(0.4, 0.6)
            self.conceptual_compression(center_candidate, nodes_to_compress, compression_strength)

    def conceptual_compression(self, center_node: str, related_nodes: List[str],
                               compression_strength: float = 0.5) -> bool:
        """执行概念压缩：强化中心节点与相关节点的连接（降低能耗），封装微观结构。

        压缩后，中心节点与相关节点的边能耗降低，形成一个宏观概念节点，内部结构被封装。

        :param center_node: 压缩中心节点
        :param related_nodes: 相关节点列表
        :param compression_strength: 压缩强度因子（0~1），越小压缩越强
        :return: 是否成功执行压缩
        """
        if len(related_nodes) < 2:
            return False

        for node in related_nodes:
            if self.G.has_edge(center_node, node):
                current_energy = self.G[center_node][node]['weight']
                compressed_energy = max(0.05, current_energy * compression_strength)
                self.G[center_node][node]['weight'] = compressed_energy

        self.concept_centers[center_node] = {
            'related_nodes': related_nodes,
            'compression_strength': compression_strength,
            'iteration': self.iteration_count
        }

        return True

    def _random_migration(self) -> None:
        """随机尝试第一性原理迁移：寻找两个节点间通过原理节点的低能耗路径。"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 4:
            return

        if random.random() > 0.05:
            return

        start_node, end_node = random.sample(available_nodes, 2)

        principle_candidates = [n for n in available_nodes
                                if n not in [start_node, end_node]]

        if not principle_candidates:
            return

        num_principles = random.randint(1, min(2, len(principle_candidates)))
        selected_principles = random.sample(principle_candidates, num_principles)

        exploration_bonus = random.uniform(0.05, 0.15)
        self.first_principles_migration(start_node, end_node, selected_principles, exploration_bonus)

    def first_principles_migration(self, start_node: str, end_node: str,
                                   principle_nodes: List[str], exploration_bonus: float = 0.1) -> Optional[List[str]]:
        """第一性原理迁移：通过原理节点寻找比直接连接更节能的间接路径。

        迁移的条件是新路径的总能耗比直接连接低至少 improvement_threshold。

        :param start_node: 起始节点
        :param end_node: 目标节点
        :param principle_nodes: 候选原理节点列表
        :param exploration_bonus: 探索奖励，降低路径能耗以鼓励新发现
        :return: 迁移路径（包含原理节点）或 None
        """
        best_path = None
        best_energy = float('inf')

        # 直接连接能耗
        direct_energy = float('inf')
        if self.G.has_edge(start_node, end_node):
            direct_energy = self.G[start_node][end_node]['weight']
            best_path = [start_node, end_node]
            best_energy = direct_energy

        # 尝试通过每个原理节点
        for principle in principle_nodes:
            if (self.G.has_edge(start_node, principle) and
                    self.G.has_edge(principle, end_node)):

                path_energy = (self.G[start_node][principle]['weight'] +
                               self.G[principle][end_node]['weight'])

                # 应用探索奖励
                adjusted_energy = path_energy - exploration_bonus

                if adjusted_energy < best_energy:
                    best_energy = adjusted_energy
                    best_path = [start_node, principle, end_node]

        # 要求新路径必须明显优于直接路径
        improvement_threshold = 0.2
        if (best_path and len(best_path) > 2 and
                best_energy < direct_energy * (1 - improvement_threshold)):

            # 强化迁移路径上的连接
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                current = self.G[u][v]['weight']
                new_energy = max(0.05, current * random.uniform(0.6, 0.8))
                self.G[u][v]['weight'] = new_energy

            # 记录迁移关系
            principle_node = best_path[1]
            if 'migration_bridges' not in self.G.nodes[principle_node]:
                self.G.nodes[principle_node]['migration_bridges'] = []

            self.G.nodes[principle_node]['migration_bridges'].append({
                'from': start_node,
                'to': end_node,
                'energy_saving': direct_energy - best_energy,
                'iteration': self.iteration_count
            })

            # 模拟遍历这条新发现的优化路径
            self.traverse_path(best_path, traversal_type="soft")

            return best_path

        return None

    def get_network_stats(self) -> Dict[str, Any]:
        """获取当前网络的统计信息。

        :return: 包含节点数、边数、迭代次数、平均能耗、压缩中心数、迁移桥梁数的字典
        """
        stats = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'iterations': self.iteration_count,
            'avg_energy': self.calculate_network_energy(),
            'compression_centers': len(self.concept_centers),
            'migration_bridges': 0
        }

        for node in self.G.nodes():
            if 'migration_bridges' in self.G.nodes[node]:
                stats['migration_bridges'] += len(self.G.nodes[node]['migration_bridges'])

        return stats

    def calculate_network_energy(self) -> float:
        """计算当前网络的平均能耗（所有边权重的均值）。"""
        if self.G.number_of_edges() == 0:
            return 0
        energies = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        return np.mean(energies)

    def calculate_semantic_similarity(self, node1: str, node2: str) -> float:
        """计算两个节点间的语义相似度，默认返回0.5，子类应重写该方法以利用真实语义信息。"""
        return 0.5

    def visualize_energy_convergence(self) -> None:
        """可视化能耗收敛过程（需安装matplotlib并实现相应函数）。"""
        from utils.visualization import visualize_energy_convergence
        visualize_energy_convergence(self.energy_history, self.concept_centers)

    def visualize_cognitive_states(self) -> None:
        """可视化认知状态变化历史。"""
        from utils.visualization import visualize_cognitive_states
        for i, entry in enumerate(self.cognitive_energy_history):
            if 'iteration' not in entry:
                entry['iteration'] = i
        visualize_cognitive_states(self.cognitive_energy_history, self.energy_history)

    def visualize_graph(self, title: str = "认知图", figsize: Tuple[int, int] = (12, 8)) -> None:
        """可视化当前认知图结构。"""
        from utils.visualization import visualize_graph
        visualize_graph(self.G, self.concept_centers, title, figsize)


if __name__ == "__main__":
    params = {
        'forgetting_rate': 0.002,
        'base_learning_rate': 0.85,
        'hard_traversal_bias': 0.1,
        'soft_traversal_bias': 0.1,
        'compression_bias': 0.05,
        'migration_bias': 0.05,
        'learning_rate_variation': 0.1
    }
    cg = BaseCognitiveGraph(params)
    # 可添加更多测试逻辑...
    print("BaseCognitiveGraph 初始化成功")