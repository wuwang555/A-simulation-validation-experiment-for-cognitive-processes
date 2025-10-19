import networkx as nx
import numpy as np
import random
import math
from collections import defaultdict
from typing import Dict, Any

from core.cognitive_states import CognitiveState, CognitiveStateManager

class BaseCognitiveGraph:
    """基础认知图类"""

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        self.G = nx.Graph()
        self.traversal_history = []
        self.concept_centers = {}
        self.iteration_count = 0
        self.energy_history = []

        # 状态管理 - 现在通过state_manager访问状态相关属性
        self.state_manager = CognitiveStateManager()

        # 个体参数
        self.individual_params = individual_params
        self._setup_parameters(individual_params)

        self.last_activation_time = {}
        self.network_seed = network_seed

    def _setup_parameters(self, individual_params):
        """设置个体参数"""
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

    # 添加属性访问器，保持向后兼容
    @property
    def current_state(self):
        """当前认知状态"""
        return self.state_manager.current_state

    @property
    def subjective_energy(self):
        """主观认知能耗"""
        return self.state_manager.subjective_energy

    @property
    def cognitive_energy_history(self):
        """认知能量历史"""
        return self.state_manager.cognitive_energy_history

    def update_cognitive_state(self):
        """更新认知状态"""
        self.state_manager.update_cognitive_state()

    def _update_subjective_energy(self):
        """更新主观能耗"""
        self.state_manager._update_subjective_energy()

    def can_traverse_edge(self, edge_energy, traversal_type):
        """检查是否可以遍历某条边（考虑主观认知能耗）"""
        if traversal_type == "hard":
            required_energy = edge_energy * 0.8
            available_energy = self.subjective_energy * self.hard_traversal_energy_ratio
        else:
            required_energy = edge_energy * 0.6
            available_energy = self.subjective_energy * self.soft_traversal_energy_ratio

        return available_energy >= required_energy, available_energy - required_energy

    def traverse_path(self, path, traversal_type="hard"):
        """改进的遍历函数 - 考虑主观认知状态"""
        if random.random() < 0.1:
            self.update_cognitive_state()

        total_required_energy = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                edge_energy = self.G[u][v]['weight']
                total_required_energy += edge_energy

        can_traverse, energy_balance = self.can_traverse_edge(total_required_energy, traversal_type)

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

                similarity = 0.5
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

    def _post_traversal_state_update(self, traversal_type, energy_balance):
        """遍历后的状态更新"""
        if energy_balance > 0.3:
            if traversal_type == "hard" and random.random() < 0.4:
                self.state_manager.current_state = CognitiveState.FOCUSED
            elif traversal_type == "soft" and random.random() < 0.3:
                self.state_manager.current_state = CognitiveState.EXPLORATORY
        elif energy_balance < -0.2:
            if random.random() < 0.5:
                self.state_manager.current_state = CognitiveState.FATIGUED

        self._update_subjective_energy()

    def _apply_forgetting(self):
        """应用遗忘机制到所有边"""
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

    def forgetting_function(self, current_time, last_activation_time, current_energy, similarity):
        """基于指数衰减的遗忘时间函数"""
        time_gap = current_time - last_activation_time

        base_forgetting = 1 - math.exp(-time_gap / 500)
        energy_factor = 0.5 + 0.5 * (current_energy / 2.0)
        similarity_protection = 1 - (similarity * 0.5)

        forgetting_factor = (base_forgetting * energy_factor *
                             similarity_protection * self.forgetting_rate)

        return min(forgetting_factor, 0.1)

    def monte_carlo_iteration(self, max_iterations=5000):
        """改进的蒙特卡洛模拟 - 考虑主观认知状态"""
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

    def _select_operation_based_on_state(self):
        """基于认知状态选择操作类型"""
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

    def _state_based_hard_traversal(self):
        """基于状态的硬遍历"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return

        start_node = random.choice(available_nodes)
        path = self._find_hard_traversal_path(start_node, 3)

        if path and len(path) >= 2:
            self.traverse_path(path, "hard")

    def _state_based_soft_traversal(self):
        """基于状态的软遍历"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return

        start_node = random.choice(available_nodes)
        path = self._find_soft_traversal_path(start_node, 2)

        if path and len(path) >= 2:
            self.traverse_path(path, "soft")

    def _find_hard_traversal_path(self, start_node, max_length):
        """硬遍历路径搜索"""
        path = [start_node]
        current_node = start_node

        for _ in range(max_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break

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

    def _find_soft_traversal_path(self, start_node, max_length):
        """软遍历路径搜索"""
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

    def _random_compression(self):
        """随机概念压缩尝试"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 3:
            return

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

    def conceptual_compression(self, center_node, related_nodes, compression_strength=0.5):
        """概念压缩：强化中心节点与相关节点的连接"""
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

    def _random_migration(self):
        """随机第一性原理迁移尝试"""
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

    def first_principles_migration(self, start_node, end_node, principle_nodes, exploration_bonus=0.1):
        """第一性原理迁移 - 修复版本"""
        best_path = None
        best_energy = float('inf')

        # 修复：确保direct_energy始终有值
        direct_energy = float('inf')
        if self.G.has_edge(start_node, end_node):
            direct_energy = self.G[start_node][end_node]['weight']
            best_path = [start_node, end_node]
            best_energy = direct_energy

        # 尝试通过每个原理节点建立连接
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

    def get_network_stats(self):
        """获取网络统计信息"""
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

    def calculate_network_energy(self):
        """计算网络平均能耗"""
        if self.G.number_of_edges() == 0:
            return 0
        energies = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        return np.mean(energies)

    def calculate_semantic_similarity(self, node1, node2):
        """计算语义相似度 - 需要在子类中实现"""
        return 0.5  # 默认实现，子类应该重写

    def visualize_energy_convergence(self):
        """可视化能耗收敛过程"""
        from utils.visualization import visualize_energy_convergence
        visualize_energy_convergence(self.energy_history, self.concept_centers)

    def visualize_cognitive_states(self):
        """可视化认知状态变化"""
        from utils.visualization import visualize_cognitive_states
        # 确保认知能量历史中有迭代信息
        for i, entry in enumerate(self.cognitive_energy_history):
            if 'iteration' not in entry:
                entry['iteration'] = i
        visualize_cognitive_states(self.cognitive_energy_history, self.energy_history)

    def visualize_graph(self, title="认知图", figsize=(12, 8)):
        """可视化认知图"""
        from utils.visualization import visualize_graph
        visualize_graph(self.G, self.concept_centers, title, figsize)