import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from core.cognitive_states import CognitiveState
import random
import math


class EmergenceDetector:
    """涌现现象检测器 - 被动观察自然涌现的认知现象"""

    def __init__(self, detection_thresholds: Dict[str, float] = None):
        # 检测阈值配置
        self.thresholds = detection_thresholds or {
            'compression_synergy': 0.75,  # 提高压缩协同性阈值
            'migration_efficiency': 0.4,  # 提高迁移效率阈值
            'pattern_stability': 0.8,  # 提高模式稳定性阈值
            'energy_sync_threshold': 0.8,  # 提高能量同步阈值
            'cross_domain_gain': 0.3,  # 提高跨领域增益阈值
            'cluster_cohesion': 0.7,  # 提高集群内聚性阈值
            'min_cluster_size': 3,  # 最小集群大小
            'max_cluster_size': 8,  # 最大集群大小
        }

        # 检测历史
        self.detection_history = {
            'compressions': [],
            'migrations': [],
            'patterns': []
        }

    def detect_spontaneous_compression(self, network, energy_history, traversal_history=None):
        """检测自然发生的概念压缩 - 修复版本"""
        compression_candidates = []
        processed_nodes = set()

        # 如果能量历史太短，返回空列表
        if len(energy_history) < 50:
            return compression_candidates

        for node in network.nodes():
            if node in processed_nodes:
                continue

            # 获取邻居节点
            neighbors = list(network.neighbors(node))
            min_size = self.thresholds['min_cluster_size']
            max_size = self.thresholds['max_cluster_size']

            if len(neighbors) < min_size:
                continue

            # 计算能量协同性
            energy_sync = self._analyze_energy_synchronization(node, neighbors, network, energy_history)

            # 使用提高的阈值
            if energy_sync > self.thresholds['energy_sync_threshold']:
                # 限制压缩组的大小
                if len(neighbors) > max_size:
                    # 选择连接最强的邻居
                    neighbor_strength = []
                    for neighbor in neighbors:
                        if network.has_edge(node, neighbor):
                            strength = network[node][neighbor]['weight']
                            neighbor_strength.append((neighbor, strength))

                    neighbor_strength.sort(key=lambda x: x[1])
                    selected_neighbors = [n for n, strength in neighbor_strength[:max_size]]
                else:
                    selected_neighbors = neighbors

                # 计算集群内聚性
                cohesion = self._compute_cluster_cohesion(node, selected_neighbors, network)

                # 计算涌现强度
                emergence_strength = self._compute_emergence_strength(node, selected_neighbors, network)

                compression_candidates.append({
                    'center': node,
                    'related_nodes': selected_neighbors,
                    'energy_synergy': energy_sync,
                    'cohesion': cohesion,
                    'emergence_strength': emergence_strength,
                    'cluster_size': len(selected_neighbors)
                })

                processed_nodes.update(selected_neighbors)
                processed_nodes.add(node)

        return compression_candidates

    def detect_emergent_migration(self, network: nx.Graph, traversal_history: List[Any] = None,
                                  current_iteration: int = 0, semantic_network=None) -> List[Dict[str, Any]]:
        """
        检测自然涌现的第一性原理迁移 - 修复版本
        """
        migrations = []

        if traversal_history is None or len(traversal_history) < 5:
            return migrations

        # 分析最近的遍历模式
        recent_traversals = traversal_history[-20:]  # 分析最近20次遍历

        for i in range(1, len(recent_traversals)):
            prev_traversal = recent_traversals[i - 1]
            curr_traversal = recent_traversals[i]

            # 提取路径信息
            prev_path = self._extract_path(prev_traversal)
            curr_path = self._extract_path(curr_traversal)

            if not prev_path or not curr_path or len(prev_path) < 2 or len(curr_path) < 2:
                continue

            # 检查是否出现了新的跨领域连接
            if self._is_innovative_cross_domain_path(prev_path, curr_path, network):
                mediator = self._find_mediator_node(curr_path)
                efficiency_gain = self._calculate_path_efficiency_gain(prev_path, curr_path, network)

                if efficiency_gain > self.thresholds['migration_efficiency']:
                    migration_event = {
                        'type': 'first_principles_migration',
                        'principle_node': mediator,
                        'from_node': curr_path[0],
                        'to_node': curr_path[-1],
                        'efficiency_gain': efficiency_gain,
                        'path': curr_path,
                        'emergence_iteration': current_iteration - (len(recent_traversals) - i),
                        'domain_span': self._calculate_domain_span(curr_path)
                    }
                    migrations.append(migration_event)

        return migrations

    def _extract_path(self, traversal):
        """从遍历记录中提取路径"""
        if isinstance(traversal, (list, tuple)) and len(traversal) > 0:
            if isinstance(traversal[0], list):
                return traversal[0]
            elif len(traversal) >= 3:  # 格式: (path, type, iteration)
                return traversal[0] if isinstance(traversal[0], list) else [traversal[0]]
        return None

    def _is_innovative_cross_domain_path(self, prev_path, curr_path, network):
        """检查是否是创新的跨领域路径"""
        if not prev_path or not curr_path:
            return False

        # 检查是否跨领域
        start_domain = self._infer_domain(curr_path[0])
        end_domain = self._infer_domain(curr_path[-1])

        if start_domain == end_domain:
            return False

        # 检查路径的新颖性
        if prev_path and set(curr_path) == set(prev_path):
            return False

        # 检查是否通过原理性节点连接
        principle_nodes = [node for node in curr_path if self._is_principle_node(node)]
        if not principle_nodes:
            return False

        return True

    def _find_mediator_node(self, path):
        """在路径中寻找中介节点"""
        if len(path) < 3:
            return path[0] if path else ""

        # 寻找原理性节点作为中介
        for node in path[1:-1]:
            if self._is_principle_node(node):
                return node

        # 如果没有原理节点，返回中间的节点
        return path[len(path) // 2]

    def _calculate_path_efficiency_gain(self, prev_path, curr_path, network):
        """计算路径效率增益"""
        if not prev_path or not curr_path:
            return 0.0

        # 计算路径能耗
        prev_energy = self._calculate_path_energy(prev_path, network)
        curr_energy = self._calculate_path_energy(curr_path, network)

        if prev_energy == 0:
            return 0.0

        gain = (prev_energy - curr_energy) / prev_energy
        return max(0.0, gain)

    def _calculate_path_energy(self, path, network):
        """计算路径总能耗"""
        total_energy = 0.0
        for i in range(len(path) - 1):
            if network.has_edge(path[i], path[i + 1]):
                total_energy += network[path[i]][path[i + 1]]['weight']
        return total_energy

    def _calculate_domain_span(self, path):
        """计算路径的领域跨度"""
        if not path:
            return 0

        domains = set()
        for node in path:
            domains.add(self._infer_domain(node))

        return len(domains)

    def _is_principle_node(self, node):
        """判断是否是原理性节点"""
        principle_keywords = ['优化', '变换', '抽象', '模式', '递归', '迭代', '对称', '归纳']
        return any(keyword in node for keyword in principle_keywords)

    def _analyze_energy_synchronization(self, center, neighbors, network, energy_history):
        """分析能量同步性 - 实际实现版本"""
        if len(neighbors) < 2:
            return 0.0

        # 计算中心节点与邻居之间的能量变化相关性
        center_energies = []
        neighbor_energies = []

        # 获取当前能量状态
        for neighbor in neighbors:
            if network.has_edge(center, neighbor):
                center_energy = network[center][neighbor]['weight']
                center_energies.append(center_energy)

                # 计算该邻居与其他邻居的平均能量
                neighbor_avg = self._calculate_neighbor_energy(neighbor, neighbors, network)
                neighbor_energies.append(neighbor_avg)

        if len(center_energies) < 2 or len(neighbor_energies) < 2:
            return 0.5  # 默认值

        # 计算相关性作为同步性指标
        try:
            correlation = np.corrcoef(center_energies, neighbor_energies)[0, 1]
            if np.isnan(correlation):
                return 0.5
            return abs(correlation)  # 取绝对值，我们关心变化的相关性
        except:
            return 0.5

    def _calculate_neighbor_energy(self, node, neighbors, network):
        """计算节点在邻居集合中的平均连接能量"""
        energies = []
        for neighbor in neighbors:
            if neighbor != node and network.has_edge(node, neighbor):
                energies.append(network[node][neighbor]['weight'])

        return np.mean(energies) if energies else 1.0

    def _compute_cluster_cohesion(self, center: str, neighbors: List[str],
                                  network: nx.Graph) -> float:
        """计算集群的内聚性"""
        if len(neighbors) < 2:
            return 1.0  # 只有一个邻居时，内聚性为1

        # 计算邻居之间的连接密度
        total_possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        if total_possible_edges == 0:
            return 1.0

        actual_edges = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                if network.has_edge(n1, n2):
                    actual_edges += 1

        cohesion = actual_edges / total_possible_edges
        return cohesion

    def _compute_emergence_strength(self, center, neighbors, network):
        """计算涌现强度"""
        if not neighbors:
            return 0.0

        # 基于连接强度和集群内聚性计算涌现强度
        connection_strengths = []
        for neighbor in neighbors:
            if network.has_edge(center, neighbor):
                # 能量越低，连接越强
                energy = network[center][neighbor]['weight']
                strength = 1.0 / (1.0 + energy)  # 能量越低，强度越高
                connection_strengths.append(strength)

        avg_strength = np.mean(connection_strengths) if connection_strengths else 0
        cohesion = self._compute_cluster_cohesion(center, neighbors, network)

        # 综合强度计算
        emergence_strength = 0.6 * avg_strength + 0.4 * cohesion
        return min(1.0, emergence_strength)

    def detect_cognitive_patterns(self, cognitive_history: List[Dict],
                                  energy_history: List[float]) -> List[Dict[str, Any]]:
        """
        检测认知模式的自然涌现
        """
        if len(cognitive_history) < 50:
            return []

        patterns = []
        state_energy_patterns = self._detect_state_energy_correlations(cognitive_history, energy_history)

        for pattern in state_energy_patterns:
            if pattern['stability'] > self.thresholds['pattern_stability']:
                pattern_event = {
                    'type': 'cognitive_pattern',
                    'state_transition': pattern['transition'],
                    'energy_impact': pattern['energy_impact'],
                    'stability_score': pattern['stability'],
                    'frequency': pattern['frequency'],
                    'detection_window': len(cognitive_history)
                }
                patterns.append(pattern_event)

        return patterns

    def _detect_state_energy_correlations(self, cognitive_history: List[Dict],
                                          energy_history: List[float]) -> List[Dict[str, Any]]:
        """检测认知状态与能量的关联模式"""
        patterns = []

        if len(cognitive_history) < 10:
            return patterns

        state_transitions = defaultdict(list)

        for i in range(1, len(cognitive_history)):
            if i >= len(energy_history):
                break

            prev_entry = cognitive_history[i - 1]
            curr_entry = cognitive_history[i]

            prev_state = prev_entry.get('state')
            curr_state = curr_entry.get('state')

            if not prev_state or not curr_state:
                continue

            energy_change = energy_history[i] - energy_history[i - 1]

            transition = f"{prev_state.name if hasattr(prev_state, 'name') else str(prev_state)}->{curr_state.name if hasattr(curr_state, 'name') else str(curr_state)}"
            state_transitions[transition].append(energy_change)

        for transition, changes in state_transitions.items():
            if len(changes) >= 3:
                avg_change = np.mean(changes)
                std_change = np.std(changes)
                stability = 1.0 - (std_change / (abs(avg_change) + 1e-8))

                pattern = {
                    'transition': transition,
                    'energy_impact': avg_change,
                    'stability': max(0, min(1, stability)),
                    'frequency': len(changes)
                }
                patterns.append(pattern)

        return patterns

    def _infer_domain(self, concept: str) -> str:
        """推断概念所属领域"""
        domain_keywords = {
            'physics': ['力学', '运动', '能量', '牛顿', '引力', '动量', '摩擦力', '静电力'],
            'math': ['积分', '几何', '代数', '概率', '统计', '微积分', '拓扑学', '线性代数'],
            'cs': ['算法', '数据', '网络', '学习', '智能', '机器学习', '神经网络', '计算机视觉'],
            'principles': ['优化', '变换', '抽象', '模式', '递归', '迭代', '对称', '归纳'],
            'ai': ['学习', '智能', '网络', '深度学习', '强化学习', '生成对抗'],
            'cognitive': ['认知', '记忆', '注意', '元认知', '心理理论'],
            'neuroscience': ['神经', '突触', '海马', '前额叶'],
            'philosophy': ['认识论', '本体论', '逻辑学', '伦理学']
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in concept for keyword in keywords):
                return domain

        return 'other'

    def get_detection_summary(self) -> Dict[str, Any]:
        """获取检测摘要"""
        total_compressions = len(self.detection_history['compressions'])
        total_migrations = len(self.detection_history['migrations'])
        total_patterns = len(self.detection_history['patterns'])

        avg_compression_strength = np.mean([
            c.get('emergence_strength', 0) for c in self.detection_history['compressions']
        ]) if total_compressions > 0 else 0

        avg_migration_gain = np.mean([
            m.get('efficiency_gain', 0) for m in self.detection_history['migrations']
        ]) if total_migrations > 0 else 0

        return {
            'total_emergence_events': total_compressions + total_migrations + total_patterns,
            'compressions_detected': total_compressions,
            'migrations_detected': total_migrations,
            'patterns_detected': total_patterns,
            'avg_compression_strength': avg_compression_strength,
            'avg_migration_efficiency': avg_migration_gain
        }

    def reset_detection_history(self):
        """重置检测历史"""
        self.detection_history = {
            'compressions': [],
            'migrations': [],
            'patterns': []
        }