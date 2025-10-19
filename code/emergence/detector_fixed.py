# detector_fixed.py
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import random
import math


class EmergenceDetectorFixed:
    """修复的涌现现象检测器"""

    def __init__(self, detection_thresholds: Dict[str, float] = None):
        # 降低检测阈值
        self.thresholds = detection_thresholds or {
            'compression_synergy': 0.55,  # 降低压缩协同性阈值
            'migration_efficiency': 0.25,  # 降低迁移效率阈值
            'pattern_stability': 0.65,  # 降低模式稳定性阈值
            'energy_sync_threshold': 0.5,  # 降低能量同步阈值
            'cross_domain_gain': 0.2,  # 降低跨领域增益阈值
            'cluster_cohesion': 0.5,  # 降低集群内聚性阈值
            'min_cluster_size': 2,  # 降低最小集群大小
            'max_cluster_size': 20,  # 降低最大集群大小
        }

        self.detection_history = {
            'compressions': [],
            'migrations': [],
            'patterns': []
        }

    def detect_spontaneous_compression(self, network, energy_history, traversal_history=None):
        """修复的概念压缩检测"""
        compression_candidates = []

        # 只在能量历史足够长时检测
        if len(energy_history) < 20:
            return compression_candidates

        # 分析网络中的高连接度节点作为潜在中心
        node_degrees = dict(network.degree())
        high_degree_nodes = [node for node, degree in node_degrees.items()
                             if degree >= self.thresholds['min_cluster_size']]

        for center in high_degree_nodes:
            neighbors = list(network.neighbors(center))

            if len(neighbors) < self.thresholds['min_cluster_size']:
                continue

            # 计算连接强度（权重越低，连接越强）
            connection_strengths = []
            for neighbor in neighbors:
                if network.has_edge(center, neighbor):
                    weight = network[center][neighbor]['weight']
                    # 权重越低表示连接越强
                    strength = 1.0 / (1.0 + weight)
                    connection_strengths.append((neighbor, strength))

            # 按连接强度排序
            connection_strengths.sort(key=lambda x: x[1], reverse=True)

            # 选择最强的几个邻居
            strong_neighbors = [n for n, s in connection_strengths[:self.thresholds['max_cluster_size']]]

            if len(strong_neighbors) >= self.thresholds['min_cluster_size']:
                # 计算集群内聚性
                cohesion = self._compute_cluster_cohesion(center, strong_neighbors, network)

                # 计算能量协同性
                energy_sync = self._compute_energy_synchronization_simple(center, strong_neighbors, network)

                # 计算综合涌现强度
                emergence_strength = (0.4 * cohesion + 0.6 * energy_sync)

                if (emergence_strength > self.thresholds['compression_synergy'] and
                        cohesion > self.thresholds['cluster_cohesion']):
                    compression_candidates.append({
                        'center': center,
                        'related_nodes': strong_neighbors,
                        'energy_synergy': energy_sync,
                        'cohesion': cohesion,
                        'emergence_strength': emergence_strength,
                        'cluster_size': len(strong_neighbors)
                    })

        return compression_candidates

    def _compute_energy_synchronization_simple(self, center, neighbors, network):
        """简化的能量同步性计算"""
        if len(neighbors) < 2:
            return 0.0

        center_energies = []
        neighbor_avg_energies = []

        for neighbor in neighbors:
            if network.has_edge(center, neighbor):
                # 中心到邻居的能量
                center_energy = network[center][neighbor]['weight']
                center_energies.append(center_energy)

                # 邻居与其他邻居的平均能量
                neighbor_energies = []
                for other in neighbors:
                    if other != neighbor and network.has_edge(neighbor, other):
                        neighbor_energies.append(network[neighbor][other]['weight'])

                if neighbor_energies:
                    neighbor_avg_energies.append(np.mean(neighbor_energies))
                else:
                    neighbor_avg_energies.append(center_energy)

        if len(center_energies) < 2:
            return 0.5

        # 计算相关性
        try:
            correlation = np.corrcoef(center_energies, neighbor_avg_energies)[0, 1]
            if np.isnan(correlation):
                return 0.5
            return abs(correlation)
        except:
            return 0.5

    def _compute_cluster_cohesion(self, center: str, neighbors: List[str], network: nx.Graph) -> float:
        """计算集群内聚性"""
        if len(neighbors) < 2:
            return 0.0

        total_possible = len(neighbors) * (len(neighbors) - 1) / 2
        actual_connections = 0

        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                if network.has_edge(n1, n2):
                    actual_connections += 1

        return actual_connections / total_possible if total_possible > 0 else 0.0

    def detect_emergent_migration(self, network: nx.Graph, traversal_history: List[Any] = None,
                                  current_iteration: int = 0, semantic_network=None) -> List[Dict[str, Any]]:
        """修复的原理迁移检测"""
        migrations = []

        if traversal_history is None or len(traversal_history) < 10:
            return migrations

        # 分析最近的遍历路径
        recent_paths = []
        for traversal in traversal_history[-20:]:
            path = self._extract_traversal_path(traversal)
            if path and len(path) >= 3:  # 至少3个节点的路径才考虑
                recent_paths.append(path)

        # 寻找包含原理节点的跨领域路径
        for path in recent_paths:
            principle_nodes = [node for node in path if self._is_principle_node(node)]

            if len(principle_nodes) > 0 and self._is_cross_domain_path(path):
                # 计算路径效率
                path_efficiency = self._calculate_path_efficiency(path, network)

                if path_efficiency > self.thresholds['migration_efficiency']:
                    # 选择第一个原理节点作为迁移桥梁
                    principle_node = principle_nodes[0]

                    migration = {
                        'type': 'first_principles_migration',
                        'principle_node': principle_node,
                        'from_node': path[0],
                        'to_node': path[-1],
                        'efficiency_gain': path_efficiency,
                        'path': path,
                        'emergence_iteration': current_iteration,
                        'domain_span': self._calculate_domain_span_simple(path)
                    }
                    migrations.append(migration)

        return migrations

    def _extract_traversal_path(self, traversal):
        """从遍历记录中提取路径"""
        if isinstance(traversal, dict) and 'path' in traversal:
            return traversal['path']
        elif isinstance(traversal, (list, tuple)) and len(traversal) > 0:
            if isinstance(traversal[0], list):
                return traversal[0]
            elif len(traversal) >= 3:
                return traversal[0] if isinstance(traversal[0], list) else [traversal[0]]
        return None

    def _is_principle_node(self, node):
        """判断是否是原理性节点"""
        principle_keywords = ['优化', '变换', '抽象', '模式', '递归', '迭代', '对称', '归纳']
        return any(keyword in node for keyword in principle_keywords)

    def _is_cross_domain_path(self, path):
        """判断是否跨领域路径"""
        if len(path) < 2:
            return False

        domains = [self._infer_domain(node) for node in path]
        return len(set(domains)) > 1

    def _calculate_path_efficiency(self, path, network):
        """计算路径效率"""
        if len(path) < 2:
            return 0.0

        total_energy = 0.0
        for i in range(len(path) - 1):
            if network.has_edge(path[i], path[i + 1]):
                total_energy += network[path[i]][path[i + 1]]['weight']

        # 效率 = 1 / (总能耗 * 路径长度)，能耗越低、路径越短，效率越高
        if total_energy > 0:
            efficiency = 1.0 / (total_energy * len(path))
            return min(efficiency, 1.0)
        return 0.0

    def _calculate_domain_span_simple(self, path):
        """计算领域跨度"""
        domains = set()
        for node in path:
            domains.add(self._infer_domain(node))
        return len(domains)

    def _infer_domain(self, concept: str) -> str:
        """推断概念领域"""
        domain_keywords = {
            'physics': ['力', '运动', '能量', '动量', '牛顿', '引力', '摩擦', '静电'],
            'math': ['积分', '几何', '代数', '概率', '统计', '微积分', '拓扑', '线性'],
            'cs': ['算法', '数据', '网络', '学习', '智能', '机器', '计算机', '视觉'],
            'principles': ['优化', '变换', '抽象', '模式', '递归', '迭代', '对称', '归纳'],
            'cognitive': ['认知', '记忆', '注意', '元认知', '心理'],
            'neuroscience': ['神经', '突触', '海马', '前额叶'],
            'philosophy': ['认识论', '本体论', '逻辑学', '伦理学']
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in concept for keyword in keywords):
                return domain

        return 'other'