"""
涌现现象检测器（修复版）
-------------------------
定义 EmergenceDetectorFixed 类，用于从认知网络中检测自发的概念压缩和原理迁移现象。
包含内聚性、能量同步性、涌现强度等指标的计算。
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional


class EmergenceDetectorFixed:
    """修复的涌现现象检测器，用于检测概念压缩和原理迁移。

    检测阈值可在初始化时指定，默认值基于经验设置。包含去重机制避免重复记录相同涌现事件。
    """

    def __init__(self, detection_thresholds: Optional[Dict[str, float]] = None):
        """
        :param detection_thresholds: 阈值字典，支持以下键：
            - compression_synergy: 压缩协同性阈值 (默认0.76)
            - migration_efficiency: 迁移效率阈值 (默认0.35)
            - pattern_stability: 模式稳定性阈值 (默认0.7)
            - energy_sync_threshold: 能量同步性阈值 (默认0.65)
            - cross_domain_gain: 跨领域增益 (默认0.2)
            - cluster_cohesion: 集群内聚性阈值 (默认0.7)
            - min_cluster_size: 最小集群大小 (默认2)
            - max_cluster_size: 最大集群大小 (默认6)
            - dynamic_cluster_sizing: 是否动态确定集群大小 (默认True)
            - compression_persistence: 压缩持久性要求 (默认3)
            - migration_confidence: 迁移置信度阈值 (默认0.75)
            - min_connection_strength: 最小连接强度 (默认0.5)
        """
        self.thresholds = detection_thresholds or {
            'compression_synergy': 0.76,
            'migration_efficiency': 0.35,
            'pattern_stability': 0.7,
            'energy_sync_threshold': 0.65,
            'cross_domain_gain': 0.2,
            'cluster_cohesion': 0.7,
            'min_cluster_size': 2,
            'max_cluster_size': 6,
            'dynamic_cluster_sizing': True,
            'compression_persistence': 3,
            'migration_confidence': 0.75,
            'min_connection_strength': 0.5
        }
        self.compression_history = defaultdict(list)
        self.migration_history = defaultdict(list)
        self.detection_history = {
            'compressions': [],
            'migrations': [],
            'patterns': []
        }

    def detect_spontaneous_compression(self, network: nx.Graph,
                                       energy_history: List[float],
                                       traversal_history: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """检测自发的概念压缩现象。

        压缩候选需满足：
            - 中心节点度在合适范围内
            - 邻居连接强度高于阈值
            - 集群内聚性、能量同步性和综合涌现强度均超过阈值

        :param network: 当前认知网络
        :param energy_history: 网络平均能耗历史
        :param traversal_history: 遍历历史（未使用，保留接口）
        :return: 压缩候选列表，每个元素为包含中心节点、相关节点、各项指标的字典
        """
        compression_candidates = []

        if len(energy_history) < 50:
            return compression_candidates

        node_degrees = dict(network.degree())
        candidate_nodes = [
            node for node, degree in node_degrees.items()
            if 3 <= degree <= 15
        ]

        for center in candidate_nodes:
            neighbors = list(network.neighbors(center))

            if len(neighbors) < self.thresholds['min_cluster_size']:
                continue

            connection_strengths = []
            for neighbor in neighbors:
                if network.has_edge(center, neighbor):
                    weight = network[center][neighbor]['weight']
                    strength = 1.0 / (1.0 + weight)
                    if strength > self.thresholds['min_connection_strength']:
                        connection_strengths.append((neighbor, strength))

            if len(connection_strengths) < self.thresholds['min_cluster_size']:
                continue

            connection_strengths.sort(key=lambda x: x[1], reverse=True)

            if self.thresholds['dynamic_cluster_sizing']:
                cluster_size = self._determine_dynamic_cluster_size(connection_strengths)
            else:
                cluster_size = self.thresholds['max_cluster_size']

            selected_neighbors = [n for n, s in connection_strengths[:cluster_size]]

            cohesion = self._compute_cluster_cohesion(center, selected_neighbors, network)

            energy_sync = self._compute_energy_synchronization_improved(
                center, selected_neighbors, network, energy_history
            )

            emergence_strength = self._compute_comprehensive_emergence_strength(
                center, selected_neighbors, network, cohesion, energy_sync
            )

            if (emergence_strength > self.thresholds['compression_synergy'] and
                    cohesion > self.thresholds['cluster_cohesion'] and
                    energy_sync > self.thresholds['energy_sync_threshold']):

                if not self._is_duplicate_compression(center, selected_neighbors):
                    potential = self._compute_compression_potential(center, selected_neighbors, network)
                    compression_candidates.append({
                        'center': center,
                        'related_nodes': selected_neighbors,
                        'energy_synergy': energy_sync,
                        'cohesion': cohesion,
                        'emergence_strength': emergence_strength,
                        'cluster_size': len(selected_neighbors),
                        'avg_connection_strength': np.mean([s for _, s in connection_strengths[:cluster_size]]),
                        'compression_potential': potential  # 新增字段
                    })

        return compression_candidates[:10]

    def _determine_dynamic_cluster_size(self, connection_strengths: List[Tuple[str, float]]) -> int:
        """根据连接强度分布动态确定集群大小。

        找到强度明显下降的点（低于均值减0.5倍标准差），若无则返回适中大小。

        :param connection_strengths: 按强度降序排列的 (节点, 强度) 列表
        :return: 集群大小
        """
        strengths = [s for _, s in connection_strengths]
        avg_strength = np.mean(strengths)
        std_strength = np.std(strengths)

        for i in range(1, len(strengths)):
            if strengths[i] < avg_strength - 0.5 * std_strength:
                return max(self.thresholds['min_cluster_size'],
                           min(i, self.thresholds['max_cluster_size']))

        return min(len(strengths),
                   max(self.thresholds['min_cluster_size'] + 2,
                       len(strengths) // 2))

    def _compute_energy_synchronization_improved(self, center: str, neighbors: List[str],
                                                 network: nx.Graph, energy_history: List[float]) -> float:
        """计算中心节点与邻居之间的能量变化同步性。

        :return: 同步性得分 [0,1]
        """
        if len(neighbors) < 2:
            return 0.0

        center_trend = self._compute_energy_trend(center, network, energy_history)
        neighbor_trends = []
        for neighbor in neighbors:
            trend = self._compute_energy_trend(neighbor, network, energy_history)
            neighbor_trends.append(trend)

        if not neighbor_trends:
            return 0.0

        avg_neighbor_trend = np.mean(neighbor_trends)
        trend_similarity = 1.0 - abs(center_trend - avg_neighbor_trend)
        return max(0.0, trend_similarity)

    def _compute_energy_trend(self, node: str, network: nx.Graph,
                              energy_history: List[float], window_size: int = 20) -> float:
        """估计节点相关的能量趋势（基于其邻边的当前平均能耗）。"""
        if len(energy_history) < window_size:
            return 0.0

        recent_energies = []
        for neighbor in network.neighbors(node):
            if network.has_edge(node, neighbor):
                recent_energies.append(network[node][neighbor]['weight'])

        if not recent_energies:
            return 0.0

        return np.mean(recent_energies)

    def _compute_comprehensive_emergence_strength(self, center: str, neighbors: List[str],
                                                  network: nx.Graph, cohesion: float,
                                                  energy_sync: float) -> float:
        """综合涌现强度，结合内聚性、能量同步性、连接密度和语义连贯性。"""
        connection_density = self._compute_connection_density(neighbors, network)
        semantic_coherence = self._estimate_semantic_coherence(center, neighbors)

        emergence_strength = (
                0.3 * cohesion +
                0.3 * energy_sync +
                0.2 * connection_density +
                0.2 * semantic_coherence
        )
        return min(1.0, emergence_strength)

    def _compute_connection_density(self, nodes: List[str], network: nx.Graph) -> float:
        """计算节点集合内部的连接密度。"""
        if len(nodes) < 2:
            return 0.0

        possible = len(nodes) * (len(nodes) - 1) / 2
        actual = 0
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                if network.has_edge(n1, n2):
                    actual += 1
        return actual / possible if possible > 0 else 0.0

    def _estimate_semantic_coherence(self, center: str, neighbors: List[str]) -> float:
        """估计节点间的语义连贯性（基于节点名称的字符重叠启发式）。"""
        center_words = set(center)
        scores = []
        for neighbor in neighbors:
            neighbor_words = set(neighbor)
            overlap = len(center_words.intersection(neighbor_words))
            total = len(center_words.union(neighbor_words))
            if total > 0:
                scores.append(overlap / total)
        return np.mean(scores) if scores else 0.0

    def _compute_cluster_cohesion(self, center: str, neighbors: List[str], network: nx.Graph) -> float:
        """计算邻居节点之间的内聚性（边密度）。"""
        if len(neighbors) < 2:
            return 0.0

        total_possible = len(neighbors) * (len(neighbors) - 1) / 2
        actual_connections = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                if network.has_edge(n1, n2):
                    actual_connections += 1
        return actual_connections / total_possible if total_possible > 0 else 0.0

    def detect_emergent_migration(self, network: nx.Graph,
                                  traversal_history: Optional[List[Any]] = None,
                                  current_iteration: int = 0,
                                  semantic_network: Optional[Any] = None) -> List[Dict[str, Any]]:
        """检测涌现的原理迁移现象。

        分析最近的遍历路径，找出包含原理节点且跨领域的路径，计算效率增益。

        :param network: 当前认知网络
        :param traversal_history: 遍历历史记录
        :param current_iteration: 当前迭代次数
        :param semantic_network: 语义网络（可选，用于更精细的判断）
        :return: 迁移候选列表
        """
        migrations = []

        if traversal_history is None or len(traversal_history) < 10:
            return migrations

        recent_paths = []
        for traversal in traversal_history[-20:]:
            path = self._extract_traversal_path(traversal)
            if path and len(path) >= 3:
                recent_paths.append(path)

        for path in recent_paths:
            principle_nodes = [node for node in path if self._is_principle_node(node)]

            if len(principle_nodes) > 0 and self._is_cross_domain_path(path):
                path_efficiency = self._calculate_path_efficiency(path, network)

                if path_efficiency > self.thresholds['migration_efficiency']:
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

    def _extract_traversal_path(self, traversal: Any) -> Optional[List[str]]:
        """从遍历记录中提取路径列表。"""
        if isinstance(traversal, dict) and 'path' in traversal:
            return traversal['path']
        elif isinstance(traversal, (list, tuple)) and len(traversal) > 0:
            if isinstance(traversal[0], list):
                return traversal[0]
            elif len(traversal) >= 3:
                return traversal[0] if isinstance(traversal[0], list) else [traversal[0]]
        return None

    def _is_principle_node(self, node: str) -> bool:
        """判断节点名是否包含原理性关键词。"""
        principle_keywords = ['优化', '变换', '抽象', '模式', '递归', '迭代', '对称', '归纳']
        return any(keyword in node for keyword in principle_keywords)

    def _is_cross_domain_path(self, path: List[str]) -> bool:
        """判断路径是否跨越多个领域。"""
        if len(path) < 2:
            return False
        domains = [self._infer_domain(node) for node in path]
        return len(set(domains)) > 1

    def _calculate_path_efficiency(self, path: List[str], network: nx.Graph) -> float:
        """计算路径效率：1 / (总能耗 * 路径长度)。"""
        total_energy = 0.0
        for i in range(len(path) - 1):
            if network.has_edge(path[i], path[i + 1]):
                total_energy += network[path[i]][path[i + 1]]['weight']
        if total_energy > 0:
            efficiency = 1.0 / (total_energy * len(path))
            return min(efficiency, 1.0)
        return 0.0

    def _calculate_domain_span_simple(self, path: List[str]) -> int:
        """计算路径所涉及的领域数量。"""
        domains = set()
        for node in path:
            domains.add(self._infer_domain(node))
        return len(domains)

    def _infer_domain(self, concept: str) -> str:
        """根据关键词推断概念领域。"""
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

    def calculate_compression_confidence(self, compression: Dict[str, Any]) -> float:
        """计算压缩事件的置信度。"""
        synergy = compression.get('energy_synergy', 0)
        cohesion = compression.get('cohesion', 0)
        cluster_size = compression.get('cluster_size', 0)

        confidence = (
                0.4 * synergy +
                0.3 * cohesion +
                0.2 * min(1.0, cluster_size / 6) +
                0.1 * self._calculate_temporal_stability(compression)
        )
        return min(1.0, confidence)

    def calculate_migration_confidence(self, migration: Dict[str, Any]) -> float:
        """计算迁移事件的置信度。"""
        efficiency = migration.get('efficiency_gain', 0)
        domain_span = migration.get('domain_span', 0)
        path_length = len(migration.get('path', []))

        confidence = (
                0.5 * efficiency +
                0.2 * min(1.0, domain_span / 3) +
                0.2 * (1.0 / max(1, path_length)) +
                0.1 * self._calculate_innovation_score(migration)
        )
        return min(1.0, confidence)

    def _calculate_temporal_stability(self, compression: Dict[str, Any]) -> float:
        """计算压缩的时间稳定性（基于历史记录）。"""
        center = compression['center']
        current_strength = compression['emergence_strength']

        if center in self.compression_history:
            historical_strengths = [c['emergence_strength'] for c in self.compression_history[center]]
            avg_historical = np.mean(historical_strengths)
            stability = 1.0 - abs(current_strength - avg_historical)
            return max(0, stability)
        return 0.5

    def _calculate_innovation_score(self, migration: Dict[str, Any]) -> float:
        """计算迁移的创新性得分。"""
        path = migration.get('path', [])
        principle_node = migration.get('principle_node', '')

        innovation_score = 0.0
        if len(path) >= 3:
            if principle_node not in self.migration_history:
                innovation_score += 0.3

            domains = [self._infer_domain(node) for node in path]
            unique_domains = len(set(domains))
            innovation_score += min(0.7, unique_domains * 0.2)

        return innovation_score

    def _is_duplicate_compression(self, center: str, neighbors: List[str]) -> bool:
        """检查是否已记录过相同的压缩事件。"""
        key = (center, tuple(sorted(neighbors)))
        if key in self.compression_history:
            return True
        self.compression_history[key] = True
        return False

    def _compute_compression_potential(self, center: str, neighbors: List[str], network: nx.Graph) -> float:
        """计算集群压缩势 Φ = 内部平均能耗 / 外部平均能耗"""
        cluster_nodes = {center} | set(neighbors)
        internal_energy = 0.0
        internal_count = 0
        external_energy = 0.0
        external_count = 0
        processed_edges = set()

        for v in cluster_nodes:
            for u in network.neighbors(v):
                edge = tuple(sorted((v, u)))
                if edge in processed_edges:
                    continue
                processed_edges.add(edge)
                weight = network[v][u]['weight']
                if u in cluster_nodes:
                    internal_energy += weight
                    internal_count += 1
                else:
                    external_energy += weight
                    external_count += 1

        avg_internal = internal_energy / internal_count if internal_count > 0 else 0.0
        avg_external = external_energy / external_count if external_count > 0 else None

        if avg_external is None or avg_external == 0:
            # 无外部连接时压缩势无定义，设为 None（JSON 序列化为 null）
            return None
        return avg_internal / avg_external


if __name__ == "__main__":
    # 简单测试
    detector = EmergenceDetectorFixed()
    print("EmergenceDetectorFixed 初始化成功")