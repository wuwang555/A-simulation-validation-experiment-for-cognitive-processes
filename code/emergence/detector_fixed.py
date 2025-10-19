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
        # 改进的检测阈值
        self.thresholds = detection_thresholds or {
            'compression_synergy': 0.76,      # 提高压缩协同性阈值
            'migration_efficiency': 0.35,     # 提高迁移效率阈值
            'pattern_stability': 0.7,
            'energy_sync_threshold': 0.65,
            'cross_domain_gain': 0.2,
            'cluster_cohesion': 0.7,
            'min_cluster_size': 2,            # 允许更小的集群
            'max_cluster_size': 6,            # 允许更大的集群
            'dynamic_cluster_sizing': True,   # 新增：动态集群大小
            'compression_persistence': 3,
            'migration_confidence': 0.75,
            'min_connection_strength': 0.5    # 新增：最小连接强度
        }
        # 添加检测历史用于去重
        self.compression_history = defaultdict(list)
        self.migration_history = defaultdict(list)
        self.detection_history = {
            'compressions': [],
            'migrations': [],
            'patterns': []
        }

    def detect_spontaneous_compression(self, network, energy_history, traversal_history=None):
        """修复的概念压缩检测 - 改进版本"""
        compression_candidates = []

        # 只在能量历史足够长时检测
        if len(energy_history) < 50:
            return compression_candidates

        # 分析网络中的高连接度节点作为潜在中心
        node_degrees = dict(network.degree())
        # 选择度适中的节点作为中心候选（避免过度连接的节点）
        candidate_nodes = [
            node for node, degree in node_degrees.items()
            if 3 <= degree <= 15  # 限制度范围，避免太大或太小的集群
        ]

        for center in candidate_nodes:
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
                    if strength > self.thresholds['min_connection_strength']:
                        connection_strengths.append((neighbor, strength))

            if len(connection_strengths) < self.thresholds['min_cluster_size']:
                continue

            # 按连接强度排序
            connection_strengths.sort(key=lambda x: x[1], reverse=True)

            # 动态确定集群大小
            if self.thresholds['dynamic_cluster_sizing']:
                # 基于连接强度分布确定集群大小
                cluster_size = self._determine_dynamic_cluster_size(connection_strengths)
            else:
                cluster_size = self.thresholds['max_cluster_size']

            # 选择最强的邻居
            selected_neighbors = [n for n, s in connection_strengths[:cluster_size]]

            # 计算集群内聚性
            cohesion = self._compute_cluster_cohesion(center, selected_neighbors, network)

            # 计算能量协同性
            energy_sync = self._compute_energy_synchronization_improved(
                center, selected_neighbors, network, energy_history
            )

            # 计算综合涌现强度
            emergence_strength = self._compute_comprehensive_emergence_strength(
                center, selected_neighbors, network, cohesion, energy_sync
            )

            # 应用更严格的阈值
            if (emergence_strength > self.thresholds['compression_synergy'] and
                    cohesion > self.thresholds['cluster_cohesion'] and
                    energy_sync > self.thresholds['energy_sync_threshold']):

                # 检查是否重复
                if not self._is_duplicate_compression(center, selected_neighbors):
                    compression_candidates.append({
                        'center': center,
                        'related_nodes': selected_neighbors,
                        'energy_synergy': energy_sync,
                        'cohesion': cohesion,
                        'emergence_strength': emergence_strength,
                        'cluster_size': len(selected_neighbors),
                        'avg_connection_strength': np.mean([s for _, s in connection_strengths[:cluster_size]])
                    })

        return compression_candidates[:10]  # 限制返回数量，避免过多检测

    def _determine_dynamic_cluster_size(self, connection_strengths):
        """动态确定集群大小"""
        strengths = [s for _, s in connection_strengths]

        # 基于强度分布确定集群大小
        avg_strength = np.mean(strengths)
        std_strength = np.std(strengths)

        # 找到强度明显下降的点
        for i in range(1, len(strengths)):
            if strengths[i] < avg_strength - 0.5 * std_strength:
                return max(self.thresholds['min_cluster_size'],
                           min(i, self.thresholds['max_cluster_size']))

        # 如果没有明显下降，使用适中的大小
        return min(len(strengths),
                   max(self.thresholds['min_cluster_size'] + 2,
                       len(strengths) // 2))

    def _compute_energy_synchronization_improved(self, center, neighbors, network, energy_history):
        """改进的能量同步性计算"""
        if len(neighbors) < 2:
            return 0.0

        # 计算中心节点与邻居的能量变化趋势
        center_energy_trend = self._compute_energy_trend(center, network, energy_history)
        neighbor_trends = []

        for neighbor in neighbors:
            trend = self._compute_energy_trend(neighbor, network, energy_history)
            neighbor_trends.append(trend)

        if not neighbor_trends:
            return 0.0

        # 计算趋势一致性
        avg_neighbor_trend = np.mean(neighbor_trends)
        trend_similarity = 1.0 - abs(center_energy_trend - avg_neighbor_trend)

        return max(0.0, trend_similarity)

    def _compute_energy_trend(self, node, network, energy_history, window_size=20):
        """计算节点能量趋势"""
        if len(energy_history) < window_size:
            return 0.0

        # 简化的趋势计算：基于最近的能量变化
        recent_energies = []
        for neighbor in network.neighbors(node):
            if network.has_edge(node, neighbor):
                recent_energies.append(network[node][neighbor]['weight'])

        if not recent_energies:
            return 0.0

        current_avg = np.mean(recent_energies)
        # 这里简化处理，实际应该基于历史数据
        return current_avg

    def _compute_comprehensive_emergence_strength(self, center, neighbors, network, cohesion, energy_sync):
        """计算综合涌现强度"""
        # 考虑多个因素
        connection_density = self._compute_connection_density(neighbors, network)
        semantic_coherence = self._estimate_semantic_coherence(center, neighbors)

        # 加权综合
        emergence_strength = (
                0.3 * cohesion +
                0.3 * energy_sync +
                0.2 * connection_density +
                0.2 * semantic_coherence
        )

        return min(1.0, emergence_strength)

    def _compute_connection_density(self, nodes, network):
        """计算节点间的连接密度"""
        if len(nodes) < 2:
            return 0.0

        possible_connections = len(nodes) * (len(nodes) - 1) / 2
        actual_connections = 0

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                if network.has_edge(node1, node2):
                    actual_connections += 1

        return actual_connections / possible_connections if possible_connections > 0 else 0.0

    def _estimate_semantic_coherence(self, center, neighbors):
        """估计语义连贯性（简化版本）"""
        # 在实际应用中，这里应该使用语义网络计算
        # 这里使用基于节点名称的简单启发式
        center_words = set(center)
        coherence_scores = []

        for neighbor in neighbors:
            neighbor_words = set(neighbor)
            overlap = len(center_words.intersection(neighbor_words))
            total = len(center_words.union(neighbor_words))
            if total > 0:
                coherence_scores.append(overlap / total)

        return np.mean(coherence_scores) if coherence_scores else 0.0

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

    def calculate_compression_confidence(self, compression):
        """计算压缩置信度"""
        synergy = compression.get('energy_synergy', 0)
        cohesion = compression.get('cohesion', 0)
        cluster_size = compression.get('cluster_size', 0)

        # 置信度公式
        confidence = (
                0.4 * synergy +
                0.3 * cohesion +
                0.2 * min(1.0, cluster_size / 6) +  # 集群大小因子
                0.1 * self._calculate_temporal_stability(compression)  # 时间稳定性
        )

        return min(1.0, confidence)

    def calculate_migration_confidence(self, migration):
        """计算迁移置信度"""
        efficiency = migration.get('efficiency_gain', 0)
        domain_span = migration.get('domain_span', 0)
        path_length = len(migration.get('path', []))

        # 置信度公式
        confidence = (
                0.5 * efficiency +
                0.2 * min(1.0, domain_span / 3) +  # 领域跨度因子
                0.2 * (1.0 / max(1, path_length)) +  # 路径长度因子
                0.1 * self._calculate_innovation_score(migration)  # 创新性得分
        )

        return min(1.0, confidence)

    def _calculate_temporal_stability(self, compression):
        """计算时间稳定性"""
        center = compression['center']
        current_strength = compression['emergence_strength']

        # 检查历史记录
        if center in self.compression_history:
            historical_strengths = [c['emergence_strength'] for c in self.compression_history[center]]
            avg_historical = np.mean(historical_strengths)
            stability = 1.0 - abs(current_strength - avg_historical)
            return max(0, stability)

        return 0.5  # 默认值

    def _calculate_innovation_score(self, migration):
        """计算创新性得分"""
        path = migration.get('path', [])
        principle_node = migration.get('principle_node', '')

        # 检查是否包含新的连接模式
        innovation_score = 0.0
        if len(path) >= 3:
            # 检查中间节点是否是第一次作为迁移桥梁
            if principle_node not in self.migration_history:
                innovation_score += 0.3

            # 检查是否连接了新的领域组合
            domains = [self._infer_domain(node) for node in path]
            unique_domains = len(set(domains))
            innovation_score += min(0.7, unique_domains * 0.2)

        return innovation_score

    def _is_duplicate_compression(self, center, neighbors):
        """检查重复压缩"""
        key = (center, tuple(sorted(neighbors)))
        if key in self.compression_history:
            return True
        self.compression_history[key] = True
        return False