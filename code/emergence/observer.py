"""
涌现观察器 - 被动观察认知宇宙中的自然涌现现象
从机制设计转向现象观察的核心组件
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class EmergenceObserver:
    """涌现现象观察器 - 被动记录，主动发现"""

    def __init__(self):
        self.emergence_log = []
        self.detection_history = []

        # 涌现检测阈值 - 基于实证调整
        self.detection_thresholds = {
            'compression_synergy': 0.65,  # 压缩协同性阈值
            'migration_efficiency': 0.25,  # 迁移效率阈值
            'pattern_stability': 0.75,  # 模式稳定性阈值
            'energy_sync_threshold': 0.6,  # 能量同步阈值
            'innovation_significance': 0.15  # 创新显著性阈值
        }

        # 观察统计
        self.observation_stats = {
            'compression_events': 0,
            'migration_events': 0,
            'pattern_emergence': 0,
            'energy_minimization_steps': 0
        }

    def observe_compression_emergence(self, network, energy_history, iteration) -> List[Dict]:
        """
        观察概念压缩的自然涌现
        检测标准：节点集群的能耗协同降低和连接强度增加
        """
        emergent_compressions = []

        try:
            # 1. 检测能量同步的节点集群
            energy_sync_clusters = self._find_energy_sync_clusters(network, energy_history)

            for cluster in energy_sync_clusters:
                # 2. 评估是否正在形成概念压缩
                if self._is_compression_emerging(cluster, network):
                    compression_strength = self._compute_compression_strength(cluster, network)

                    if compression_strength > self.detection_thresholds['compression_synergy']:
                        emergent_compression = {
                            'type': 'conceptual_compression',
                            'emergence_iteration': iteration,
                            'nodes': cluster['nodes'],
                            'center_candidate': cluster['center'],
                            'energy_synergy': cluster['synergy'],
                            'compression_strength': compression_strength,
                            'cluster_cohesion': self._compute_cluster_cohesion(cluster['nodes'], network),
                            'evidence': {
                                'energy_reduction': cluster['energy_reduction'],
                                'connection_strengthening': cluster['connection_growth']
                            }
                        }

                        emergent_compressions.append(emergent_compression)
                        self.observation_stats['compression_events'] += 1

                        print(f"🔍 观察到概念压缩涌现: {cluster['center']} 周围 {len(cluster['nodes'])} 个节点")
                        print(f"   协同性: {cluster['synergy']:.3f}, 强度: {compression_strength:.3f}")

            return emergent_compressions

        except Exception as e:
            print(f"观察压缩涌现时出错: {e}")
            return []

    def observe_migration_emergence(self, network, traversal_history, semantic_network, iteration) -> List[Dict]:
        """
        观察第一性原理迁移的自然涌现
        检测标准：跨领域高效路径的发现和强化
        """
        emergent_migrations = []

        try:
            if len(traversal_history) < 10:  # 需要足够的历史数据
                return []

            # 1. 分析遍历模式的变化
            pattern_shifts = self._analyze_traversal_pattern_shifts(traversal_history, network)

            for shift in pattern_shifts:
                # 2. 评估是否正在形成原理迁移
                if self._is_migration_emerging(shift, semantic_network):
                    migration_efficiency = self._compute_migration_efficiency(shift, network)

                    if migration_efficiency > self.detection_thresholds['migration_efficiency']:
                        emergent_migration = {
                            'type': 'first_principles_migration',
                            'emergence_iteration': iteration,
                            'discovery_path': shift['new_pattern']['path'],
                            'principle_node': shift['mediator'],
                            'efficiency_gain': shift['efficiency_gain'],
                            'migration_efficiency': migration_efficiency,
                            'domain_span': self._calculate_domain_span(shift['new_pattern']['path'], semantic_network),
                            'innovation_significance': self._compute_innovation_significance(shift),
                            'evidence': {
                                'path_length_reduction': shift['path_length_reduction'],
                                'energy_saving': shift['energy_saving']
                            }
                        }

                        emergent_migrations.append(emergent_migration)
                        self.observation_stats['migration_events'] += 1

                        print(f"🌉 观察到原理迁移涌现: 通过 {shift['mediator']}")
                        print(f"   效率提升: {shift['efficiency_gain']:.3f}, 创新性: {migration_efficiency:.3f}")

            return emergent_migrations

        except Exception as e:
            print(f"观察迁移涌现时出错: {e}")
            return []

    def observe_energy_minimization_patterns(self, energy_history, network, iteration) -> List[Dict]:
        """
        观察能量最小化模式
        检测标准：全局能量下降的显著模式和节奏
        """
        energy_patterns = []

        try:
            if len(energy_history) < 50:  # 需要足够的历史数据
                return []

            # 1. 检测能量下降的显著阶段
            minimization_phases = self._detect_minimization_phases(energy_history)

            for phase in minimization_phases:
                if phase['significance'] > self.detection_thresholds['pattern_stability']:
                    energy_pattern = {
                        'type': 'energy_minimization_phase',
                        'detection_iteration': iteration,
                        'phase_duration': phase['duration'],
                        'energy_reduction': phase['energy_reduction'],
                        'significance': phase['significance'],
                        'phase_type': phase['type'],
                        'rate_of_change': phase['rate_of_change'],
                        'stability_measure': phase['stability']
                    }

                    energy_patterns.append(energy_pattern)
                    self.observation_stats['energy_minimization_steps'] += 1

            return energy_patterns

        except Exception as e:
            print(f"观察能量模式时出错: {e}")
            return []

    def observe_traversal_pattern_differentiation(self, traversal_history, network, iteration) -> List[Dict]:
        """
        观察遍历模式的分化
        检测标准：硬遍历和软遍历模式的自然分化
        """
        pattern_differentiations = []

        try:
            if len(traversal_history) < 20:
                return []

            # 1. 分析遍历模式的分化
            differentiations = self._analyze_traversal_differentiation(traversal_history, network)

            for diff in differentiations:
                if diff['differentiation_strength'] > self.detection_thresholds['pattern_stability']:
                    pattern_diff = {
                        'type': 'traversal_pattern_differentiation',
                        'detection_iteration': iteration,
                        'hard_traversal_pattern': diff['hard_pattern'],
                        'soft_traversal_pattern': diff['soft_pattern'],
                        'differentiation_strength': diff['differentiation_strength'],
                        'pattern_stability': diff['stability'],
                        'evidence': {
                            'hard_pattern_frequency': diff['hard_frequency'],
                            'soft_pattern_frequency': diff['soft_frequency'],
                            'pattern_distinctness': diff['distinctness']
                        }
                    }

                    pattern_differentiations.append(pattern_diff)
                    self.observation_stats['pattern_emergence'] += 1

            return pattern_differentiations

        except Exception as e:
            print(f"观察遍历模式时出错: {e}")
            return []

    def _find_energy_sync_clusters(self, network, energy_history, window_size=50) -> List[Dict]:
        """发现能量同步降低的节点集群"""
        clusters = []

        if len(energy_history) < window_size:
            return clusters

        # 获取最近的网络变化
        recent_energy_changes = self._compute_recent_energy_changes(network, energy_history, window_size)

        # 寻找能量协同降低的节点组
        for node in network.nodes():
            neighbors = list(network.neighbors(node))
            if len(neighbors) < 2:
                continue

            # 计算节点与邻居的能量协同性
            sync_score = self._compute_energy_sync(node, neighbors, recent_energy_changes)

            if sync_score > self.detection_thresholds['energy_sync_threshold']:
                cluster = {
                    'center': node,
                    'nodes': neighbors,
                    'synergy': sync_score,
                    'energy_reduction': self._compute_cluster_energy_reduction(node, neighbors, recent_energy_changes),
                    'connection_growth': self._compute_connection_strengthening(node, neighbors, network)
                }
                clusters.append(cluster)

        return sorted(clusters, key=lambda x: x['synergy'], reverse=True)[:5]  # 返回前5个

    def _is_compression_emerging(self, cluster: Dict, network) -> bool:
        """判断是否正在形成概念压缩"""
        if len(cluster['nodes']) < 2:
            return False

        # 检查连接密度
        connection_density = self._compute_connection_density(cluster['nodes'], network)
        if connection_density < 0.3:
            return False

        # 检查能量协同性
        if cluster['synergy'] < self.detection_thresholds['energy_sync_threshold']:
            return False

        # 检查连接强化趋势
        if cluster['connection_growth'] < 0.1:
            return False

        return True

    def _is_migration_emerging(self, pattern_shift: Dict, semantic_network) -> bool:
        """判断是否正在形成原理迁移"""
        if not pattern_shift.get('mediator'):
            return False

        # 检查效率提升
        if pattern_shift['efficiency_gain'] < self.detection_thresholds['innovation_significance']:
            return False

        # 检查跨领域性
        domain_span = self._calculate_domain_span(pattern_shift['new_pattern']['path'], semantic_network)
        if domain_span < 2:  # 至少跨越2个领域
            return False

        # 检查创新性
        innovation_score = self._compute_innovation_significance(pattern_shift)
        if innovation_score < self.detection_thresholds['innovation_significance']:
            return False

        return True

    def _compute_energy_sync(self, center, neighbors, energy_changes) -> float:
        """计算节点集群的能量协同性"""
        if center not in energy_changes:
            return 0.0

        center_change = energy_changes[center]
        neighbor_changes = [energy_changes.get(n, 0) for n in neighbors if n in energy_changes]

        if not neighbor_changes:
            return 0.0

        # 计算变化方向的一致性
        same_direction = sum(1 for nc in neighbor_changes
                             if (center_change > 0 and nc > 0) or (center_change < 0 and nc < 0))

        sync_ratio = same_direction / len(neighbor_changes)

        # 考虑变化幅度的一致性
        avg_neighbor_change = np.mean([abs(nc) for nc in neighbor_changes])
        magnitude_sync = 1 - abs(abs(center_change) - avg_neighbor_change) / max(abs(center_change),
                                                                                 avg_neighbor_change, 0.001)

        return 0.6 * sync_ratio + 0.4 * magnitude_sync

    def _compute_innovation_significance(self, pattern_shift: Dict) -> float:
        """计算模式创新的显著性"""
        base_score = pattern_shift.get('efficiency_gain', 0)

        # 考虑路径长度的创新性
        path_innovation = 1.0 / max(1, pattern_shift.get('path_length_reduction', 1))

        # 考虑能耗节省
        energy_innovation = pattern_shift.get('energy_saving', 0)

        return base_score * 0.5 + path_innovation * 0.3 + energy_innovation * 0.2

    def _calculate_domain_span(self, path: List, semantic_network) -> int:
        """计算路径跨越的领域数量"""
        domains = set()
        for node in path:
            domain = semantic_network.get_domain(node) if hasattr(semantic_network, 'get_domain') else "unknown"
            domains.add(domain)
        return len(domains)

    def _compute_recent_energy_changes(self, network, energy_history, window_size: int) -> Dict:
        """计算最近时间窗口内的能量变化"""
        # 简化的实现 - 在实际中需要更精细的能量变化跟踪
        changes = {}

        # 这里使用边的平均能量变化作为代理
        if len(energy_history) >= window_size:
            recent_avg = np.mean(energy_history[-window_size:])
            previous_avg = np.mean(energy_history[-2 * window_size:-window_size])
            global_change = recent_avg - previous_avg

            # 为每个节点分配基于度的变化估计
            for node in network.nodes():
                degree = network.degree(node)
                changes[node] = global_change * (1 + 0.1 * degree)  # 高度数节点变化更大

        return changes

    def _compute_compression_strength(self, cluster: Dict, network) -> float:
        """计算压缩强度"""
        synergy = cluster['synergy']
        cohesion = self._compute_cluster_cohesion(cluster['nodes'], network)
        energy_reduction = min(1.0, abs(cluster['energy_reduction']) * 10)  # 归一化

        return 0.4 * synergy + 0.4 * cohesion + 0.2 * energy_reduction

    def _compute_cluster_cohesion(self, nodes: List, network) -> float:
        """计算节点集群的内聚性"""
        if len(nodes) < 2:
            return 0.0

        actual_edges = 0
        possible_edges = len(nodes) * (len(nodes) - 1) / 2

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                if network.has_edge(node1, node2):
                    actual_edges += 1

        return actual_edges / possible_edges if possible_edges > 0 else 0.0

    def _compute_connection_density(self, nodes: List, network) -> float:
        """计算连接密度"""
        return self._compute_cluster_cohesion(nodes, network)

    def _compute_cluster_energy_reduction(self, center, neighbors, energy_changes) -> float:
        """计算集群能量降低"""
        total_reduction = energy_changes.get(center, 0)
        for neighbor in neighbors:
            total_reduction += energy_changes.get(neighbor, 0)
        return total_reduction

    def _compute_connection_strengthening(self, center, neighbors, network) -> float:
        """计算连接强化程度"""
        total_strength = 0
        count = 0

        for neighbor in neighbors:
            if network.has_edge(center, neighbor):
                weight = network[center][neighbor].get('weight', 1.0)
                # 假设权重降低表示强化（能耗降低）
                strength = 2.0 - weight  # 权重越低，强化程度越高
                total_strength += strength
                count += 1

        return total_strength / count if count > 0 else 0.0

    def _analyze_traversal_pattern_shifts(self, traversal_history, network) -> List[Dict]:
        """分析遍历模式的转变"""
        shifts = []

        if len(traversal_history) < 10:
            return shifts

        # 分析最近的遍历模式变化
        recent_patterns = traversal_history[-10:]

        for i in range(len(recent_patterns) - 1):
            old_pattern = recent_patterns[i]
            new_pattern = recent_patterns[i + 1]

            # 检查是否发现了新的高效路径
            if self._is_efficient_path_discovery(old_pattern, new_pattern, network):
                shift = {
                    'old_pattern': old_pattern,
                    'new_pattern': new_pattern,
                    'mediator': self._find_mediator_node(old_pattern, new_pattern),
                    'efficiency_gain': self._compute_efficiency_gain(old_pattern, new_pattern, network),
                    'path_length_reduction': self._compute_path_length_reduction(old_pattern, new_pattern),
                    'energy_saving': self._compute_energy_saving(old_pattern, new_pattern, network)
                }
                shifts.append(shift)

        return shifts

    def _analyze_traversal_differentiation(self, traversal_history, network) -> List[Dict]:
        """分析遍历模式的分化"""
        differentiations = []

        # 分离硬遍历和软遍历
        hard_traversals = [t for t in traversal_history if t[1] == "hard"]
        soft_traversals = [t for t in traversal_history if t[1] == "soft"]

        if len(hard_traversals) > 5 and len(soft_traversals) > 5:
            # 分析模式差异
            hard_pattern = self._characterize_traversal_pattern(hard_traversals, network)
            soft_pattern = self._characterize_traversal_pattern(soft_traversals, network)

            differentiation = {
                'hard_pattern': hard_pattern,
                'soft_pattern': soft_pattern,
                'differentiation_strength': self._compute_pattern_differentiation(hard_pattern, soft_pattern),
                'stability': min(hard_pattern['stability'], soft_pattern['stability']),
                'hard_frequency': len(hard_traversals) / len(traversal_history),
                'soft_frequency': len(soft_traversals) / len(traversal_history),
                'distinctness': self._compute_pattern_distinctness(hard_pattern, soft_pattern)
            }

            differentiations.append(differentiation)

        return differentiations

    def _is_efficient_path_discovery(self, old_pattern, new_pattern, network) -> bool:
        """检查是否发现了新的高效路径"""
        old_path, old_type, _ = old_pattern
        new_path, new_type, _ = new_pattern

        # 简单的启发式：新路径包含新节点或更短
        if len(set(new_path) - set(old_path)) > 0 or len(new_path) < len(old_path):
            old_energy = self._compute_path_energy(old_path, network)
            new_energy = self._compute_path_energy(new_path, network)
            return new_energy < old_energy * 0.9  # 至少10%的改进

        return False

    def _find_mediator_node(self, old_pattern, new_pattern):
        """寻找中介节点（原理节点）"""
        old_path, _, _ = old_pattern
        new_path, _, _ = new_pattern

        # 寻找在新路径中但不在旧路径中的节点
        new_nodes = set(new_path) - set(old_path)
        if new_nodes:
            return list(new_nodes)[0]  # 返回第一个新节点作为候选中介

        return None

    def _compute_efficiency_gain(self, old_pattern, new_pattern, network) -> float:
        """计算效率提升"""
        old_path, _, _ = old_pattern
        new_path, _, _ = new_pattern

        old_energy = self._compute_path_energy(old_path, network)
        new_energy = self._compute_path_energy(new_path, network)

        if old_energy > 0:
            return (old_energy - new_energy) / old_energy
        return 0.0

    def _compute_path_energy(self, path, network) -> float:
        """计算路径总能量"""
        total_energy = 0
        for i in range(len(path) - 1):
            if network.has_edge(path[i], path[i + 1]):
                total_energy += network[path[i]][path[i + 1]].get('weight', 1.0)
        return total_energy

    def _compute_path_length_reduction(self, old_pattern, new_pattern) -> int:
        """计算路径长度减少"""
        old_path, _, _ = old_pattern
        new_path, _, _ = new_pattern
        return len(old_path) - len(new_path)

    def _compute_energy_saving(self, old_pattern, new_pattern, network) -> float:
        """计算能耗节省"""
        return self._compute_efficiency_gain(old_pattern, new_pattern, network)

    def _compute_migration_efficiency(self, pattern_shift: Dict, network) -> float:
        """计算迁移效率"""
        efficiency_gain = pattern_shift.get('efficiency_gain', 0)
        path_reduction = min(1.0, pattern_shift.get('path_length_reduction', 0) / 5.0)  # 归一化
        energy_saving = pattern_shift.get('energy_saving', 0)

        return 0.5 * efficiency_gain + 0.3 * path_reduction + 0.2 * energy_saving

    def _characterize_traversal_pattern(self, traversals, network) -> Dict:
        """特征化遍历模式"""
        if not traversals:
            return {}

        path_lengths = [len(t[0]) for t in traversals]
        energy_levels = [self._compute_path_energy(t[0], network) for t in traversals]
        node_diversity = len(set(node for t in traversals for node in t[0])) / len(
            traversals[0][0]) if traversals else 0

        return {
            'avg_path_length': np.mean(path_lengths),
            'avg_energy': np.mean(energy_levels),
            'node_diversity': node_diversity,
            'stability': 1.0 - (np.std(path_lengths) / max(1, np.mean(path_lengths))),
            'frequency': len(traversals)
        }

    def _compute_pattern_differentiation(self, pattern1: Dict, pattern2: Dict) -> float:
        """计算模式分化程度"""
        length_diff = abs(pattern1.get('avg_path_length', 0) - pattern2.get('avg_path_length', 0)) / max(1,
                                                                                                         pattern1.get(
                                                                                                             'avg_path_length',
                                                                                                             1))
        energy_diff = abs(pattern1.get('avg_energy', 0) - pattern2.get('avg_energy', 0)) / max(1, pattern1.get(
            'avg_energy', 1))
        diversity_diff = abs(pattern1.get('node_diversity', 0) - pattern2.get('node_diversity', 0))

        return 0.4 * length_diff + 0.4 * energy_diff + 0.2 * diversity_diff

    def _compute_pattern_distinctness(self, pattern1: Dict, pattern2: Dict) -> float:
        """计算模式独特性"""
        return self._compute_pattern_differentiation(pattern1, pattern2)

    def _detect_minimization_phases(self, energy_history) -> List[Dict]:
        """检测能量最小化阶段"""
        phases = []

        if len(energy_history) < 10:
            return phases

        # 简单的阶段检测：寻找持续下降的阶段
        window_size = 10
        for i in range(len(energy_history) - window_size):
            segment = energy_history[i:i + window_size]
            if self._is_monotonic_decreasing(segment):
                phase = {
                    'start': i,
                    'duration': window_size,
                    'energy_reduction': segment[0] - segment[-1],
                    'significance': (segment[0] - segment[-1]) / segment[0] if segment[0] > 0 else 0,
                    'type': 'monotonic_decrease',
                    'rate_of_change': (segment[0] - segment[-1]) / window_size,
                    'stability': 1.0 - (np.std(segment) / max(1, np.mean(segment)))
                }
                phases.append(phase)

        return sorted(phases, key=lambda x: x['significance'], reverse=True)[:3]  # 返回前3个显著阶段

    def _is_monotonic_decreasing(self, sequence) -> bool:
        """检查序列是否单调递减"""
        return all(sequence[i] >= sequence[i + 1] for i in range(len(sequence) - 1))

    def get_observation_summary(self) -> Dict:
        """获取观察总结"""
        return {
            'total_observations': len(self.emergence_log),
            'compression_events': self.observation_stats['compression_events'],
            'migration_events': self.observation_stats['migration_events'],
            'pattern_emergence': self.observation_stats['pattern_emergence'],
            'energy_minimization_steps': self.observation_stats['energy_minimization_steps'],
            'detection_thresholds': self.detection_thresholds
        }

    def visualize_emergence_patterns(self, emergence_data: List[Dict]):
        """可视化涌现模式"""
        if not emergence_data:
            print("没有足够的涌现数据用于可视化")
            return

        # 提取时间和类型信息
        iterations = [e.get('emergence_iteration', e.get('detection_iteration', 0)) for e in emergence_data]
        types = [e['type'] for e in emergence_data]

        # 创建类型到颜色的映射
        type_colors = {
            'conceptual_compression': 'red',
            'first_principles_migration': 'blue',
            'energy_minimization_phase': 'green',
            'traversal_pattern_differentiation': 'orange'
        }

        colors = [type_colors.get(t, 'gray') for t in types]

        plt.figure(figsize=(12, 6))
        plt.scatter(iterations, range(len(emergence_data)), c=colors, alpha=0.7, s=100)
        plt.xlabel('迭代次数')
        plt.ylabel('涌现事件索引')
        plt.title('认知涌现事件时间线')

        # 添加图例
        for t, color in type_colors.items():
            plt.plot([], [], 'o', color=color, label=t)
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"\n=== 涌现观察总结 ===")
        summary = self.get_observation_summary()
        for key, value in summary.items():
            if key != 'detection_thresholds':
                print(f"{key}: {value}")