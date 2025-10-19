"""
涌现现象量化指标模块
用于测量和评估从认知系统中自然涌现的各种现象
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import math


class NaturalEmergenceMetrics:
    """自然涌现现象的量化指标计算器"""

    def __init__(self):
        self.metric_history = defaultdict(list)

    def measure_compression_emergence(self, network_history: List[nx.Graph],
                                      window_size: int = 100) -> Dict[str, float]:
        """
        测量概念压缩的涌现强度

        参数:
            network_history: 网络演化历史
            window_size: 分析窗口大小

        返回:
            压缩涌现指标字典
        """
        if len(network_history) < window_size:
            return {}

        current_network = network_history[-1]
        previous_network = network_history[-window_size]

        metrics = {
            'node_clustering_increase': self._calculate_clustering_increase(
                current_network, previous_network
            ),
            'inter_cluster_energy_decrease': self._calculate_inter_cluster_energy_decrease(
                current_network, previous_network
            ),
            'intra_cluster_cohesion': self._calculate_intra_cluster_cohesion(current_network),
            'modularity_increase': self._calculate_modularity_increase(
                current_network, previous_network
            ),
            'compression_emergence_strength': 0.0
        }

        # 计算综合压缩涌现强度
        metrics['compression_emergence_strength'] = (
                metrics['node_clustering_increase'] * 0.3 +
                metrics['inter_cluster_energy_decrease'] * 0.4 +
                metrics['intra_cluster_cohesion'] * 0.2 +
                metrics['modularity_increase'] * 0.1
        )

        self.metric_history['compression_metrics'].append(metrics)
        return metrics

    def measure_migration_emergence(self, traversal_history: List[Dict],
                                    network: nx.Graph) -> Dict[str, float]:
        """
        测量原理迁移的涌现强度

        参数:
            traversal_history: 遍历历史记录
            network: 当前网络状态

        返回:
            迁移涌现指标字典
        """
        if len(traversal_history) < 10:
            return {}

        metrics = {
            'cross_domain_efficiency': self._calculate_cross_domain_efficiency(
                traversal_history, network
            ),
            'principle_node_centrality': self._calculate_principle_node_centrality(network),
            'path_innovation_rate': self._calculate_path_innovation_rate(traversal_history),
            'migration_discovery_frequency': self._calculate_migration_discovery_frequency(
                traversal_history
            ),
            'migration_emergence_strength': 0.0
        }

        # 计算综合迁移涌现强度
        metrics['migration_emergence_strength'] = (
                metrics['cross_domain_efficiency'] * 0.4 +
                metrics['principle_node_centrality'] * 0.2 +
                metrics['path_innovation_rate'] * 0.3 +
                metrics['migration_discovery_frequency'] * 0.1
        )

        self.metric_history['migration_metrics'].append(metrics)
        return metrics

    def measure_energy_minimization(self, energy_history: List[float],
                                    iteration_window: int = 500) -> Dict[str, float]:
        """
        测量能量最小化过程的效率

        参数:
            energy_history: 能量历史记录
            iteration_window: 分析窗口大小

        返回:
            能量最小化指标字典
        """
        if len(energy_history) < iteration_window:
            return {}

        recent_energies = energy_history[-iteration_window:]
        earlier_energies = energy_history[-2 * iteration_window:-iteration_window]

        metrics = {
            'energy_reduction_rate': self._calculate_energy_reduction_rate(
                earlier_energies, recent_energies
            ),
            'convergence_stability': self._calculate_convergence_stability(recent_energies),
            'optimization_efficiency': self._calculate_optimization_efficiency(energy_history),
            'energy_volatility': self._calculate_energy_volatility(recent_energies),
            'minimization_quality': 0.0
        }

        # 计算综合最小化质量
        metrics['minimization_quality'] = (
                metrics['energy_reduction_rate'] * 0.4 +
                (1 - metrics['energy_volatility']) * 0.3 +
                metrics['convergence_stability'] * 0.2 +
                metrics['optimization_efficiency'] * 0.1
        )

        self.metric_history['energy_metrics'].append(metrics)
        return metrics

    def measure_structural_emergence(self, network_history: List[nx.Graph],
                                     concept_domains: Dict[str, List[str]] = None) -> Dict[str, float]:
        """
        测量结构涌现现象

        参数:
            network_history: 网络演化历史
            concept_domains: 概念领域映射

        返回:
            结构涌现指标字典
        """
        if len(network_history) < 2:
            return {}

        current_network = network_history[-1]
        initial_network = network_history[0]

        metrics = {
            'small_world_emergence': self._detect_small_world_emergence(current_network),
            'scale_free_emergence': self._detect_scale_free_emergence(current_network),
            'hierarchy_emergence': self._detect_hierarchy_emergence(current_network),
            'modularity_emergence': self._detect_modularity_emergence(current_network),
            'cross_domain_integration': self._measure_cross_domain_integration(
                current_network, concept_domains
            ),
            'structural_emergence_strength': 0.0
        }

        # 计算综合结构涌现强度
        metrics['structural_emergence_strength'] = (
                metrics['small_world_emergence'] * 0.25 +
                metrics['scale_free_emergence'] * 0.2 +
                metrics['hierarchy_emergence'] * 0.2 +
                metrics['modularity_emergence'] * 0.2 +
                metrics['cross_domain_integration'] * 0.15
        )

        self.metric_history['structural_metrics'].append(metrics)
        return metrics

    def _calculate_clustering_increase(self, current: nx.Graph, previous: nx.Graph) -> float:
        """计算节点聚类系数的增加"""
        try:
            current_clustering = nx.average_clustering(current)
            previous_clustering = nx.average_clustering(previous)
            increase = current_clustering - previous_clustering
            return max(0.0, increase)  # 只关心增加
        except:
            return 0.0

    def _calculate_inter_cluster_energy_decrease(self, current: nx.Graph, previous: nx.Graph) -> float:
        """计算簇间能耗的降低"""
        try:
            # 简化的簇间能耗估算
            current_avg_energy = np.mean([current[u][v]['weight'] for u, v in current.edges()])
            previous_avg_energy = np.mean([previous[u][v]['weight'] for u, v in previous.edges()])
            decrease = previous_avg_energy - current_avg_energy
            return max(0.0, decrease / previous_avg_energy) if previous_avg_energy > 0 else 0.0
        except:
            return 0.0

    def _calculate_intra_cluster_cohesion(self, network: nx.Graph) -> float:
        """计算簇内凝聚性"""
        try:
            # 使用社区检测评估簇内凝聚性
            communities = self._detect_communities(network)
            if not communities:
                return 0.0

            cohesion_scores = []
            for community in communities:
                if len(community) < 2:
                    continue
                subgraph = network.subgraph(community)
                internal_edges = subgraph.number_of_edges()
                possible_edges = len(community) * (len(community) - 1) / 2
                if possible_edges > 0:
                    cohesion_scores.append(internal_edges / possible_edges)

            return np.mean(cohesion_scores) if cohesion_scores else 0.0
        except:
            return 0.0

    def _calculate_modularity_increase(self, current: nx.Graph, previous: nx.Graph) -> float:
        """计算模块度的增加"""
        try:
            current_modularity = nx.algorithms.community.modularity(
                current, self._detect_communities(current)
            )
            previous_modularity = nx.algorithms.community.modularity(
                previous, self._detect_communities(previous)
            )
            increase = current_modularity - previous_modularity
            return max(0.0, increase)
        except:
            return 0.0

    def _calculate_cross_domain_efficiency(self, traversal_history: List[Dict],
                                           network: nx.Graph) -> float:
        """计算跨领域遍历效率"""
        cross_domain_paths = []

        for traversal in traversal_history[-100:]:  # 分析最近100次遍历
            path = traversal.get('path', [])
            if len(path) >= 3 and self._is_cross_domain_path(path):
                efficiency = self._calculate_path_efficiency(path, network)
                cross_domain_paths.append(efficiency)

        return np.mean(cross_domain_paths) if cross_domain_paths else 0.0

    def _calculate_principle_node_centrality(self, network: nx.Graph) -> float:
        """计算原理节点的中心性"""
        try:
            # 识别原理节点（基于节点名称或度中心性）
            principle_nodes = [
                node for node in network.nodes()
                if any(keyword in node for keyword in ['优化', '变换', '迭代', '抽象', '模式'])
            ]

            if not principle_nodes:
                return 0.0

            betweenness = nx.betweenness_centrality(network)
            principle_centrality = np.mean([betweenness[node] for node in principle_nodes])
            return principle_centrality
        except:
            return 0.0

    def _calculate_path_innovation_rate(self, traversal_history: List[Dict]) -> float:
        """计算路径创新率"""
        if len(traversal_history) < 10:
            return 0.0

        innovative_paths = 0
        recent_paths = [traversal.get('path', []) for traversal in traversal_history[-10:]]

        for i in range(1, len(recent_paths)):
            if self._is_innovative_path(recent_paths[i], recent_paths[i - 1]):
                innovative_paths += 1

        return innovative_paths / len(recent_paths)

    def _calculate_migration_discovery_frequency(self, traversal_history: List[Dict]) -> float:
        """计算迁移发现频率"""
        migration_discoveries = 0
        total_iterations = len(traversal_history)

        for traversal in traversal_history:
            if traversal.get('type') == 'migration_discovery':
                migration_discoveries += 1

        return migration_discoveries / total_iterations if total_iterations > 0 else 0.0

    def _calculate_energy_reduction_rate(self, earlier: List[float], recent: List[float]) -> float:
        """计算能量降低速率"""
        earlier_avg = np.mean(earlier)
        recent_avg = np.mean(recent)

        if earlier_avg > 0:
            reduction_rate = (earlier_avg - recent_avg) / earlier_avg
            return max(0.0, reduction_rate)
        return 0.0

    def _calculate_convergence_stability(self, energies: List[float]) -> float:
        """计算收敛稳定性"""
        if len(energies) < 2:
            return 0.0

        # 计算能量序列的稳定性（变异系数的倒数）
        cv = np.std(energies) / np.mean(energies) if np.mean(energies) > 0 else 1.0
        stability = 1.0 / (1.0 + cv)  # 转换为稳定性指标
        return stability

    def _calculate_optimization_efficiency(self, energy_history: List[float]) -> float:
        """计算优化效率"""
        if len(energy_history) < 10:
            return 0.0

        total_reduction = energy_history[0] - energy_history[-1]
        max_possible_reduction = energy_history[0] - 0.1  # 假设最小能耗为0.1

        if max_possible_reduction > 0:
            efficiency = total_reduction / max_possible_reduction
            return max(0.0, min(1.0, efficiency))
        return 0.0

    def _calculate_energy_volatility(self, energies: List[float]) -> float:
        """计算能量波动性"""
        if len(energies) < 2:
            return 0.0

        returns = np.diff(energies) / energies[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.0
        return volatility

    def _detect_small_world_emergence(self, network: nx.Graph) -> float:
        """检测小世界现象涌现"""
        try:
            # 小世界网络特征：高聚类系数和短平均路径长度
            clustering = nx.average_clustering(network)
            try:
                path_length = nx.average_shortest_path_length(network)
            except:
                # 对于不连通图，使用最大连通分量
                largest_cc = max(nx.connected_components(network), key=len)
                subgraph = network.subgraph(largest_cc)
                path_length = nx.average_shortest_path_length(subgraph)

            # 简化的小世界指数（实际需要与随机网络比较）
            small_world_index = clustering / (path_length + 1e-6)  # 避免除零
            return min(1.0, small_world_index * 10)  # 缩放至合理范围
        except:
            return 0.0

    def _detect_scale_free_emergence(self, network: nx.Graph) -> float:
        """检测无标度现象涌现"""
        try:
            degrees = [d for n, d in network.degree()]
            if len(degrees) < 3:
                return 0.0

            # 计算度分布的幂律特征（简化版本）
            from collections import Counter
            degree_count = Counter(degrees)
            x = np.log(list(degree_count.keys()))
            y = np.log(list(degree_count.values()))

            # 线性拟合斜率（幂律指数）
            if len(x) > 1:
                slope, _ = np.polyfit(x, y, 1)
                # 无标度网络通常有2-3的幂律指数
                scale_free_likelihood = 1.0 - abs(slope - 2.5) / 2.5
                return max(0.0, scale_free_likelihood)
            return 0.0
        except:
            return 0.0

    def _detect_hierarchy_emergence(self, network: nx.Graph) -> float:
        """检测层次结构涌现"""
        try:
            # 层次结构特征：度分布与聚类系数的负相关性
            degrees = []
            clusterings = []

            for node in network.nodes():
                degrees.append(network.degree(node))
                clusterings.append(nx.clustering(network, node))

            if len(degrees) > 2:
                correlation = np.corrcoef(degrees, clusterings)[0, 1]
                # 层次网络通常显示负相关
                hierarchy_strength = max(0.0, -correlation)  # 负相关越强，层次性越强
                return hierarchy_strength
            return 0.0
        except:
            return 0.0

    def _detect_modularity_emergence(self, network: nx.Graph) -> float:
        """检测模块化结构涌现"""
        try:
            communities = self._detect_communities(network)
            if len(communities) < 2:
                return 0.0

            modularity = nx.algorithms.community.modularity(network, communities)
            return max(0.0, modularity)
        except:
            return 0.0

    def _measure_cross_domain_integration(self, network: nx.Graph,
                                          concept_domains: Dict[str, List[str]]) -> float:
        """测量跨领域整合程度"""
        if not concept_domains:
            return 0.0

        cross_domain_edges = 0
        total_edges = network.number_of_edges()

        if total_edges == 0:
            return 0.0

        for u, v in network.edges():
            domain_u = self._get_concept_domain(u, concept_domains)
            domain_v = self._get_concept_domain(v, concept_domains)

            if domain_u != domain_v and domain_u != "other" and domain_v != "other":
                cross_domain_edges += 1

        return cross_domain_edges / total_edges

    def _detect_communities(self, network: nx.Graph) -> List[List[Any]]:
        """检测网络中的社区结构"""
        try:
            # 使用Louvain算法检测社区
            import community as community_louvain
            partition = community_louvain.best_partition(network)

            communities_dict = defaultdict(list)
            for node, community_id in partition.items():
                communities_dict[community_id].append(node)

            return list(communities_dict.values())
        except:
            # 回退到连通分量
            return [list(comp) for comp in nx.connected_components(network)]

    def _is_cross_domain_path(self, path: List[str]) -> bool:
        """判断路径是否跨领域"""
        if len(path) < 2:
            return False

        domains = [self._infer_domain(node) for node in path]
        return len(set(domains)) > 1

    def _calculate_path_efficiency(self, path: List[str], network: nx.Graph) -> float:
        """计算路径效率"""
        if len(path) < 2:
            return 0.0

        total_energy = 0
        for i in range(len(path) - 1):
            if network.has_edge(path[i], path[i + 1]):
                total_energy += network[path[i]][path[i + 1]]['weight']

        # 效率 = 1 / 总能耗（能耗越低，效率越高）
        return 1.0 / (total_energy + 1e-6) if total_energy > 0 else 0.0

    def _is_innovative_path(self, current_path: List[str], previous_path: List[str]) -> bool:
        """判断路径是否具有创新性"""
        if not current_path or not previous_path:
            return False

        # 创新性标准：新路径包含新节点或新的连接模式
        current_nodes = set(current_path)
        previous_nodes = set(previous_path)

        new_nodes = current_nodes - previous_nodes
        structural_innovation = len(current_path) != len(previous_path)

        return len(new_nodes) > 0 or structural_innovation

    def _get_concept_domain(self, concept: str, concept_domains: Dict[str, List[str]]) -> str:
        """获取概念所属领域"""
        for domain, concepts in concept_domains.items():
            if concept in concepts:
                return domain
        return "other"

    def _infer_domain(self, concept: str) -> str:
        """推断概念领域（简化版本）"""
        domain_keywords = {
            'physics': ['力', '运动', '能量', '动量', '牛顿'],
            'math': ['积分', '几何', '代数', '概率', '统计'],
            'cs': ['算法', '数据', '学习', '网络', '计算'],
            'principles': ['优化', '变换', '迭代', '抽象', '模式']
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in concept for keyword in keywords):
                return domain

        return "other"

    def get_metric_trends(self, metric_type: str, window_size: int = 10) -> Dict[str, Any]:
        """获取指标趋势分析"""
        if metric_type not in self.metric_history:
            return {}

        metrics_list = self.metric_history[metric_type]
        if len(metrics_list) < window_size:
            return {}

        recent_metrics = metrics_list[-window_size:]
        trends = {}

        for key in recent_metrics[0].keys():
            values = [metric[key] for metric in recent_metrics]
            trends[f'{key}_mean'] = np.mean(values)
            trends[f'{key}_std'] = np.std(values)
            trends[f'{key}_trend'] = self._calculate_trend(values)

        return trends

    def _calculate_trend(self, values: List[float]) -> float:
        """计算数值趋势（斜率）"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    def generate_emergence_report(self, all_metrics: Dict[str, Dict]) -> str:
        """生成涌现现象报告"""
        report = []
        report.append("=== 自然涌现现象量化报告 ===")

        for phenomenon, metrics in all_metrics.items():
            report.append(f"\n【{phenomenon}】")
            for metric_name, value in metrics.items():
                if 'strength' in metric_name or 'quality' in metric_name:
                    report.append(f"  {metric_name}: {value:.3f}")
                else:
                    report.append(f"  {metric_name}: {value:.3f}")

        # 计算综合涌现强度
        overall_strength = self._calculate_overall_emergence_strength(all_metrics)
        report.append(f"\n综合涌现强度: {overall_strength:.3f}")

        return "\n".join(report)

    def _calculate_overall_emergence_strength(self, all_metrics: Dict[str, Dict]) -> float:
        """计算综合涌现强度"""
        strength_metrics = []
        weights = {
            'compression_emergence_strength': 0.3,
            'migration_emergence_strength': 0.3,
            'minimization_quality': 0.2,
            'structural_emergence_strength': 0.2
        }

        for metric_name, weight in weights.items():
            for phenomenon_metrics in all_metrics.values():
                if metric_name in phenomenon_metrics:
                    strength_metrics.append(phenomenon_metrics[metric_name] * weight)

        return np.sum(strength_metrics) if strength_metrics else 0.0


# 便捷函数
def create_metrics_tracker() -> NaturalEmergenceMetrics:
    """创建指标跟踪器实例"""
    return NaturalEmergenceMetrics()