import numpy as np
import networkx as nx
from typing import Dict, List, Any
from collections import defaultdict

class NaturalEmergenceMetrics:
    """自然涌现指标计算 - 精简版本"""

    def __init__(self):
        self.metric_history = defaultdict(list)

    def calculate_emergence_metrics(self, network_history, traversal_history, energy_history):
        """计算综合涌现指标"""
        if len(network_history) < 2:
            return {}
            
        current_net = network_history[-1] if hasattr(network_history[-1], 'nodes') else network_history[-1]['network']
        previous_net = network_history[0] if hasattr(network_history[0], 'nodes') else network_history[0]['network']

        metrics = {
            'compression_emergence': self._measure_compression(current_net, previous_net),
            'migration_emergence': self._measure_migration(traversal_history, current_net),
            'energy_optimization': self._measure_energy_optimization(energy_history),
            'structural_evolution': self._measure_structural_evolution(current_net, previous_net)
        }
        
        # 计算综合涌现强度
        metrics['overall_emergence_strength'] = (
            metrics['compression_emergence'] * 0.3 +
            metrics['migration_emergence'] * 0.3 +
            metrics['energy_optimization'] * 0.2 +
            metrics['structural_evolution'] * 0.2
        )
        
        self.metric_history['emergence'].append(metrics)
        return metrics

    def _measure_compression(self, current_net, previous_net):
        """测量概念压缩涌现"""
        try:
            # 聚类系数增加
            current_clustering = nx.average_clustering(current_net)
            previous_clustering = nx.average_clustering(previous_net)
            clustering_increase = max(0, current_clustering - previous_clustering)
            
            # 模块度
            current_modularity = self._compute_modularity(current_net)
            previous_modularity = self._compute_modularity(previous_net)
            modularity_increase = max(0, current_modularity - previous_modularity)
            
            return min(1.0, (clustering_increase * 0.6 + modularity_increase * 0.4) * 5)
        except:
            return 0.0

    def _measure_migration(self, traversal_history, network):
        """测量原理迁移涌现"""
        if len(traversal_history) < 10:
            return 0.0
            
        cross_domain_paths = 0
        efficient_paths = 0
        
        for traversal in traversal_history[-20:]:
            path = self._extract_path(traversal)
            if not path or len(path) < 3:
                continue
                
            # 检查跨领域
            domains = set(self._infer_domain(node) for node in path)
            if len(domains) > 1:
                cross_domain_paths += 1
                
            # 检查效率
            efficiency = self._compute_path_efficiency(path, network)
            if efficiency > 0.5:
                efficient_paths += 1
                
        cross_domain_ratio = cross_domain_paths / len(traversal_history[-20:])
        efficiency_ratio = efficient_paths / len(traversal_history[-20:])
        
        return min(1.0, (cross_domain_ratio * 0.6 + efficiency_ratio * 0.4) * 2)

    def _measure_energy_optimization(self, energy_history):
        """测量能量优化"""
        if len(energy_history) < 50:
            return 0.0
            
        recent = energy_history[-50:]
        earlier = energy_history[-100:-50] if len(energy_history) >= 100 else energy_history[:50]
        
        if not earlier:
            return 0.0
            
        # 能量降低率
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        
        if earlier_avg > 0:
            reduction_rate = (earlier_avg - recent_avg) / earlier_avg
        else:
            reduction_rate = 0
            
        # 收敛稳定性
        stability = 1.0 - (np.std(recent) / (np.mean(recent) + 1e-8))
        
        return min(1.0, max(0, reduction_rate) * 0.7 + stability * 0.3)

    def _measure_structural_evolution(self, current_net, previous_net):
        """测量结构演化"""
        try:
            # 小世界特征
            current_sw = self._small_world_index(current_net)
            previous_sw = self._small_world_index(previous_net)
            sw_improvement = max(0, current_sw - previous_sw)
            
            # 连接性
            current_conn = nx.density(current_net)
            previous_conn = nx.density(previous_net)
            conn_change = abs(current_conn - previous_conn)
            
            return min(1.0, sw_improvement * 0.6 + conn_change * 0.4)
        except:
            return 0.0

    def _compute_modularity(self, network):
        """计算模块度 - 简化版本"""
        try:
            # 使用连通分量作为社区
            components = list(nx.connected_components(network))
            if len(components) < 2:
                return 0.0
            return nx.algorithms.community.modularity(network, components)
        except:
            return 0.0

    def _small_world_index(self, network):
        """计算小世界指数 - 简化版本"""
        try:
            clustering = nx.average_clustering(network)
            if nx.is_connected(network):
                path_length = nx.average_shortest_path_length(network)
            else:
                # 对于不连通图，使用最大连通分量
                largest = max(nx.connected_components(network), key=len)
                subgraph = network.subgraph(largest)
                path_length = nx.average_shortest_path_length(subgraph)
                
            return clustering / (path_length + 1e-8)
        except:
            return 0.0

    def _extract_path(self, traversal):
        """从遍历记录提取路径"""
        if isinstance(traversal, (list, tuple)) and len(traversal) > 0:
            return traversal[0] if isinstance(traversal[0], list) else [traversal[0]]
        elif isinstance(traversal, dict) and 'path' in traversal:
            return traversal['path']
        return None

    def _compute_path_efficiency(self, path, network):
        """计算路径效率"""
        total_energy = 0
        valid_edges = 0
        
        for i in range(len(path)-1):
            if network.has_edge(path[i], path[i+1]):
                total_energy += network[path[i]][path[i+1]]['weight']
                valid_edges += 1
                
        if valid_edges == 0:
            return 0.0
            
        avg_energy = total_energy / valid_edges
        return 1.0 / (avg_energy + 0.1)  # 能耗越低效率越高

    def _infer_domain(self, concept):
        """推断概念领域"""
        if '力' in concept or '能量' in concept or '运动' in concept:
            return 'physics'
        elif '积分' in concept or '几何' in concept or '代数' in concept:
            return 'math'
        elif '算法' in concept or '数据' in concept or '网络' in concept:
            return 'cs'
        elif '优化' in concept or '迭代' in concept or '抽象' in concept:
            return 'principles'
        else:
            return 'other'

    def get_metric_trends(self):
        """获取指标趋势"""
        if not self.metric_history['emergence']:
            return {}
            
        metrics = self.metric_history['emergence']
        trends = {}
        
        for key in metrics[0].keys():
            values = [m[key] for m in metrics]
            trends[f'{key}_mean'] = np.mean(values)
            trends[f'{key}_trend'] = self._compute_trend(values)
            
        return trends

    def _compute_trend(self, values):
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]  # 斜率