import numpy as np
import networkx as nx
from typing import Dict, List, Any
from collections import defaultdict

class EmergenceObserver:
    """涌现现象观察器 - 精简版本"""

    def __init__(self):
        self.observations = {
            'compressions': [],
            'migrations': [],
            'energy_patterns': []
        }
        
        # 简化阈值配置
        self.thresholds = {
            'compression_synergy': 0.7,
            'migration_efficiency': 0.3,
            'pattern_stability': 0.6
        }

    def observe_compression_emergence(self, network, energy_history, iteration):
        """观察概念压缩涌现 - 简化版本"""
        compressions = []
        
        # 分析节点集群
        for node in network.nodes():
            neighbors = list(network.neighbors(node))
            if len(neighbors) < 3:
                continue
                
            # 计算集群特性
            cohesion = self._compute_cohesion(node, neighbors, network)
            energy_sync = self._compute_energy_sync(node, neighbors, energy_history)
            
            if cohesion > 0.5 and energy_sync > self.thresholds['compression_synergy']:
                compression = {
                    'center': node,
                    'related_nodes': neighbors,
                    'cohesion': cohesion,
                    'energy_sync': energy_sync,
                    'iteration': iteration
                }
                compressions.append(compression)
                
        return compressions

    def observe_migration_emergence(self, network, traversal_history, iteration):
        """观察原理迁移涌现 - 简化版本"""
        migrations = []
        
        if len(traversal_history) < 10:
            return migrations
            
        # 分析最近的遍历
        for traversal in traversal_history[-10:]:
            path = self._extract_path(traversal)
            if not path or len(path) < 3:
                continue
                
            # 检查是否跨领域
            if self._is_cross_domain(path):
                efficiency = self._compute_path_efficiency(path, network)
                
                if efficiency > self.thresholds['migration_efficiency']:
                    mediator = self._find_mediator(path)
                    migration = {
                        'principle_node': mediator,
                        'path': path,
                        'efficiency': efficiency,
                        'iteration': iteration
                    }
                    migrations.append(migration)
                    
        return migrations

    def _compute_cohesion(self, center, neighbors, network):
        """计算集群内聚性"""
        connections = 0
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if network.has_edge(n1, n2):
                    connections += 1
                    
        return connections / possible if possible > 0 else 0

    def _compute_energy_sync(self, center, neighbors, energy_history):
        """计算能量同步性"""
        if len(energy_history) < 20:
            return 0.5
            
        # 简化计算：使用最近能量变化的趋势
        recent = energy_history[-20:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        return max(0, 1 - abs(trend))  # 趋势越平缓，同步性越高

    def _extract_path(self, traversal):
        """从遍历记录提取路径"""
        if isinstance(traversal, (list, tuple)) and len(traversal) > 0:
            return traversal[0] if isinstance(traversal[0], list) else [traversal[0]]
        elif isinstance(traversal, dict) and 'path' in traversal:
            return traversal['path']
        return None

    def _is_cross_domain(self, path):
        """检查是否跨领域路径"""
        domains = set()
        for node in path:
            domain = self._infer_domain(node)
            domains.add(domain)
        return len(domains) > 1

    def _compute_path_efficiency(self, path, network):
        """计算路径效率"""
        total_energy = 0
        for i in range(len(path)-1):
            if network.has_edge(path[i], path[i+1]):
                total_energy += network[path[i]][path[i+1]]['weight']
                
        return 1.0 / (total_energy + 0.1) if total_energy > 0 else 0  # 能耗越低效率越高

    def _find_mediator(self, path):
        """寻找中介节点"""
        if len(path) < 3:
            return path[0] if path else ""
        return path[len(path)//2]  # 返回中间节点

    def _infer_domain(self, concept):
        """推断概念领域"""
        domain_keywords = {
            'physics': ['力', '运动', '能量', '牛顿'],
            'math': ['积分', '几何', '代数', '概率'],
            'cs': ['算法', '数据', '网络', '学习'],
            'principles': ['优化', '变换', '迭代', '抽象']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in concept for keyword in keywords):
                return domain
        return 'other'

    def get_observation_summary(self):
        """获取观察总结"""
        return {
            'total_compressions': len(self.observations['compressions']),
            'total_migrations': len(self.observations['migrations']),
            'total_patterns': len(self.observations['energy_patterns'])
        }