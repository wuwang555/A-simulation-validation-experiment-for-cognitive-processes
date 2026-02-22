"""
涌现现象观察器（精简版）
------------------------
定义 EmergenceObserver 类，实时观察并记录概念压缩和原理迁移现象。
"""

import numpy as np
from typing import List, Dict, Any, Optional


class EmergenceObserver:
    """涌现现象观察器，定期从网络中检测涌现事件并记录。"""

    def __init__(self):
        self.observations = {
            'compressions': [],
            'migrations': [],
            'energy_patterns': []
        }

        self.thresholds = {
            'compression_synergy': 0.7,
            'migration_efficiency': 0.3,
            'pattern_stability': 0.6
        }

    def observe_compression_emergence(self, network: Any, energy_history: List[float],
                                      iteration: int) -> List[Dict[str, Any]]:
        """观察概念压缩涌现。

        :param network: 认知网络（应具有 nodes() 和 neighbors() 方法）
        :param energy_history: 能耗历史
        :param iteration: 当前迭代次数
        :return: 本次观察到的压缩事件列表
        """
        compressions = []

        for node in network.nodes():
            neighbors = list(network.neighbors(node))
            if len(neighbors) < 3:
                continue

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
                self.observations['compressions'].append(compression)

        return compressions

    def observe_migration_emergence(self, network: Any, traversal_history: List[Any],
                                    iteration: int) -> List[Dict[str, Any]]:
        """观察原理迁移涌现。

        :param network: 认知网络
        :param traversal_history: 遍历历史
        :param iteration: 当前迭代次数
        :return: 本次观察到的迁移事件列表
        """
        migrations = []

        if len(traversal_history) < 10:
            return migrations

        for traversal in traversal_history[-10:]:
            path = self._extract_path(traversal)
            if not path or len(path) < 3:
                continue

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
                    self.observations['migrations'].append(migration)

        return migrations

    def _compute_cohesion(self, center: str, neighbors: List[str], network: Any) -> float:
        """计算邻居节点间的内聚性（边密度）。"""
        connections = 0
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if network.has_edge(n1, n2):
                    connections += 1
        return connections / possible if possible > 0 else 0

    def _compute_energy_sync(self, center: str, neighbors: List[str], energy_history: List[float]) -> float:
        """计算能量同步性：基于最近能量变化的趋势平缓程度。"""
        if len(energy_history) < 20:
            return 0.5
        recent = energy_history[-20:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        return max(0, 1 - abs(trend))

    def _extract_path(self, traversal: Any) -> Optional[List[str]]:
        """从遍历记录中提取路径。"""
        if isinstance(traversal, (list, tuple)) and len(traversal) > 0:
            return traversal[0] if isinstance(traversal[0], list) else [traversal[0]]
        elif isinstance(traversal, dict) and 'path' in traversal:
            return traversal['path']
        return None

    def _is_cross_domain(self, path: List[str]) -> bool:
        """判断路径是否跨领域。"""
        domains = set()
        for node in path:
            domain = self._infer_domain(node)
            domains.add(domain)
        return len(domains) > 1

    def _compute_path_efficiency(self, path: List[str], network: Any) -> float:
        """计算路径效率：1 / (总能耗 + 0.1)。"""
        total_energy = 0
        for i in range(len(path)-1):
            if network.has_edge(path[i], path[i+1]):
                total_energy += network[path[i]][path[i+1]]['weight']
        return 1.0 / (total_energy + 0.1) if total_energy > 0 else 0

    def _find_mediator(self, path: List[str]) -> str:
        """返回路径中间节点作为中介。"""
        if len(path) < 3:
            return path[0] if path else ""
        return path[len(path)//2]

    def _infer_domain(self, concept: str) -> str:
        """简单领域推断。"""
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

    def get_observation_summary(self) -> Dict[str, int]:
        """返回观察统计摘要。"""
        return {
            'total_compressions': len(self.observations['compressions']),
            'total_migrations': len(self.observations['migrations']),
            'total_patterns': len(self.observations['energy_patterns'])
        }


if __name__ == "__main__":
    obs = EmergenceObserver()
    print("EmergenceObserver 初始化成功")