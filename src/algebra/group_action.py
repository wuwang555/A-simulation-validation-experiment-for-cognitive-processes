# algebra/group_action.py
"""
群作用模块

根据论文第4.3节，认知对称群在认知状态空间上有一个群作用。
该模块实现了群作用、轨道和稳定子的计算，并验证轨道-稳定子定理。
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple


class GroupActionOnCognitiveSpace:
    """群在认知状态空间上的作用。

    该类封装了对称群对认知网络的作用，包括计算轨道、稳定子以及验证
    轨道-稳定子定理。

    Attributes:
        group (CognitiveSymmetryGroup): 关联的认知对称群。
    """

    def __init__(self, symmetry_group: 'CognitiveSymmetryGroup'):
        """
        Args:
            symmetry_group: 认知对称群实例。
        """
        self.group = symmetry_group
        self.orbits_cache = {}
        self.stabilizers_cache = {}

    def apply_group_element(self, network: nx.Graph,
                            permutation: Dict) -> nx.Graph:
        """应用群元素（节点置换）到认知网络。

        Args:
            network (nx.Graph): 原始网络。
            permutation (Dict): 置换映射，如 {旧节点: 新节点}。

        Returns:
            nx.Graph: 置换后的新网络。
        """
        new_network = nx.Graph()
        # 添加节点（使用置换后的名称）
        for node in network.nodes():
            new_node = permutation.get(node, node)
            new_network.add_node(new_node)
        # 添加边（保持权重）
        for u, v, data in network.edges(data=True):
            new_u = permutation.get(u, u)
            new_v = permutation.get(v, v)
            # 深拷贝边数据，避免引用
            new_data = data.copy()
            new_network.add_edge(new_u, new_v, **new_data)
        return new_network

    def compute_orbit(self, network: nx.Graph) -> List[nx.Graph]:
        """计算认知状态的轨道（群作用下的所有像）。

        Args:
            network (nx.Graph): 初始网络。

        Returns:
            List[nx.Graph]: 轨道中的网络列表（去重）。
        """
        network_hash = self._network_hash(network)
        if network_hash in self.orbits_cache:
            return self.orbits_cache[network_hash]

        orbit_set = set()  # 存储哈希值，避免重复
        orbit_networks = []
        for g in self.group.automorphisms:
            transformed = self.apply_group_element(network, g)
            h = self._network_hash(transformed)
            if h not in orbit_set:
                orbit_set.add(h)
                orbit_networks.append(transformed)

        self.orbits_cache[network_hash] = orbit_networks
        return orbit_networks

    def compute_stabilizer(self, network: nx.Graph) -> List[Dict]:
        """计算认知状态的稳定子群（使网络保持不变的群元素）。

        Args:
            network (nx.Graph): 初始网络。

        Returns:
            List[Dict]: 稳定子中的自同构列表。
        """
        network_hash = self._network_hash(network)
        if network_hash in self.stabilizers_cache:
            return self.stabilizers_cache[network_hash]

        stabilizer = []
        for g in self.group.automorphisms:
            transformed = self.apply_group_element(network, g)
            if self._networks_equal(network, transformed):
                stabilizer.append(g)

        self.stabilizers_cache[network_hash] = stabilizer
        return stabilizer

    def verify_orbit_stabilizer_theorem(self, network: nx.Graph) -> bool:
        """验证轨道-稳定子定理：|轨道| = |群| / |稳定子|。

        Args:
            network (nx.Graph): 初始网络。

        Returns:
            bool: 定理是否成立（在给定容差内）。
        """
        if len(self.group.automorphisms) == 0:
            raise ValueError("群为空，无法验证定理")
        stabilizer = self.compute_stabilizer(network)
        if len(stabilizer) == 0:
            # 理论上至少应有恒等映射，若没有，说明实现有误
            raise ValueError("稳定子为空，可能自同构检测遗漏恒等映射")
        orbit = self.compute_orbit(network)
        expected = len(self.group.automorphisms) / len(stabilizer)
        # 期望值应为整数（根据拉格朗日定理）
        if not expected.is_integer():
            return False
        return len(orbit) == int(expected)

    def _network_hash(self, network: nx.Graph) -> str:
        """生成网络的简单哈希表示（用于缓存和去重）。

        Args:
            network (nx.Graph): 网络。

        Returns:
            str: 基于边权重排序的字符串表示。
        """
        edges = sorted([(u, v, network[u][v]['weight'])
                        for u, v in network.edges()])
        return str(edges)

    def _networks_equal(self, G1: nx.Graph, G2: nx.Graph, rtol: float = 1e-5) -> bool:
        """比较两个网络是否相等（考虑节点、边和权重）。

        Args:
            G1 (nx.Graph): 网络1。
            G2 (nx.Graph): 网络2。

        Returns:
            bool: 如果网络结构及权重完全相同返回True。
        """
        if set(G1.nodes()) != set(G2.nodes()):
            return False
        if G1.number_of_edges() != G2.number_of_edges():
            return False

        # 比较边和权重
        for u, v in G1.edges():
            if not G2.has_edge(u, v):
                return False
            w1 = G1[u][v].get('weight', 0.0)
            w2 = G2[u][v].get('weight', 0.0)
            if not np.isclose(w1, w2, rtol=rtol):
                return False
        return True


# 简单测试
if __name__ == "__main__":
    from algebra.cognitive_symmetry import CognitiveSymmetryGroup

    G = nx.Graph()
    nodes = ["A", "B"]
    G.add_nodes_from(nodes)
    G.add_edge("A", "B", weight=1.0)

    sym_group = CognitiveSymmetryGroup(G)
    sym_group.automorphisms = [{"A": "A", "B": "B"}, {"A": "B", "B": "A"}]  # 手动设置

    action = GroupActionOnCognitiveSpace(sym_group)
    orbit = action.compute_orbit(G)
    stabilizer = action.compute_stabilizer(G)
    print(f"轨道大小: {len(orbit)}")
    print(f"稳定子大小: {len(stabilizer)}")
    print(f"定理成立: {action.verify_orbit_stabilizer_theorem(G)}")