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
        # 创建新图
        new_network = nx.Graph()

        # 添加节点
        for node in network.nodes():
            new_node = permutation.get(node, node)
            new_network.add_node(new_node)

        # 添加边（应用置换）
        for u, v, data in network.edges(data=True):
            new_u = permutation.get(u, u)
            new_v = permutation.get(v, v)
            new_network.add_edge(new_u, new_v, **data)

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

        orbit = []
        for automorphism in self.group.automorphisms:
            transformed_network = self.apply_group_element(network, automorphism)
            orbit.append(transformed_network)

        # 去重
        unique_orbit = []
        seen_hashes = set()
        for net in orbit:
            net_hash = self._network_hash(net)
            if net_hash not in seen_hashes:
                seen_hashes.add(net_hash)
                unique_orbit.append(net)

        self.orbits_cache[network_hash] = unique_orbit
        return unique_orbit

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
        for automorphism in self.group.automorphisms:
            transformed_network = self.apply_group_element(network, automorphism)
            if self._networks_equal(network, transformed_network):
                stabilizer.append(automorphism)

        self.stabilizers_cache[network_hash] = stabilizer
        return stabilizer

    def verify_orbit_stabilizer_theorem(self, network: nx.Graph) -> bool:
        """验证轨道-稳定子定理：|轨道| = |群| / |稳定子|。

        Args:
            network (nx.Graph): 初始网络。

        Returns:
            bool: 定理是否成立（在给定容差内）。
        """
        orbit = self.compute_orbit(network)
        stabilizer = self.compute_stabilizer(network)

        expected_orbit_size = len(self.group.automorphisms) / max(1, len(stabilizer))
        actual_orbit_size = len(orbit)

        return np.isclose(expected_orbit_size, actual_orbit_size, rtol=0.1)

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

    def _networks_equal(self, net1: nx.Graph, net2: nx.Graph) -> bool:
        """比较两个网络是否相等（考虑节点、边和权重）。

        Args:
            net1 (nx.Graph): 网络1。
            net2 (nx.Graph): 网络2。

        Returns:
            bool: 如果网络结构及权重完全相同返回True。
        """
        if net1.number_of_nodes() != net2.number_of_nodes():
            return False

        if net1.number_of_edges() != net2.number_of_edges():
            return False

        # 比较边和权重
        for u, v, w1 in net1.edges(data='weight'):
            if not net2.has_edge(u, v):
                return False
            w2 = net2[u][v]['weight']
            if not np.isclose(w1, w2, rtol=1e-5):
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