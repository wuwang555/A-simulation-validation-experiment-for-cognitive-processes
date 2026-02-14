# algebra/group_action.py
import networkx as nx
import numpy as np
from typing import List, Dict, Set


class GroupActionOnCognitiveSpace:
    """群在认知状态空间上的作用"""

    def __init__(self, symmetry_group: 'CognitiveSymmetryGroup'):
        self.group = symmetry_group
        self.orbits_cache = {}
        self.stabilizers_cache = {}

    def apply_group_element(self, network: nx.Graph,
                            permutation: Dict) -> nx.Graph:
        """应用群元素（节点置换）到认知网络"""
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
        """计算认知状态的轨道"""
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
        """计算认知状态的稳定子群"""
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
        """验证轨道-稳定子定理：|轨道| = |群| / |稳定子|"""
        orbit = self.compute_orbit(network)
        stabilizer = self.compute_stabilizer(network)

        expected_orbit_size = len(self.group.automorphisms) / max(1, len(stabilizer))
        actual_orbit_size = len(orbit)

        return np.isclose(expected_orbit_size, actual_orbit_size, rtol=0.1)

    def _network_hash(self, network: nx.Graph) -> str:
        """网络的简单哈希表示"""
        edges = sorted([(u, v, network[u][v]['weight'])
                        for u, v in network.edges()])
        return str(edges)

    def _networks_equal(self, net1: nx.Graph, net2: nx.Graph) -> bool:
        """比较两个网络是否相等（考虑权重）"""
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