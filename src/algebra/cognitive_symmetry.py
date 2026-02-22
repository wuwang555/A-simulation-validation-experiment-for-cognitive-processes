# algebra/cognitive_symmetry.py
"""
认知对称群模块

根据论文第4.2节，认知对称群是保持认知网络关键性质不变的变换群。
该模块实现了概念同构检测、守恒量计算以及Noether型命题的验证。
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Any
import itertools
import math
import random


class CognitiveSymmetryGroup:
    """认知对称群实现。

    该类负责检测认知网络中的概念同构（节点置换保持边权重），计算守恒量
    （全局能量、结构熵、分形维数），并验证Noether型命题。

    Attributes:
        network (nx.Graph): 当前认知网络。
        automorphisms (List[Dict]): 已找到的自同构列表。
        conserved_quantities (Dict): 计算得到的守恒量。
    """

    def __init__(self, network: nx.Graph):
        """
        Args:
            network (nx.Graph): 待分析的认知网络。
        """
        self.network = network
        self.automorphisms = []
        self.conserved_quantities = {}

    def find_concept_isomorphisms(self, max_samples=1000) -> List[Dict]:
        """寻找概念同构（保持语义的节点置换）。

        由于同构检测是NP难问题，对于大规模网络采用随机采样近似。

        Args:
            max_samples (int): 最大采样次数（用于大规模网络）。

        Returns:
            List[Dict]: 每个字典表示一个自同构映射 {原节点: 映射节点}。
        """
        automorphisms = []
        nodes = list(self.network.nodes())
        n = len(nodes)

        if n <= 8:
            # 小规模网络：尝试所有排列
            permutations = itertools.permutations(range(n))
            total_perms = math.factorial(n)
            max_to_check = min(1000, total_perms)  # 最多检查1000个排列
        else:
            # 大规模网络：随机采样
            max_to_check = max_samples

        checked = 0
        isomorphisms_found = 0

        # 先计算节点的度序列，快速筛选
        degree_sequence = [self.network.degree(node) for node in nodes]

        while checked < max_to_check and isomorphisms_found < 50:  # 最多找50个同构
            if n <= 8:
                try:
                    perm = next(permutations)
                except StopIteration:
                    break
            else:
                # 随机生成排列
                perm = list(range(n))
                np.random.shuffle(perm)

            checked += 1

            # 快速检查：度序列必须匹配
            perm_degrees = [degree_sequence[i] for i in perm]
            if degree_sequence != perm_degrees:
                continue

            # 详细检查置换是否保持连接关系
            is_isomorphism = True
            # 只检查部分连接以减少计算量
            check_edges = min(20, n * (n - 1) // 2)

            edges_to_check = []
            if len(list(self.network.edges())) <= check_edges:
                edges_to_check = list(self.network.edges())
            else:
                # 随机选择边检查
                edges = list(self.network.edges())
                edges_to_check = random.sample(edges, check_edges)

            for u, v in edges_to_check:
                i = nodes.index(u)
                j = nodes.index(v)
                v1, v2 = nodes[perm[i]], nodes[perm[j]]

                # 检查边是否存在
                has_edge_original = self.network.has_edge(u, v)
                has_edge_permuted = self.network.has_edge(v1, v2)

                if has_edge_original != has_edge_permuted:
                    is_isomorphism = False
                    break

                # 如果边存在，检查权重是否相近
                if has_edge_original:
                    w_original = self.network[u][v]['weight']
                    w_permuted = self.network[v1][v2]['weight']
                    if not np.isclose(w_original, w_permuted, rtol=0.2):  # 放宽容差
                        is_isomorphism = False
                        break

            if is_isomorphism:
                permutation_map = {nodes[i]: nodes[perm[i]] for i in range(n)}
                automorphisms.append(permutation_map)
                isomorphisms_found += 1

        print(f"检查了 {checked} 个置换，找到 {len(automorphisms)} 个同构")
        self.automorphisms = automorphisms
        return automorphisms

    def compute_conserved_quantities(self) -> Dict[str, float]:
        """计算认知系统的守恒量。

        根据Noether型命题，对称性对应守恒量。此处计算：
        - 全局认知能量（总权重）
        - 结构熵（度分布熵）
        - 分形维数（聚类系数/平均最短路径）

        Returns:
            Dict[str, float]: 守恒量名称到值的映射。
        """
        conserved = {}

        # 1. 全局能量守恒（时间平移对称性）
        total_energy = sum(self.network[u][v]['weight']
                           for u, v in self.network.edges())
        conserved['total_energy'] = total_energy

        # 2. 结构熵守恒（概念置换对称性）
        degrees = [d for _, d in self.network.degree()]
        if degrees:
            degree_dist = np.histogram(degrees, bins=min(10, len(set(degrees))))[0]
            degree_dist = degree_dist / np.sum(degree_dist)
            entropy = -np.sum(degree_dist * np.log(degree_dist + 1e-10))
            conserved['structural_entropy'] = entropy
        else:
            conserved['structural_entropy'] = 0.0

        # 3. 分形维数守恒（尺度变换对称性）
        # 使用加权网络计算
        try:
            # 使用权重的倒数作为距离（能耗越高，距离越远）
            weighted_network = self.network.copy()
            for u, v in weighted_network.edges():
                weight = weighted_network[u][v]['weight']
                # 防止除零
                if weight > 0:
                    weighted_network[u][v]['distance'] = 1.0 / weight
                else:
                    weighted_network[u][v]['distance'] = 100.0  # 极大距离

            # 计算加权聚类系数
            if nx.is_connected(weighted_network):
                # 使用networkx的加权聚类系数
                clustering = nx.average_clustering(weighted_network, weight='distance')

                # 计算加权平均最短路径
                try:
                    # 使用Floyd-Warshall算法计算所有节点对的最短路径
                    import itertools
                    path_lengths = []
                    nodes = list(weighted_network.nodes())

                    for i, src in enumerate(nodes):
                        for dst in nodes[i + 1:]:
                            try:
                                # 使用Dijkstra算法计算最短路径
                                length = nx.dijkstra_path_length(weighted_network, src, dst, weight='distance')
                                path_lengths.append(length)
                            except nx.NetworkXNoPath:
                                # 如果没有路径，跳过
                                continue

                    if path_lengths:
                        avg_path_length = np.mean(path_lengths)
                        fractal_dim = clustering / (avg_path_length + 1e-8)
                        conserved['fractal_dimension'] = fractal_dim
                    else:
                        conserved['fractal_dimension'] = 0.0

                except Exception as e:
                    print(f"计算平均最短路径时出错: {e}")
                    conserved['fractal_dimension'] = 0.0
            else:
                conserved['fractal_dimension'] = 0.0

        except Exception as e:
            print(f"计算分形维数时出错: {e}")
            conserved['fractal_dimension'] = 0.0

        self.conserved_quantities = conserved
        return conserved

    def verify_noether_theorem(self, before_network: nx.Graph, after_network: nx.Graph,
                               transformation_type: str,
                               tolerance: float = 0.2) -> Tuple[bool, Dict]:
        """验证Noether型命题：对称性变换前后守恒量是否保持不变。

        Args:
            before_network (nx.Graph): 变换前的网络。
            after_network (nx.Graph): 变换后的网络。
            transformation_type (str): 变换类型描述（仅用于日志）。
            tolerance (float): 允许的相对变化阈值。

        Returns:
            Tuple[bool, Dict]: 第一个元素表示是否所有守恒量保持；
                第二个元素为各守恒量的详细变化信息。
        """
        before_group = CognitiveSymmetryGroup(before_network)
        after_group = CognitiveSymmetryGroup(after_network)

        before_conserved = before_group.compute_conserved_quantities()
        after_conserved = after_group.compute_conserved_quantities()

        all_conserved = True
        conservation_details = {}

        for key in before_conserved.keys():
            if key in after_conserved:
                before_val = before_conserved[key]
                after_val = after_conserved[key]

                # 处理可能为0的情况
                if abs(before_val) < 1e-10 and abs(after_val) < 1e-10:
                    is_conserved = True
                elif abs(before_val) < 1e-10:
                    is_conserved = False
                else:
                    # 相对变化
                    relative_change = abs(before_val - after_val) / abs(before_val)
                    is_conserved = relative_change < tolerance

                conservation_details[key] = {
                    'before': before_val,
                    'after': after_val,
                    'relative_change': relative_change if abs(before_val) > 1e-10 else float('inf'),
                    'conserved': is_conserved
                }

                if not is_conserved:
                    all_conserved = False
                    print(f"守恒量 {key} 变化超过阈值 {tolerance * 100:.0f}%: "
                          f"{before_val:.3f} -> {after_val:.3f} "
                          f"(变化: {relative_change * 100:.1f}%)")

        return all_conserved, conservation_details


# 简单测试
if __name__ == "__main__":
    G = nx.Graph()
    nodes = ["A", "B", "C"]
    G.add_nodes_from(nodes)
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("A", "C", weight=1.0)

    sym = CognitiveSymmetryGroup(G)
    autos = sym.find_concept_isomorphisms()
    conserved = sym.compute_conserved_quantities()
    print(f"找到 {len(autos)} 个自同构")
    print(f"守恒量: {conserved}")