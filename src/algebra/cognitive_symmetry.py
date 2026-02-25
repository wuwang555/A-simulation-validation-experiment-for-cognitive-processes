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
from networkx.algorithms.isomorphism import GraphMatcher
import math
import random

np.random.seed(42)
random.seed(42)

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
        策略说明：
        - 节点数 ≤ 8：穷举全排列，但限制最多检查1000个（实际可能小于全排列数）。
        - 节点数 9~12：使用 networkx 的 GraphMatcher 精确搜索。
        - 节点数 > 12：随机采样排列，快速度序列过滤后验证同构性。

        Args:
            max_samples (int): 最大采样次数（用于大规模网络）。

        Returns:
            List[Dict]: 每个字典表示一个自同构映射 {原节点: 映射节点}。
        """
        nodes = list(self.network.nodes())
        n = len(nodes)
        automorphisms = []

        # ---------- 策略1：小规模网络（≤8），穷举全排列 ----------
        if n <= 8:
            permutations = itertools.permutations(range(n))
            total_perms = math.factorial(n)
            max_to_check = min(1000, total_perms)  # 最多检查1000个
            checked = 0
            for perm in permutations:
                if checked >= max_to_check:
                    break
                checked += 1
                # 快速筛选：度序列匹配
                if [self.network.degree(nodes[i]) for i in perm] != [self.network.degree(nodes[i]) for i in range(n)]:
                    continue
                if self._is_isomorphism(perm, nodes):
                    mapping = {nodes[i]: nodes[perm[i]] for i in range(n)}
                    automorphisms.append(mapping)
            # 穷举后添加恒等映射（以防万一，但理论上穷举会找到它）
            identity = {node: node for node in nodes}
            if identity not in automorphisms:
                automorphisms.append(identity)
            print(f"穷举检查了 {checked} 个排列，找到 {len(automorphisms)} 个自同构")
            return automorphisms

        # ---------- 策略2：中等规模（9 ≤ n ≤ 12），使用 GraphMatcher 精确搜索 ----------
        if 9 <= n <= 12:
            try:
                # 定义边匹配函数：边权重需近似相等
                def edge_match(e1_attr, e2_attr):
                    return np.isclose(e1_attr.get('weight', 0), e2_attr.get('weight', 0), rtol=0.2)

                GM = GraphMatcher(self.network, self.network,
                                  node_match=lambda n1, n2: True,  # 节点无额外属性，总是匹配
                                  edge_match=edge_match)

                for mapping in GM.isomorphisms_iter():
                    # mapping 已经是 {原节点: 映射节点} 的字典，直接添加
                    automorphisms.append(mapping)

            except Exception as e:
                print(f"GraphMatcher 出错: {e}，回退到随机采样")
                automorphisms = self._random_sample_automorphisms(nodes, max_samples)

        # ---------- 策略3：大规模网络（>12），随机采样 + 强制恒等 ----------
        automorphisms = self._random_sample_automorphisms(nodes, max_samples)
        # 确保恒等映射已加入
        nodes = list(self.network.nodes())
        identity = {node: node for node in nodes}
        if identity not in automorphisms:
            automorphisms.append(identity)
            print("强制加入了恒等映射（因随机采样可能漏掉）")

        # 关键：将结果保存到实例属性
        self.automorphisms = automorphisms

        return automorphisms

    def _is_isomorphism(self, perm, nodes):
        """检查排列 perm 是否保持边和权重（用于穷举和随机采样）。

        Args:
            perm (tuple): 节点的索引排列。
            nodes (list): 节点列表。

        Returns:
            bool: 如果是自同构返回True。
        """
        # 随机抽样部分边以加速（但恒等映射一定能通过）
        edges_to_check = list(self.network.edges())
        if len(edges_to_check) > 20:
            edges_to_check = random.sample(edges_to_check, 20)
        for u, v in edges_to_check:
            i, j = nodes.index(u), nodes.index(v)
            v1, v2 = nodes[perm[i]], nodes[perm[j]]
            if not self.network.has_edge(v1, v2):
                return False
            w_orig = self.network[u][v]['weight']
            w_perm = self.network[v1][v2]['weight']
            if not np.isclose(w_orig, w_perm, rtol=0.2):
                return False
        return True

    def _random_sample_automorphisms(self, nodes, max_samples):
        """随机采样排列，寻找自同构（不含恒等映射，因为概率太低）。

        Args:
            nodes (list): 节点列表。
            max_samples (int): 最大采样次数。

        Returns:
            list: 找到的自同构映射列表。
        """
        n = len(nodes)
        automorphisms = []
        degree_seq = [self.network.degree(node) for node in nodes]
        for _ in range(max_samples):
            perm = list(range(n))
            random.shuffle(perm)
            # 快速度序列过滤
            if [degree_seq[i] for i in perm] != degree_seq:
                continue
            if self._is_isomorphism(perm, nodes):
                mapping = {nodes[i]: nodes[perm[i]] for i in range(n)}
                automorphisms.append(mapping)
        return automorphisms

    def compute_conserved_quantities(self) -> Dict[str, float]:
        """计算认知系统的守恒量。

        根据Noether型命题，对称性对应守恒量。此处计算：
        - 全局认知能量（总权重）——对应时间平移对称性
        - 结构熵（度分布熵）——对应概念置换对称性
        - 分形维数（聚类系数/平均最短路径）——对应尺度变换对称性

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
                第二个元素为各守恒量的详细变化信息，包含 before、after、relative_change 和 conserved。
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