# algebra/lie_group_cognitive.py
"""
认知演化的李群描述模块

根据论文第4.4节，连续时间认知演化可用李群方程 dG/dt = A(t)G(t) 描述。
该模块实现了李代数生成元（基本认知操作）以及基于指数映射的演化过程。
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from scipy.linalg import expm

np.random.seed(42)

class LieAlgebraGenerator:
    """李代数生成元，对应基本认知操作。

    Attributes:
        name (str): 生成元名称（如 "E" 表示能量优化）。
        matrix (np.ndarray): 效果矩阵，表示该操作对网络的影响。
    """

    def __init__(self, name: str, effect_matrix: np.ndarray):
        """
        Args:
            name (str): 生成元名称。
            effect_matrix (np.ndarray): 形状为 (n, n) 的矩阵，作用于邻接矩阵。
        """
        self.name = name
        self.matrix = effect_matrix  # 操作的效果矩阵

    def __repr__(self):
        return f"LieAlgebraGenerator('{self.name}', shape={self.matrix.shape})"


class CognitiveLieGroup:
    """认知演化的李群描述。

    该类管理李代数生成元（能量优化算子 E、概念压缩算子 C、原理迁移算子 M），
    并提供演化方法：通过指数映射 exp(tA) 将初始网络沿指定方向演化。

    Attributes:
        network_size (int): 网络的节点数量。
        generators (Dict[str, LieAlgebraGenerator]): 生成元字典。
    """

    def __init__(self, network_size: int):
        """
        Args:
            network_size (int): 认知网络的节点数，用于生成对应大小的矩阵。
        """
        self.network_size = network_size
        self.generators = {}

        # 定义基本生成元
        self._initialize_generators()

    def _initialize_generators(self):
        """初始化李代数生成元：能量优化(E)、概念压缩(C)、原理迁移(M)。"""
        n = self.network_size

        # 1. 能量优化算子 E - 使矩阵元素减小（负方向变化）
        E_matrix = np.random.uniform(-0.1, -0.01, (n, n))
        np.fill_diagonal(E_matrix, 0)  # 对角线为0
        self.generators['E'] = LieAlgebraGenerator('能量优化', E_matrix)

        # 2. 概念压缩算子 C - 增加相关性（使某些元素更接近）
        C_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 压缩效应：使相似节点更接近
                    C_matrix[i, j] = np.random.uniform(-0.05, 0.05)
        self.generators['C'] = LieAlgebraGenerator('概念压缩', C_matrix)

        # 3. 原理迁移算子 M - 建立新连接
        M_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 迁移效应：中等强度的变化
                    M_matrix[i, j] = np.random.uniform(-0.08, 0.08)
        self.generators['M'] = LieAlgebraGenerator('原理迁移', M_matrix)

    def evolve_network(self, initial_network: nx.Graph,
                       time_steps: int = 10,
                       generator_coeffs: Dict[str, float] = None) -> List[nx.Graph]:
        """使用李群演化认知网络。

        根据演化方程 dG/dt = A(t)G(t)，其中 A(t) 由生成元线性组合得到。
        采用指数映射进行离散时间演化：G(t+Δt) = exp(Δt * A) G(t)。
        此处 Δt 固定为 0.1。

        Args:
            initial_network (nx.Graph): 初始认知网络。
            time_steps (int): 演化步数。
            generator_coeffs (Dict[str, float], optional): 各生成元的系数，
                如 {'E':0.7, 'C':0.2, 'M':0.1}。默认平衡系数。

        Returns:
            List[nx.Graph]: 演化过程中的网络快照列表（包含初始网络）。
        """
        if generator_coeffs is None:
            generator_coeffs = {'E': 0.5, 'C': 0.3, 'M': 0.2}

        # 构造李代数元素 A = Σ coeff_i * generator_i.matrix
        A = np.zeros((self.network_size, self.network_size))
        for gen_name, coeff in generator_coeffs.items():
            if gen_name in self.generators:
                A += coeff * self.generators[gen_name].matrix

        networks = [initial_network]
        current_network = initial_network

        for t in range(1, time_steps + 1):
            # 指数映射：exp(t * A) （此处用固定步长0.1，可调整）
            exp_tA = expm(t * 0.1 * A)

            # 应用变换到网络
            new_network = self._apply_lie_transform(current_network, exp_tA)
            networks.append(new_network)
            current_network = new_network

        return networks

    def _apply_lie_transform(self, network: nx.Graph, transform_matrix: np.ndarray) -> nx.Graph:
        """应用李变换到认知网络：使用变换矩阵更新邻接矩阵。

        Args:
            network (nx.Graph): 输入网络。
            transform_matrix (np.ndarray): 变换矩阵（指数映射结果）。

        Returns:
            nx.Graph: 变换后的网络。
        """
        # 将网络的邻接矩阵与变换矩阵结合
        nodes = list(network.nodes())
        adj_matrix = nx.to_numpy_array(network, nodelist=nodes)

        # 应用变换（简化实现：邻接矩阵右乘变换矩阵）
        transformed_adj = adj_matrix @ transform_matrix

        # 创建新网络
        new_network = nx.Graph()
        new_network.add_nodes_from(nodes)

        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                new_weight = transformed_adj[i, j]
                if new_weight > 0.01:  # 阈值，仅保留大于0.01的连接
                    new_network.add_edge(nodes[i], nodes[j], weight=new_weight)

        return new_network


# 简单测试
if __name__ == "__main__":
    # 创建一个3节点完全图
    G = nx.complete_graph(["A", "B", "C"])
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0

    lie_group = CognitiveLieGroup(3)
    evolved = lie_group.evolve_network(G, time_steps=3, generator_coeffs={'E': 0.8, 'C': 0.1, 'M': 0.1})
    print(f"演化步数: {len(evolved)}")
    for i, net in enumerate(evolved):
        print(f"Step {i}: edges = {list(net.edges(data=True))}")