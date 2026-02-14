# algebra/lie_group_cognitive.py
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from scipy.linalg import expm


class LieAlgebraGenerator:
    """李代数生成元（对应基本认知操作）"""

    def __init__(self, name: str, effect_matrix: np.ndarray):
        self.name = name
        self.matrix = effect_matrix  # 操作的效果矩阵

    def __repr__(self):
        return f"LieAlgebraGenerator('{self.name}', shape={self.matrix.shape})"


class CognitiveLieGroup:
    """认知演化的李群描述"""

    def __init__(self, network_size: int):
        self.network_size = network_size
        self.generators = {}

        # 定义基本生成元
        self._initialize_generators()

    def _initialize_generators(self):
        """初始化李代数生成元"""
        n = self.network_size

        # 1. 能量优化算子 E - 使矩阵元素减小
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
        """使用李群演化认知网络"""
        if generator_coeffs is None:
            generator_coeffs = {'E': 0.5, 'C': 0.3, 'M': 0.2}

        # 构造李代数元素
        A = np.zeros((self.network_size, self.network_size))
        for gen_name, coeff in generator_coeffs.items():
            if gen_name in self.generators:
                A += coeff * self.generators[gen_name].matrix

        networks = [initial_network]
        current_network = initial_network

        for t in range(1, time_steps + 1):
            # 指数映射：exp(t * A)
            exp_tA = expm(t * 0.1 * A)  # 0.1为时间步长缩放

            # 应用变换到网络
            new_network = self._apply_lie_transform(current_network, exp_tA)
            networks.append(new_network)
            current_network = new_network

        return networks

    def _apply_lie_transform(self, network: nx.Graph, transform_matrix: np.ndarray) -> nx.Graph:
        """应用李变换到认知网络"""
        # 将网络的邻接矩阵与变换矩阵结合
        nodes = list(network.nodes())
        adj_matrix = nx.to_numpy_array(network, nodelist=nodes)

        # 应用变换（简化实现）
        transformed_adj = adj_matrix @ transform_matrix

        # 创建新网络
        new_network = nx.Graph()
        new_network.add_nodes_from(nodes)

        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                new_weight = transformed_adj[i, j]
                if new_weight > 0.01:  # 阈值
                    new_network.add_edge(nodes[i], nodes[j], weight=new_weight)

        return new_network