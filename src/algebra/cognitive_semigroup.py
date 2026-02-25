# algebra/cognitive_semigroup.py
"""
认知操作半群模块

根据论文定理4.1.3，基本认知操作在复合下构成一个半群。
该模块定义了认知操作（CognitiveOperation）以及由这些操作构成的半群
（CognitiveSemigroup），并提供了结合律验证等功能。
"""

import networkx as nx
from typing import List, Tuple, Callable
import numpy as np


class CognitiveOperation:
    """基本认知操作的代数表示。

    每个操作是一个可调用对象，接受一个认知网络并返回变换后的网络。

    Attributes:
        name (str): 操作名称（如"learning"）。
        operation (Callable): 具体的操作函数，签名应为
            func(network: nx.Graph, **kwargs) -> nx.Graph。
    """

    def __init__(self, name: str, operation_func: Callable):
        """
        Args:
            name (str): 操作名称。
            operation_func (Callable): 操作函数，应接受 network 和任意关键字参数，
                返回变换后的 network。
        """
        self.name = name
        self.operation = operation_func

    def __call__(self, network: nx.Graph, **kwargs):
        """调用操作函数。

        Args:
            network (nx.Graph): 输入认知网络。
            **kwargs: 传递给操作函数的额外参数（如学习率、路径等）。

        Returns:
            nx.Graph: 变换后的网络。
        """
        return self.operation(network, **kwargs)

    def __repr__(self):
        return f"CognitiveOperation('{self.name}')"


class CognitiveSemigroup:
    """认知操作半群 - 代数实现。

    该类维护一个操作字典，并提供操作复合与结合律验证的功能。
    操作复合定义为 (op2 ∘ op1)(G) = op2(op1(G))，即先执行 op1，再执行 op2。

    Attributes:
        operations (dict): 操作名称到 CognitiveOperation 对象的映射。
        composition_table (dict): 缓存已创建的复合操作，键为 (op1, op2)。
    """

    def __init__(self):
        self.operations = {}          # 操作名称 -> CognitiveOperation
        self.composition_table = {}    # (op1, op2) -> 复合后的操作名称

    def add_operation(self, name: str, operation_func: Callable) -> CognitiveOperation:
        """添加一个认知操作到半群中。

        Args:
            name (str): 操作名称，应唯一。
            operation_func (Callable): 操作函数。

        Returns:
            CognitiveOperation: 创建的操作对象。
        """
        op = CognitiveOperation(name, operation_func)
        self.operations[name] = op
        return op

    def compose(self, op1: str, op2: str) -> CognitiveOperation:
        """复合两个操作：返回 op2 ∘ op1 的复合操作（先执行 op1，再执行 op2）。

        Args:
            op1 (str): 第一个操作的名称。
            op2 (str): 第二个操作的名称。

        Returns:
            CognitiveOperation: 复合操作对象。
        """
        key = (op1, op2)
        if key not in self.composition_table:
            def composed_op(network, **kwargs):
                # 执行第一个操作
                result1 = self.operations[op1](network, **kwargs)
                # 执行第二个操作
                result2 = self.operations[op2](result1, **kwargs)
                return result2

            comp_name = f"{op1}∘{op2}"
            self.composition_table[key] = self.add_operation(comp_name, composed_op)

        return self.composition_table[key]

    def verify_associativity(self, op1: str, op2: str, op3: str, test_network: nx.Graph) -> bool:
        """验证结合律：(op3 ∘ op2) ∘ op1 = op3 ∘ (op2 ∘ op1)。

        通过比较两种复合方式作用在 test_network 上的总能耗是否近似相等来判断。

        Args:
            op1 (str): 第一个操作名称。
            op2 (str): 第二个操作名称。
            op3 (str): 第三个操作名称。
            test_network (nx.Graph): 用于测试的初始网络。

        Returns:
            bool: 如果结合律成立返回True，否则False。
        """
        left = self.compose(op1, self.compose(op2, op3).name)
        right = self.compose(self.compose(op1, op2).name, op3)

        # 在测试网络上验证
        result_left = left(test_network.copy())
        result_right = right(test_network.copy())

        # 比较网络结构（简化比较：总能量）
        left_energy = sum(result_left[u][v]['weight'] for u, v in result_left.edges())
        right_energy = sum(result_right[u][v]['weight'] for u, v in result_right.edges())

        return np.isclose(left_energy, right_energy, rtol=1e-5)

    def find_identity(self, test_network: nx.Graph) -> str:
        """寻找可能的单位元操作（可选，半群不一定有单位元）。

        单位元 e 满足：对任意操作 o，有 e∘o = o 且 o∘e = o。

        Args:
            test_network (nx.Graph): 用于测试的初始网络。

        Returns:
            str or None: 如果存在单位元，返回其名称；否则返回None。
        """
        for op_name, op in self.operations.items():
            identity = True
            for other_name, other_op in self.operations.items():
                # 检查 op ∘ other = other 且 other ∘ op = other
                comp1 = self.compose(other_name, op_name)
                comp2 = self.compose(op_name, other_name)

                result1 = comp1(test_network.copy())
                result2 = comp2(test_network.copy())
                result_other = other_op(test_network.copy())

                # 简化比较：能量相近
                energy1 = sum(result1[u][v]['weight'] for u, v in result1.edges())
                energy2 = sum(result2[u][v]['weight'] for u, v in result2.edges())
                energy_other = sum(result_other[u][v]['weight'] for u, v in result_other.edges())

                if not (np.isclose(energy1, energy_other, rtol=1e-5) and
                        np.isclose(energy2, energy_other, rtol=1e-5)):
                    identity = False
                    break

            if identity:
                return op_name
        return None


# 简单测试
if __name__ == "__main__":
    # 创建一个简单的网络用于测试
    G = nx.Graph()
    G.add_nodes_from(["A", "B", "C"])
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)

    semigroup = CognitiveSemigroup()

    # 定义两个简单操作
    def op1(net, **kwargs):
        net["A"]["B"]["weight"] *= 0.9
        return net

    def op2(net, **kwargs):
        net["B"]["C"]["weight"] *= 0.8
        return net

    semigroup.add_operation("op1", op1)
    semigroup.add_operation("op2", op2)

    comp = semigroup.compose("op1", "op2")
    print(f"复合操作名称: {comp.name}")
    # 验证结合律（需要三个操作，这里省略）
    print("测试完成。")