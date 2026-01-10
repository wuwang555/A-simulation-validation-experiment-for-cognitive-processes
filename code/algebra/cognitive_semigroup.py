# algebra/cognitive_semigroup.py
import networkx as nx
from typing import List, Tuple, Callable
import numpy as np


class CognitiveOperation:
    """基本认知操作的代数表示"""

    def __init__(self, name: str, operation_func: Callable):
        self.name = name
        self.operation = operation_func

    def __call__(self, network: nx.Graph, **kwargs):
        return self.operation(network, **kwargs)

    def __repr__(self):
        return f"CognitiveOperation('{self.name}')"


class CognitiveSemigroup:
    """认知操作半群 - 代数实现"""

    def __init__(self):
        self.operations = {}
        self.composition_table = {}

    def add_operation(self, name: str, operation_func: Callable):
        """添加认知操作"""
        op = CognitiveOperation(name, operation_func)
        self.operations[name] = op
        return op

    def compose(self, op1: str, op2: str) -> CognitiveOperation:
        """复合操作：op2 ∘ op1（先执行op1，再执行op2）"""
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
        """验证结合律：(op3 ∘ op2) ∘ op1 = op3 ∘ (op2 ∘ op1)"""
        left = self.compose(op1, self.compose(op2, op3).name)
        right = self.compose(self.compose(op1, op2).name, op3)

        # 在测试网络上验证
        result_left = left(test_network.copy())
        result_right = right(test_network.copy())

        # 比较网络结构（简化比较）
        left_energy = sum(result_left[u][v]['weight'] for u, v in result_left.edges())
        right_energy = sum(result_right[u][v]['weight'] for u, v in result_right.edges())

        return np.isclose(left_energy, right_energy, rtol=1e-5)

    def find_identity(self, test_network: nx.Graph) -> str:
        """寻找单位元操作"""
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