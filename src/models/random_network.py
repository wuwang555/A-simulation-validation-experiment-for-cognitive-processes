"""
随机网络模型 - 无智能基准模型
完全随机地调整网络边权重，无优化目标
用于建立无智能机制的基线
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, Any
from core.cognitive_graph import BaseCognitiveGraph

np.random.seed(42)
random.seed(42)

class RandomNetworkModel(BaseCognitiveGraph):
    """随机网络模型 - 无智能基准。

    该类完全随机地调整网络权重，没有任何优化目标，用于与其他智能模型进行对比。
    """

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        """初始化随机网络模型。

        Args:
            individual_params (Dict[str, Any]): 个体参数（实际未使用）。
            network_seed (int): 随机种子。
        """
        super().__init__(individual_params, network_seed)
        self.random_weight_std = 0.25  # 增加随机权重调整的标准差
        self.random_activation_prob = 0.2  # 降低随机激活概率
        self.forgetting_enabled = False  # 禁用遗忘机制，使其更"随机"

    def initialize_random_network(self, num_nodes=51, connection_prob=0.2):
        """初始化随机网络。

        创建Erdos-Renyi随机图，并为边分配随机权重。

        Args:
            num_nodes (int): 节点数量。
            connection_prob (float): 边连接概率。
        """
        # 创建随机图
        self.G = nx.erdos_renyi_graph(num_nodes, connection_prob, seed=self.network_seed)

        # 为边分配随机权重
        for u, v in self.G.edges():
            weight = np.random.uniform(0.8, 2.0)  # 提高初始权重范围
            self.G[u][v]['weight'] = weight
            self.G[u][v]['original_weight'] = weight
            self.G[u][v]['traversal_count'] = 0
            self.last_activation_time[(u, v)] = 0

        # 为节点命名（使用简单编号）
        node_names = [f"概念_{i}" for i in range(num_nodes)]
        mapping = {i: node_names[i] for i in range(num_nodes)}
        self.G = nx.relabel_nodes(self.G, mapping)

        print(f"随机网络初始化: {num_nodes}节点, {self.G.number_of_edges()}条边")
        print(f"初始平均能耗: {self.calculate_network_energy():.3f}")

    def random_weight_adjustment(self):
        """随机权重调整 - 无智能机制。

        随机选择一条边，以50%概率增加或减少其权重。
        """
        if self.G.number_of_edges() == 0:
            return

        # 随机选择一条边
        edges = list(self.G.edges())
        u, v = random.choice(edges)

        # 生成随机变化 - 增加随机性，不偏向优化
        current_weight = self.G[u][v]['weight']

        # 50%概率增加，50%概率减少
        if random.random() < 0.5:
            random_change = np.random.uniform(0, self.random_weight_std)
            new_weight = min(3.0, current_weight + random_change)  # 上限3.0
        else:
            random_change = np.random.uniform(0, self.random_weight_std * 0.5)  # 减少的幅度小一些
            new_weight = max(0.1, current_weight - random_change)  # 下限0.1

        # 应用变化（无任何优化目标）
        self.G[u][v]['weight'] = new_weight

        # 随机记录激活时间
        if random.random() < self.random_activation_prob:
            self.last_activation_time[(u, v)] = self.iteration_count

    def random_traversal(self):
        """随机遍历 - 无目标导向。

        随机选择起点，随机行走若干步，并可能随机调整经过边的权重。
        """
        nodes = list(self.G.nodes())
        if len(nodes) < 2:
            return

        # 随机起点
        start_node = random.choice(nodes)
        path = [start_node]
        current_node = start_node

        # 随机步数
        path_length = random.randint(1, 3)  # 减少步数

        for step in range(path_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break

            next_node = random.choice(neighbors)
            if next_node not in path:
                path.append(next_node)
                current_node = next_node
            else:
                break

        # 记录遍历（但不对权重进行有目的调整）
        if len(path) >= 2:
            self.traversal_history.append({
                'path': path.copy(),
                'iteration': self.iteration_count,
                'type': 'random_walk'
            })

            # 随机决定是否对遍历的边进行权重调整
            if random.random() < 0.3:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if self.G.has_edge(u, v):
                        # 随机调整权重
                        current = self.G[u][v]['weight']
                        adjustment = np.random.uniform(-0.2, 0.2)
                        new_weight = max(0.1, min(3.0, current + adjustment))
                        self.G[u][v]['weight'] = new_weight

    def random_forgetting(self):
        """随机遗忘 - 无模式。

        随机选择边，以一定概率向原始权重回归。
        """
        if not self.forgetting_enabled:
            return

        current_time = self.iteration_count

        for u, v in self.G.edges():
            # 随机决定是否应用遗忘
            if random.random() < 0.05:  # 5%的概率应用遗忘
                current_energy = self.G[u][v]['weight']
                original_energy = self.G[u][v].get('original_weight', 2.0)

                # 随机遗忘因子，可能增加也可能减少
                forget_factor = random.uniform(-0.1, 0.1)
                new_energy = current_energy + (original_energy - current_energy) * forget_factor
                new_energy = max(0.1, min(3.0, new_energy))

                self.G[u][v]['weight'] = new_energy

    def random_monte_carlo_iteration(self, max_iterations=5000):
        """随机蒙特卡洛模拟 - 无智能机制。

        主循环，随机选择操作（权重调整、遍历、遗忘）并执行。

        Args:
            max_iterations (int): 最大迭代次数。

        Returns:
            float: 能耗变化百分比（可能为负）。
        """
        print(f"开始随机网络模拟: {max_iterations}次迭代")
        print("注意：这是一个无智能的基准模型，预期性能较差")

        initial_energy = self.calculate_network_energy()
        self.energy_history.append(initial_energy)

        for iteration in range(max_iterations):
            self.iteration_count += 1

            # 随机选择操作类型
            operation_choice = random.random()

            if operation_choice < 0.7:  # 70%概率随机权重调整
                self.random_weight_adjustment()
            elif operation_choice < 0.9:  # 20%概率随机遍历
                self.random_traversal()
            else:  # 10%概率随机遗忘
                self.random_forgetting()

            # 记录能量历史
            current_energy = self.calculate_network_energy()
            self.energy_history.append(current_energy)

            # 定期报告
            if iteration % 500 == 0:
                improvement = ((initial_energy - current_energy) / initial_energy * 100) if initial_energy > 0 else 0
                print(f"迭代 {iteration}: 网络能耗 = {current_energy:.3f} (变化: {improvement:.1f}%)")

        final_energy = self.calculate_network_energy()
        total_improvement = ((initial_energy - final_energy) / initial_energy * 100) if initial_energy > 0 else 0

        print(f"\n随机网络模拟完成!")
        print(f"初始能耗: {initial_energy:.3f}, 最终能耗: {final_energy:.3f}")
        print(f"总变化: {total_improvement:.1f}%")
        print(f"注意：正值表示能耗降低（改善），负值表示能耗增加（恶化）")

        return total_improvement

    def run_experiment(self, num_nodes=51, max_iterations=5000):
        """运行完整实验。

        Args:
            num_nodes (int): 节点数量。
            max_iterations (int): 最大迭代次数。

        Returns:
            dict: 实验结果字典。
        """
        self.initialize_random_network(num_nodes)
        improvement = self.random_monte_carlo_iteration(max_iterations)

        stats = self.get_network_stats()

        return {
            'model_type': 'random_network',
            'num_nodes': num_nodes,
            'iterations': max_iterations,
            'initial_energy': self.energy_history[0] if self.energy_history else 0,
            'final_energy': self.calculate_network_energy(),
            'improvement': improvement,
            'network_stats': stats,
            'note': '无智能基准模型，预期性能较差'
        }


if __name__ == "__main__":
    # 简单测试：运行小型随机网络
    print("测试 RandomNetworkModel...")
    params = {}
    model = RandomNetworkModel(params)
    result = model.run_experiment(num_nodes=51, max_iterations=8000)
    print("测试完成。改善率:", result['improvement'])