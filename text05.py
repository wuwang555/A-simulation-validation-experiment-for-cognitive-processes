import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from collections import defaultdict
import random

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MonteCarloCognitiveGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.traversal_history = []
        self.concept_centers = {}
        self.iteration_count = 0
        self.energy_history = []  # 记录平均能耗变化

    def initialize_graph(self, nodes, initial_edges):
        """初始化认知图"""
        self.G.add_nodes_from(nodes)

        for edge in initial_edges:
            u, v, weight = edge
            self.G.add_edge(u, v, weight=weight, traversal_count=0, original_weight=weight)

    def monte_carlo_iteration(self, max_iterations=1000,
                              hard_traversal_prob=0.6,
                              soft_traversal_prob=0.3,
                              compression_prob=0.05,
                              migration_prob=0.05):
        """
        蒙特卡洛模拟主循环

        Args:
            max_iterations: 最大迭代次数
            hard_traversal_prob: 硬遍历概率
            soft_traversal_prob: 软遍历概率
            compression_prob: 概念压缩概率
            migration_prob: 第一性原理迁移概率
        """
        print(f"开始蒙特卡洛模拟，最大迭代次数: {max_iterations}")

        for iteration in range(max_iterations):
            self.iteration_count += 1

            # 记录当前平均能耗
            current_avg_energy = self.get_average_energy()
            self.energy_history.append(current_avg_energy)

            # 随机选择操作类型
            rand_val = random.random()
            cumulative_prob = 0

            # 硬遍历
            cumulative_prob += hard_traversal_prob
            if rand_val <= cumulative_prob:
                self._random_hard_traversal()
                continue

            # 软遍历
            cumulative_prob += soft_traversal_prob
            if rand_val <= cumulative_prob:
                self._random_soft_traversal()
                continue

            # 概念压缩
            cumulative_prob += compression_prob
            if rand_val <= cumulative_prob:
                self._random_compression()
                continue

            # 第一性原理迁移
            cumulative_prob += migration_prob
            if rand_val <= cumulative_prob:
                self._random_migration()
                continue

            # 小概率无操作（探索）
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, 平均能耗: {current_avg_energy:.3f}")

    def _random_hard_traversal(self):
        """随机硬遍历 - 领域内学习"""
        # 随机选择一个领域内的短路径（2-3个节点）
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 2:
            return

        start_node = random.choice(available_nodes)

        # 找到与起始节点连接较好的节点（低能耗边）
        good_connections = []
        for neighbor in self.G.neighbors(start_node):
            if self.G[start_node][neighbor]['weight'] < 1.0:  # 能耗较低的连接
                good_connections.append(neighbor)

        if not good_connections:
            good_connections = list(self.G.neighbors(start_node))

        if not good_connections:
            return

        # 构建短路径
        path_length = random.randint(2, 3)
        path = [start_node]
        current_node = start_node

        for _ in range(path_length - 1):
            if not good_connections:
                break
            next_node = random.choice(good_connections)
            path.append(next_node)
            # 更新候选节点为当前节点的邻居
            good_connections = [n for n in self.G.neighbors(next_node)
                                if n not in path and self.G[next_node][n]['weight'] < 1.5]
            if not good_connections:
                break
            current_node = next_node

        if len(path) >= 2:
            self.traverse_path(path, traversal_type="hard")

    def _random_soft_traversal(self):
        """随机软遍历 - 跨领域探索"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 3:
            return

        # 选择两个相对遥远的节点
        start_node, end_node = random.sample(available_nodes, 2)

        # 计算当前直接连接的能耗（如果存在）
        direct_energy = float('inf')
        if self.G.has_edge(start_node, end_node):
            direct_energy = self.G[start_node][end_node]['weight']

        # 寻找中介节点（尝试找到比直接连接更好的路径）
        possible_mediators = [n for n in available_nodes
                              if n != start_node and n != end_node]

        best_path = None
        best_energy = direct_energy

        # 随机检查几个可能的中介节点
        for _ in range(min(5, len(possible_mediators))):
            mediator = random.choice(possible_mediators)
            possible_mediators.remove(mediator)

            if (self.G.has_edge(start_node, mediator) and
                    self.G.has_edge(mediator, end_node)):

                path_energy = (self.G[start_node][mediator]['weight'] +
                               self.G[mediator][end_node]['weight'])

                if path_energy < best_energy:
                    best_energy = path_energy
                    best_path = [start_node, mediator, end_node]

        # 如果找到更好的路径，就遍历它
        if best_path and len(best_path) == 3:
            self.traverse_path(best_path, traversal_type="soft")
        elif self.G.has_edge(start_node, end_node):
            # 否则遍历直接连接（如果存在）
            self.traverse_path([start_node, end_node], traversal_type="soft")

    def _random_compression(self):
        """随机概念压缩尝试"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 3:
            return

        # 随机选择一个候选中心节点
        center_candidate = random.choice(available_nodes)

        # 找到与该节点连接较好的邻居
        good_neighbors = []
        for neighbor in self.G.neighbors(center_candidate):
            if self.G[center_candidate][neighbor]['weight'] < 1.0:
                good_neighbors.append(neighbor)

        # 需要至少2个好的连接才能考虑压缩
        if len(good_neighbors) >= 2:
            # 随机选择部分邻居进行压缩
            num_to_compress = random.randint(2, min(4, len(good_neighbors)))
            nodes_to_compress = random.sample(good_neighbors, num_to_compress)

            compression_strength = random.uniform(0.3, 0.7)
            self.conceptual_compression(center_candidate, nodes_to_compress,
                                        compression_strength)

    def _random_migration(self):
        """随机第一性原理迁移尝试"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 4:
            return

        # 随机选择起点和终点
        start_node, end_node = random.sample(available_nodes, 2)

        # 寻找可能的基础原理节点
        principle_candidates = [n for n in available_nodes
                                if n not in [start_node, end_node]]

        if not principle_candidates:
            return

        # 随机选择1-3个原理节点尝试
        num_principles = random.randint(1, min(3, len(principle_candidates)))
        selected_principles = random.sample(principle_candidates, num_principles)

        exploration_bonus = random.uniform(0.05, 0.15)
        self.first_principles_migration(start_node, end_node,
                                        selected_principles, exploration_bonus)

    def traverse_path(self, path, traversal_type="hard"):
        """模拟遍历路径，降低相关边的能耗"""
        self.traversal_history.append((path, traversal_type, self.iteration_count))

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                # 遍历次数增加
                self.G[u][v]['traversal_count'] += 1

                # 根据遍历类型调整学习速率
                if traversal_type == "hard":
                    learning_rate = random.uniform(0.7, 0.9)  # 硬遍历学习更快
                else:
                    learning_rate = random.uniform(0.85, 0.95)  # 软遍历学习较慢

                # 添加随机噪声模拟真实学习过程
                noise = random.gauss(0, 0.02)
                learning_rate = max(0.6, min(0.98, learning_rate + noise))

                # 能耗随着遍历次数增加而降低（学习效应）
                current_weight = self.G[u][v]['weight']
                new_weight = max(0.05, current_weight * learning_rate)
                self.G[u][v]['weight'] = new_weight

    def conceptual_compression(self, center_node, related_nodes, compression_strength=0.5):
        """概念压缩：强化中心节点与相关节点的连接"""
        if len(related_nodes) < 2:
            return False

        # 强化中心节点与每个相关节点的连接
        for node in related_nodes:
            if self.G.has_edge(center_node, node):
                current_energy = self.G[center_node][node]['weight']
                compressed_energy = max(0.03, current_energy * compression_strength)
                self.G[center_node][node]['weight'] = compressed_energy

        # 记录概念压缩关系
        self.concept_centers[center_node] = {
            'related_nodes': related_nodes,
            'compression_strength': compression_strength,
            'iteration': self.iteration_count
        }

        return True

    def first_principles_migration(self, start_node, end_node, principle_nodes, exploration_bonus=0.1):
        """第一性原理迁移：通过基础原理节点发现跨领域低能耗路径"""
        best_path = None
        best_energy = float('inf')

        # 检查直接连接
        if self.G.has_edge(start_node, end_node):
            direct_energy = self.G[start_node][end_node]['weight']
            best_path = [start_node, end_node]
            best_energy = direct_energy

        # 尝试通过每个原理节点建立连接
        for principle in principle_nodes:
            if (self.G.has_edge(start_node, principle) and
                    self.G.has_edge(principle, end_node)):

                path_energy = (self.G[start_node][principle]['weight'] +
                               self.G[principle][end_node]['weight'])

                # 应用探索奖励
                adjusted_energy = path_energy - exploration_bonus

                if adjusted_energy < best_energy:
                    best_energy = adjusted_energy
                    best_path = [start_node, principle, end_node]

        # 如果找到比直接连接更好的路径，则强化该路径
        if best_path and len(best_path) > 2:
            # 强化迁移路径上的连接
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                current = self.G[u][v]['weight']
                new_energy = max(0.05, current * random.uniform(0.6, 0.8))
                self.G[u][v]['weight'] = new_energy

            # 记录迁移关系
            principle_node = best_path[1]
            if 'migration_bridges' not in self.G.nodes[principle_node]:
                self.G.nodes[principle_node]['migration_bridges'] = []

            self.G.nodes[principle_node]['migration_bridges'].append({
                'from': start_node,
                'to': end_node,
                'energy_saving': best_energy,
                'iteration': self.iteration_count
            })

            # 模拟遍历这条新发现的优化路径
            self.traverse_path(best_path, traversal_type="soft")

            return best_path

        return None

    def get_average_energy(self):
        """计算网络平均能耗"""
        if self.G.number_of_edges() == 0:
            return 0
        energies = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        return np.mean(energies)

    def get_network_stats(self):
        """获取网络统计信息"""
        stats = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'iterations': self.iteration_count,
            'avg_energy': self.get_average_energy(),
            'compression_centers': len(self.concept_centers),
            'migration_bridges': 0
        }

        # 计算迁移桥梁数量
        for node in self.G.nodes():
            if 'migration_bridges' in self.G.nodes[node]:
                stats['migration_bridges'] += len(self.G.nodes[node]['migration_bridges'])

        return stats

    def visualize_energy_convergence(self):
        """可视化能耗收敛过程"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.energy_history, 'b-', alpha=0.7)
        plt.xlabel('迭代次数')
        plt.ylabel('平均认知能耗')
        plt.title('蒙特卡洛模拟 - 认知能耗收敛过程')
        plt.grid(True, alpha=0.3)

        # 标记重要事件
        for center, info in self.concept_centers.items():
            iteration = info['iteration']
            if iteration < len(self.energy_history):
                plt.axvline(x=iteration, color='red', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

    def visualize_graph(self, title="认知图", figsize=(12, 8)):
        """可视化认知图"""
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42)

        # 设置节点颜色和大小
        node_colors = []
        node_sizes = []
        for node in self.G.nodes():
            if node in self.concept_centers:
                node_colors.append('red')  # 概念压缩中心
                node_sizes.append(2000)
            elif any('migration_bridges' in self.G.nodes[n] for n in self.G.nodes()):
                node_colors.append('orange')  # 迁移桥梁节点
                node_sizes.append(1500)
            else:
                node_colors.append('lightblue')
                node_sizes.append(1000)

        # 设置边颜色和宽度
        edge_colors = []
        edge_widths = []
        for u, v in self.G.edges():
            energy = self.G[u][v]['weight']
            # 边宽度与能耗成反比
            edge_widths.append(max(1, 4 - energy * 2))

            # 边颜色基于能耗
            if energy < 0.3:
                edge_colors.append('green')  # 低能耗
            elif energy < 0.7:
                edge_colors.append('blue')  # 中能耗
            else:
                edge_colors.append('gray')  # 高能耗

        # 绘制图形
        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes,
                               node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(self.G, pos, width=edge_widths,
                               alpha=0.7, edge_color=edge_colors)
        nx.draw_networkx_labels(self.G, pos, font_size=9,
                                font_family='SimHei')

        plt.title(title, fontsize=16, fontfamily='SimHei')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # 打印统计信息
        stats = self.get_network_stats()
        print(f"网络统计:")
        print(f"  节点: {stats['nodes']}, 边: {stats['edges']}")
        print(f"  迭代次数: {stats['iterations']}")
        print(f"  平均能耗: {stats['avg_energy']:.3f}")
        print(f"  概念压缩中心: {stats['compression_centers']}")
        print(f"  迁移桥梁: {stats['migration_bridges']}")


def create_sample_network():
    """创建示例网络"""
    # 定义节点 - 包含多个知识领域
    nodes = [
        # 物理学领域
        "牛顿定律", "运动学", "力学", "能量守恒", "动量",
        # 数学领域
        "微积分", "线性代数", "概率论", "优化理论",
        # 计算机领域
        "算法", "数据结构", "机器学习", "神经网络",
        # 基础原理节点（用于第一性原理迁移）
        "优化", "变换", "迭代", "抽象", "模式识别"
    ]

    # 初始化边（基于语义相关性设置初始能耗）
    np.random.seed(42)
    initial_edges = []

    # 领域分组
    physics_nodes = ["牛顿定律", "运动学", "力学", "能量守恒", "动量"]
    math_nodes = ["微积分", "线性代数", "概率论", "优化理论"]
    cs_nodes = ["算法", "数据结构", "机器学习", "神经网络"]
    principle_nodes = ["优化", "变换", "迭代", "抽象", "模式识别"]

    all_nodes = physics_nodes + math_nodes + cs_nodes + principle_nodes

    # 创建连接
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            u, v = all_nodes[i], all_nodes[j]

            # 基于领域关系设置初始能耗
            if (u in physics_nodes and v in physics_nodes) or \
                    (u in math_nodes and v in math_nodes) or \
                    (u in cs_nodes and v in cs_nodes):
                # 同一领域内：低能耗
                weight = round(np.random.uniform(0.4, 1.0), 2)
            elif (u in principle_nodes or v in principle_nodes):
                # 与原理节点的连接：中等能耗
                weight = round(np.random.uniform(0.6, 1.3), 2)
            else:
                # 跨领域：高能耗
                weight = round(np.random.uniform(1.2, 2.0), 2)

            initial_edges.append((u, v, weight))

    return nodes, initial_edges


def run_monte_carlo_experiment():
    """运行蒙特卡洛实验"""
    # 创建认知图实例
    cog_graph = MonteCarloCognitiveGraph()

    # 初始化网络
    nodes, edges = create_sample_network()
    cog_graph.initialize_graph(nodes, edges)

    print("=== 初始状态 ===")
    initial_stats = cog_graph.get_network_stats()
    print(f"初始平均能耗: {initial_stats['avg_energy']:.3f}")
    cog_graph.visualize_graph("初始认知图")

    # 运行蒙特卡洛模拟
    print("\n=== 开始蒙特卡洛模拟 ===")
    cog_graph.monte_carlo_iteration(
        max_iterations=2000,
        hard_traversal_prob=0.5,  # 50%概率硬遍历
        soft_traversal_prob=0.3,  # 30%概率软遍历
        compression_prob=0.1,  # 10%概率概念压缩
        migration_prob=0.1  # 10%概率第一性原理迁移
    )

    # 显示结果
    print("\n=== 模拟结果 ===")
    final_stats = cog_graph.get_network_stats()
    improvement = ((initial_stats['avg_energy'] - final_stats['avg_energy']) /
                   initial_stats['avg_energy'] * 100)

    print(f"初始平均能耗: {initial_stats['avg_energy']:.3f}")
    print(f"最终平均能耗: {final_stats['avg_energy']:.3f}")
    print(f"能耗降低: {improvement:.1f}%")
    print(f"发现的概念压缩中心: {list(cog_graph.concept_centers.keys())}")

    # 可视化结果
    cog_graph.visualize_graph("蒙特卡洛模拟后的认知图")
    cog_graph.visualize_energy_convergence()

    # 显示一些发现的迁移路径
    print("\n=== 发现的重要迁移路径 ===")
    for node in cog_graph.G.nodes():
        if 'migration_bridges' in cog_graph.G.nodes[node]:
            bridges = cog_graph.G.nodes[node]['migration_bridges']
            for bridge in bridges[:2]:  # 显示前两个
                print(f"  {bridge['from']} -> {node} -> {bridge['to']} "
                      f"(节能: {bridge['energy_saving']:.2f})")


if __name__ == "__main__":
    run_monte_carlo_experiment()