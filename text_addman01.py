import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from collections import defaultdict
import random
import math
import json
import requests
from collections import defaultdict
import jieba
import numpy as np
import time
from enum import Enum
from typing import Dict, Any

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedCognitiveGraph:
    def __init__(self, forgetting_rate=0.002, base_learning_rate=0.85):
        self.G = nx.Graph()
        self.traversal_history = []
        self.concept_centers = {}
        self.iteration_count = 0
        self.energy_history = []

        # 改进的参数 - 增强遗忘机制
        self.forgetting_rate = forgetting_rate  # 显著增强遗忘率
        self.base_learning_rate = base_learning_rate
        self.semantic_similarity_matrix = {}
        self.last_activation_time = {}

        # 领域分组定义
        self.physics_nodes = ["牛顿定律", "运动学", "力学", "能量守恒", "动量"]
        self.math_nodes = ["微积分", "线性代数", "概率论", "优化理论"]
        self.cs_nodes = ["算法", "数据结构", "机器学习", "神经网络"]
        self.principle_nodes = ["优化", "变换", "迭代", "抽象", "模式识别"]

        self.all_nodes = self.physics_nodes + self.math_nodes + self.cs_nodes + self.principle_nodes

    def calculate_semantic_similarity(self, node1, node2):
        """计算节点间的语义相似度"""
        key = tuple(sorted([node1, node2]))
        if key in self.semantic_similarity_matrix:
            return self.semantic_similarity_matrix[key]

        # 基于领域关系的相似度计算
        same_domain = 0
        if (node1 in self.physics_nodes and node2 in self.physics_nodes) or \
                (node1 in self.math_nodes and node2 in self.math_nodes) or \
                (node1 in self.cs_nodes and node2 in self.cs_nodes):
            same_domain = 1.0
        elif (node1 in self.principle_nodes and node2 in self.principle_nodes):
            same_domain = 0.8
        elif ((node1 in self.principle_nodes and node2 not in self.principle_nodes) or
              (node2 in self.principle_nodes and node1 not in self.principle_nodes)):
            same_domain = 0.5
        else:
            same_domain = 0.3  # 跨领域基础相似度

        # 添加随机变异使网络更真实
        similarity = max(0.1, min(0.9, same_domain + random.uniform(-0.15, 0.15)))
        self.semantic_similarity_matrix[key] = similarity
        return similarity

    def initialize_graph(self, nodes, initial_edges):
        """初始化认知图"""
        self.G.add_nodes_from(nodes)

        for edge in initial_edges:
            u, v, weight = edge
            self.G.add_edge(u, v, weight=weight, traversal_count=0, original_weight=weight)
            self.last_activation_time[(u, v)] = 0

    def monte_carlo_iteration(self, max_iterations=5000,
                              hard_traversal_prob=0.5,
                              soft_traversal_prob=0.3,
                              compression_prob=0.05,  # 降低压缩概率
                              migration_prob=0.15):
        """
        蒙特卡洛模拟主循环 - 使用增强的遗忘机制
        """
        print(f"开始蒙特卡洛模拟，最大迭代次数: {max_iterations}")
        print(f"遗忘率: {self.forgetting_rate}, 基础学习率: {self.base_learning_rate}")

        for iteration in range(max_iterations):
            self.iteration_count += 1

            # 应用增强的遗忘机制
            self._apply_forgetting()

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

            # 概念压缩 - 更严格的触发条件
            cumulative_prob += compression_prob
            if (rand_val <= cumulative_prob and
                    self.iteration_count > max_iterations * 0.4):  # 提高触发门槛
                self._random_compression()
                continue

            # 第一性原理迁移 - 更严格的触发条件
            cumulative_prob += migration_prob
            if (rand_val <= cumulative_prob and
                    self.iteration_count > max_iterations * 0.5):  # 提高触发门槛
                self._random_migration()
                continue

            # 每200次迭代打印进度
            if iteration % 500 == 0:
                stats = self.get_network_stats()
                print(f"迭代 {iteration}, 平均能耗: {current_avg_energy:.3f}, "
                      f"压缩中心: {stats['compression_centers']}, 迁移桥梁: {stats['migration_bridges']}")

    def _apply_forgetting(self):
        """应用增强的遗忘机制到所有边"""
        current_time = self.iteration_count

        for u, v in self.G.edges():
            time_since_last_activation = current_time - self.last_activation_time.get((u, v), 0)
            if time_since_last_activation > 0:
                current_energy = self.G[u][v]['weight']
                similarity = self.calculate_semantic_similarity(u, v)

                # 使用增强的遗忘时间函数
                forget_factor = self.forgetting_function(
                    current_time,
                    self.last_activation_time.get((u, v), 0),
                    current_energy,
                    similarity
                )

                new_weight = self.G[u][v]['weight'] * (1 + forget_factor)
                # 但不会超过原始能耗
                original = self.G[u][v].get('original_weight', 2.0)
                self.G[u][v]['weight'] = min(new_weight, original)

    def forgetting_function(self, current_time, last_activation_time, current_energy, similarity):
        """
        增强的遗忘时间函数

        主要改进：
        1. 更小的时间尺度参数（500而不是1000）
        2. 降低相似度保护强度
        3. 更高的基础遗忘率
        """
        # 计算时间间隔
        time_gap = current_time - last_activation_time

        # 更强的遗忘：使用更小的时间尺度参数
        base_forgetting = 1 - math.exp(-time_gap / 500)  # 从1000改为500

        # 当前能耗影响：高能耗边更容易被遗忘
        energy_factor = 0.5 + 0.5 * (current_energy / 2.0)

        # 降低相似度保护强度
        similarity_protection = 1 - (similarity * 0.5)  # 从0.7改为0.5

        # 综合遗忘因子
        forgetting_factor = (base_forgetting * energy_factor *
                             similarity_protection * self.forgetting_rate)

        # 限制最大遗忘率
        return min(forgetting_factor, 0.1)  # 从0.3改为0.1

    def traverse_path(self, path, traversal_type="hard"):
        """改进的遍历函数 - 包含相似度影响"""
        self.traversal_history.append((path, traversal_type, self.iteration_count))
        current_time = self.iteration_count

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                # 更新最后激活时间
                self.last_activation_time[(u, v)] = current_time
                self.G[u][v]['traversal_count'] += 1

                # 基于相似度的学习速率
                similarity = self.calculate_semantic_similarity(u, v)
                base_rate = self.base_learning_rate

                if traversal_type == "hard":
                    # 硬遍历：相似度高的学习更快
                    learning_rate = base_rate * (0.7 + 0.3 * similarity)
                else:
                    # 软遍历：基础学习速率，受相似度影响较小
                    learning_rate = base_rate * 0.9

                # 非线性学习效应：当前能耗越高，学习效果越明显
                current_weight = self.G[u][v]['weight']
                learning_effect = learning_rate * (current_weight / 2.0)

                # 应用学习
                new_weight = max(0.05, current_weight * (1 - learning_effect))
                self.G[u][v]['weight'] = new_weight

    def _random_hard_traversal(self):
        """随机硬遍历 - 领域内学习"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 2:
            return

        start_node = random.choice(available_nodes)

        # 基于相似度找到与起始节点连接较好的节点
        good_connections = []
        for neighbor in self.G.neighbors(start_node):
            similarity = self.calculate_semantic_similarity(start_node, neighbor)
            if similarity > 0.6:  # 相似度阈值
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
            # 更新候选节点为当前节点的相似邻居
            good_connections = [n for n in self.G.neighbors(next_node)
                                if n not in path and
                                self.calculate_semantic_similarity(next_node, n) > 0.5]
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

        # 选择两个相对遥远的节点（基于相似度）
        start_node = random.choice(available_nodes)

        # 找到与起始节点相似度较低的节点作为目标
        low_similarity_nodes = []
        for node in available_nodes:
            if node != start_node:
                similarity = self.calculate_semantic_similarity(start_node, node)
                if similarity < 0.4:
                    low_similarity_nodes.append(node)

        if not low_similarity_nodes:
            low_similarity_nodes = [n for n in available_nodes if n != start_node]

        end_node = random.choice(low_similarity_nodes)

        # 寻找中介节点
        possible_mediators = [n for n in available_nodes
                              if n != start_node and n != end_node]

        best_path = None
        best_energy = float('inf')

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

        # 如果找到路径，就遍历它
        if best_path:
            self.traverse_path(best_path, traversal_type="soft")
        elif self.G.has_edge(start_node, end_node):
            self.traverse_path([start_node, end_node], traversal_type="soft")

    def _random_compression(self):
        """随机概念压缩尝试 - 更严格的触发条件"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 3:
            return

        # 进一步降低压缩概率
        if random.random() > 0.02:  # 只有2%的概率真正触发压缩
            return

        # 随机选择一个候选中心节点
        center_candidate = random.choice(available_nodes)

        # 找到与该节点连接较好且语义相似的邻居
        good_neighbors = []
        for neighbor in self.G.neighbors(center_candidate):
            similarity = self.calculate_semantic_similarity(center_candidate, neighbor)
            if (self.G[center_candidate][neighbor]['weight'] < 0.8 and
                    similarity > 0.6):
                good_neighbors.append(neighbor)

        # 需要至少3个好的连接才能考虑压缩
        if len(good_neighbors) >= 3:
            # 随机选择部分邻居进行压缩
            num_to_compress = random.randint(2, min(3, len(good_neighbors)))
            nodes_to_compress = random.sample(good_neighbors, num_to_compress)

            compression_strength = random.uniform(0.4, 0.6)
            self.conceptual_compression(center_candidate, nodes_to_compress,
                                        compression_strength)

    def conceptual_compression(self, center_node, related_nodes, compression_strength=0.5):
        """概念压缩：强化中心节点与相关节点的连接"""
        if len(related_nodes) < 2:
            return False

        # 强化中心节点与每个相关节点的连接
        for node in related_nodes:
            if self.G.has_edge(center_node, node):
                current_energy = self.G[center_node][node]['weight']
                compressed_energy = max(0.05, current_energy * compression_strength)
                self.G[center_node][node]['weight'] = compressed_energy

        # 记录概念压缩关系
        self.concept_centers[center_node] = {
            'related_nodes': related_nodes,
            'compression_strength': compression_strength,
            'iteration': self.iteration_count
        }

        return True

    def _random_migration(self):
        """随机第一性原理迁移尝试 - 更严格的触发条件"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 4:
            return

        # 进一步降低迁移概率
        if random.random() > 0.03:  # 只有3%的概率真正触发迁移
            return

        # 随机选择起点和终点
        start_node, end_node = random.sample(available_nodes, 2)

        # 要求起点和终点有足够低的相似度（真正需要迁移）
        similarity = self.calculate_semantic_similarity(start_node, end_node)
        if similarity > 0.4:
            return

        # 寻找可能的基础原理节点
        principle_candidates = [n for n in self.principle_nodes
                                if n in available_nodes and n not in [start_node, end_node]]

        if not principle_candidates:
            return

        # 随机选择1-2个原理节点尝试
        num_principles = random.randint(1, min(2, len(principle_candidates)))
        selected_principles = random.sample(principle_candidates, num_principles)

        exploration_bonus = random.uniform(0.05, 0.15)
        self.first_principles_migration(start_node, end_node,
                                        selected_principles, exploration_bonus)

    def first_principles_migration(self, start_node, end_node, principle_nodes, exploration_bonus=0.1):
        """第一性原理迁移：通过基础原理节点发现跨领域低能耗路径"""
        best_path = None
        best_energy = float('inf')

        # 检查直接连接
        direct_energy = float('inf')
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

        # 要求新路径必须明显优于直接路径（至少20%改进）
        improvement_threshold = 0.2
        if (best_path and len(best_path) > 2 and
                best_energy < direct_energy * (1 - improvement_threshold)):

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
                'energy_saving': direct_energy - best_energy,
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

        # 使用滑动平均使曲线更平滑
        window_size = min(100, len(self.energy_history) // 10)
        if window_size > 1:
            energy_smooth = np.convolve(self.energy_history, np.ones(window_size) / window_size, mode='valid')
            plt.plot(range(len(energy_smooth)), energy_smooth, 'b-', alpha=0.7, linewidth=1.5, label='滑动平均')
            plt.plot(self.energy_history, 'gray', alpha=0.3, linewidth=0.5, label='原始值')
        else:
            plt.plot(self.energy_history, 'b-', alpha=0.7, linewidth=1, label='能耗')

        plt.xlabel('迭代次数')
        plt.ylabel('平均认知能耗')
        plt.title('增强遗忘机制 - 认知能耗收敛过程')
        plt.grid(True, alpha=0.3)

        # 标记重要事件
        colors = ['red', 'green', 'orange', 'purple']
        for i, (center, info) in enumerate(self.concept_centers.items()):
            iteration = info['iteration']
            if iteration < len(self.energy_history):
                color = colors[i % len(colors)]
                plt.axvline(x=iteration, color=color, alpha=0.5, linestyle='--',
                            label=f'压缩: {center}' if i < 3 else "")

        # 标记迁移事件
        migration_iterations = []
        for node in self.G.nodes():
            if 'migration_bridges' in self.G.nodes[node]:
                for bridge in self.G.nodes[node]['migration_bridges']:
                    migration_iterations.append(bridge['iteration'])

        for i, iteration in enumerate(migration_iterations[:3]):  # 只标记前3个
            if iteration < len(self.energy_history):
                plt.axvline(x=iteration, color='blue', alpha=0.5, linestyle=':',
                            label=f'迁移事件' if i == 0 else "")

        if len(self.concept_centers) > 0 or len(migration_iterations) > 0:
            plt.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.show()

    def visualize_graph(self, title="认知图", figsize=(12, 8)):
        """可视化认知图"""
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42)

        # 设置节点颜色和大小
        node_colors = []
        node_sizes = []
        node_labels = {}
        for node in self.G.nodes():
            if node in self.concept_centers:
                node_colors.append('red')  # 概念压缩中心
                node_sizes.append(2000)
                node_labels[node] = f"{node}*"  # 标记压缩中心
            elif any('migration_bridges' in self.G.nodes[n] for n in self.G.nodes()):
                node_colors.append('orange')  # 迁移桥梁节点
                node_sizes.append(1500)
                node_labels[node] = node
            else:
                node_colors.append('lightblue')
                node_sizes.append(800)
                node_labels[node] = node

        # 设置边颜色和宽度
        edge_colors = []
        edge_widths = []
        for u, v in self.G.edges():
            energy = self.G[u][v]['weight']
            # 边宽度与能耗成反比
            edge_widths.append(max(0.5, 3 - energy * 1.5))

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
                               alpha=0.6, edge_color=edge_colors)
        nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=8,
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
    # 使用类中定义的节点
    physics_nodes = ["牛顿定律", "运动学", "力学", "能量守恒", "动量"]
    math_nodes = ["微积分", "线性代数", "概率论", "优化理论"]
    cs_nodes = ["算法", "数据结构", "机器学习", "神经网络"]
    principle_nodes = ["优化", "变换", "迭代", "抽象", "模式识别"]

    all_nodes = physics_nodes + math_nodes + cs_nodes + principle_nodes

    # 初始化边（基于语义相关性设置初始能耗）
    np.random.seed(42)
    initial_edges = []

    # 创建连接
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            u, v = all_nodes[i], all_nodes[j]

            # 基于领域关系设置初始能耗
            if (u in physics_nodes and v in physics_nodes) or \
                    (u in math_nodes and v in math_nodes) or \
                    (u in cs_nodes and v in cs_nodes):
                # 同一领域内：低能耗
                weight = round(np.random.uniform(0.5, 1.2), 2)
            elif (u in principle_nodes or v in principle_nodes):
                # 与原理节点的连接：中等能耗
                weight = round(np.random.uniform(0.7, 1.5), 2)
            else:
                # 跨领域：高能耗
                weight = round(np.random.uniform(1.0, 2.0), 2)

            initial_edges.append((u, v, weight))

    return all_nodes, initial_edges


def run_improved_experiment():
    """运行改进后的实验"""
    # 创建改进的认知图实例 - 使用增强的遗忘机制
    cog_graph = ImprovedCognitiveGraph(
        forgetting_rate=0.002,  # 显著增强遗忘
        base_learning_rate=0.85
    )

    # 初始化网络
    nodes, edges = create_sample_network()
    cog_graph.initialize_graph(nodes, edges)

    print("=== 初始状态 ===")
    initial_stats = cog_graph.get_network_stats()
    print(f"初始平均能耗: {initial_stats['avg_energy']:.3f}")
    cog_graph.visualize_graph("初始认知图")

    # 运行蒙特卡洛模拟
    print("\n=== 开始增强遗忘机制的蒙特卡洛模拟 ===")
    cog_graph.monte_carlo_iteration(
        max_iterations=5000,  # 使用5000次迭代
        hard_traversal_prob=0.5,
        soft_traversal_prob=0.3,
        compression_prob=0.05,  # 降低压缩概率
        migration_prob=0.15
    )

    # 显示结果
    print("\n=== 模拟结果 ===")
    final_stats = cog_graph.get_network_stats()
    improvement = ((initial_stats['avg_energy'] - final_stats['avg_energy']) /
                   initial_stats['avg_energy'] * 100)

    print(f"初始平均能耗: {initial_stats['avg_energy']:.3f}")
    print(f"最终平均能耗: {final_stats['avg_energy']:.3f}")
    print(f"能耗降低: {improvement:.1f}%")

    if cog_graph.concept_centers:
        print(f"发现的概念压缩中心: {list(cog_graph.concept_centers.keys())}")
    else:
        print("未发现概念压缩中心")

    # 可视化结果
    cog_graph.visualize_graph("增强遗忘机制后的认知图")
    cog_graph.visualize_energy_convergence()

    # 显示发现的迁移路径
    print("\n=== 发现的重要迁移路径 ===")
    meaningful_bridges = 0
    for node in cog_graph.G.nodes():
        if 'migration_bridges' in cog_graph.G.nodes[node]:
            bridges = cog_graph.G.nodes[node]['migration_bridges']
            for bridge in bridges:
                # 只显示有显著节能的路径
                if bridge['energy_saving'] > 0.3:
                    print(f"  {bridge['from']} -> {node} -> {bridge['to']} "
                          f"(节能: {bridge['energy_saving']:.2f})")
                    meaningful_bridges += 1

    if meaningful_bridges == 0:
        print("  未发现具有显著节能效果的迁移路径")
    else:
        print(f"  共发现 {meaningful_bridges} 条有意义的迁移路径")

    # 分析能耗收敛情况
    print("\n=== 能耗收敛分析 ===")
    if len(cog_graph.energy_history) > 100:
        last_100_avg = np.mean(cog_graph.energy_history[-100:])
        first_100_avg = np.mean(cog_graph.energy_history[:100])
        convergence_ratio = (first_100_avg - last_100_avg) / first_100_avg
        print(f"前100次迭代平均能耗: {first_100_avg:.3f}")
        print(f"后100次迭代平均能耗: {last_100_avg:.3f}")
        print(f"收敛稳定性: {'良好' if convergence_ratio > 0.3 else '需要调整'}")


class IndividualVariation:
    """个体差异模拟器"""

    def __init__(self, base_parameters: Dict[str, Any], variation_ranges: Dict[str, Any]):
        """
        Args:
            base_parameters: 基础参数值
            variation_ranges: 每个参数的变异范围
        """
        self.base_parameters = base_parameters
        self.variation_ranges = variation_ranges
        self.individual_parameters = {}

    def generate_individual(self, individual_id: str):
        """为单个个体生成参数"""
        params = {}
        for param, base_value in self.base_parameters.items():
            if param in self.variation_ranges:
                variation = self.variation_ranges[param]
                if isinstance(variation, (int, float)):
                    # 简单数值变异
                    min_val = base_value * (1 - variation)
                    max_val = base_value * (1 + variation)
                    params[param] = np.random.uniform(min_val, max_val)
                elif isinstance(variation, tuple) and len(variation) == 2:
                    # 指定范围的变异
                    params[param] = np.random.uniform(variation[0], variation[1])
                else:
                    params[param] = base_value
            else:
                params[param] = base_value

        self.individual_parameters[individual_id] = params
        return params


class StochasticCognitiveGraph(ImprovedCognitiveGraph):
    """带有个体差异的随机认知图"""

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        """
        Args:
            individual_params: 个体特定的参数
            network_seed: 网络结构的随机种子（保持初始认知相似）
        """
        # 使用个体特定参数初始化
        super().__init__(
            forgetting_rate=individual_params.get('forgetting_rate', 0.002),
            base_learning_rate=individual_params.get('base_learning_rate', 0.85)
        )

        self.individual_params = individual_params
        self.network_seed = network_seed

        # 个体特定的随机行为参数
        self.hard_traversal_bias = individual_params.get('hard_traversal_bias', 0.0)
        self.soft_traversal_bias = individual_params.get('soft_traversal_bias', 0.0)
        self.compression_bias = individual_params.get('compression_bias', 0.0)
        self.migration_bias = individual_params.get('migration_bias', 0.0)

        # 学习率的个体差异
        self.learning_rate_variation = individual_params.get('learning_rate_variation', 0.1)

    def traverse_path(self, path, traversal_type="hard"):
        """带有个体差异的遍历函数"""
        self.traversal_history.append((path, traversal_type, self.iteration_count))
        current_time = self.iteration_count

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                # 更新最后激活时间
                self.last_activation_time[(u, v)] = current_time
                self.G[u][v]['traversal_count'] += 1

                # 基于相似度的学习速率 + 个体差异
                similarity = self.calculate_semantic_similarity(u, v)
                base_rate = self.base_learning_rate

                # 个体学习率变异
                individual_learning_variation = np.random.uniform(
                    1 - self.learning_rate_variation,
                    1 + self.learning_rate_variation
                )

                if traversal_type == "hard":
                    # 硬遍历：相似度高的学习更快 + 个体偏好
                    learning_rate = base_rate * (0.7 + 0.3 * similarity) * individual_learning_variation
                else:
                    # 软遍历：基础学习速率 + 个体偏好
                    learning_rate = base_rate * 0.9 * individual_learning_variation

                # 非线性学习效应
                current_weight = self.G[u][v]['weight']
                learning_effect = learning_rate * (current_weight / 2.0)

                # 应用学习
                new_weight = max(0.05, current_weight * (1 - learning_effect))
                self.G[u][v]['weight'] = new_weight

    def monte_carlo_iteration(self, max_iterations=5000):
        """带有个体行为偏好的蒙特卡洛模拟"""
        # 基于个体偏好调整概率
        base_hard = 0.5 + self.hard_traversal_bias
        base_soft = 0.3 + self.soft_traversal_bias
        base_compression = 0.05 + self.compression_bias
        base_migration = 0.15 + self.migration_bias

        # 归一化概率
        total = base_hard + base_soft + base_compression + base_migration
        hard_traversal_prob = base_hard / total
        soft_traversal_prob = base_soft / total
        compression_prob = base_compression / total
        migration_prob = base_migration / total

        print(f"个体行为偏好 - 硬遍历: {hard_traversal_prob:.2f}, 软遍历: {soft_traversal_prob:.2f}, "
              f"压缩: {compression_prob:.2f}, 迁移: {migration_prob:.2f}")

        # 调用父类的蒙特卡洛模拟
        super().monte_carlo_iteration(
            max_iterations=max_iterations,
            hard_traversal_prob=hard_traversal_prob,
            soft_traversal_prob=soft_traversal_prob,
            compression_prob=compression_prob,
            migration_prob=migration_prob
        )


def create_individual_network(individual_id, individual_params, network_seed=42):
    """为个体创建认知网络（保持初始结构相似）"""
    np.random.seed(network_seed)  # 固定网络结构随机种子

    # 使用相同的节点和边创建逻辑
    physics_nodes = ["牛顿定律", "运动学", "力学", "能量守恒", "动量"]
    math_nodes = ["微积分", "线性代数", "概率论", "优化理论"]
    cs_nodes = ["算法", "数据结构", "机器学习", "神经网络"]
    principle_nodes = ["优化", "变换", "迭代", "抽象", "模式识别"]

    all_nodes = physics_nodes + math_nodes + cs_nodes + principle_nodes

    # 初始化边（基于语义相关性设置初始能耗）
    initial_edges = []

    # 创建连接（使用固定种子保证初始结构一致）
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            u, v = all_nodes[i], all_nodes[j]

            # 基于领域关系设置初始能耗
            if (u in physics_nodes and v in physics_nodes) or \
                    (u in math_nodes and v in math_nodes) or \
                    (u in cs_nodes and v in cs_nodes):
                # 同一领域内：低能耗
                weight = round(np.random.uniform(0.5, 1.2), 2)
            elif (u in principle_nodes or v in principle_nodes):
                # 与原理节点的连接：中等能耗
                weight = round(np.random.uniform(0.7, 1.5), 2)
            else:
                # 跨领域：高能耗
                weight = round(np.random.uniform(1.0, 2.0), 2)

            initial_edges.append((u, v, weight))

    # 创建个体认知图
    individual_graph = StochasticCognitiveGraph(individual_params, network_seed)
    individual_graph.initialize_graph(all_nodes, initial_edges)

    return individual_graph


def run_population_experiment(num_individuals=10, max_iterations=2000):
    """运行群体实验"""
    # 基础参数定义
    base_parameters = {
        'forgetting_rate': 0.002,
        'base_learning_rate': 0.85,
        'hard_traversal_bias': 0.0,
        'soft_traversal_bias': 0.0,
        'compression_bias': 0.0,
        'migration_bias': 0.0,
        'learning_rate_variation': 0.1
    }

    # 参数变异范围（±20%的变异）
    variation_ranges = {
        'forgetting_rate': 0.2,  # ±20%
        'base_learning_rate': 0.1,  # ±10%
        'hard_traversal_bias': (-0.1, 0.1),  # 行为偏好范围
        'soft_traversal_bias': (-0.1, 0.1),
        'compression_bias': (-0.03, 0.03),
        'migration_bias': (-0.05, 0.05),
        'learning_rate_variation': (0.05, 0.15)
    }

    # 创建个体差异模拟器
    variation_simulator = IndividualVariation(base_parameters, variation_ranges)

    population_results = []

    print(f"=== 开始群体实验：{num_individuals}个个体 ===")

    for i in range(num_individuals):
        individual_id = f"个体_{i + 1}"
        print(f"\n--- 模拟 {individual_id} ---")

        # 生成个体参数
        individual_params = variation_simulator.generate_individual(individual_id)

        # 创建个体认知网络（使用相同种子保证初始结构一致）
        individual_graph = create_individual_network(individual_id, individual_params)

        # 运行模拟
        initial_energy = individual_graph.get_average_energy()
        individual_graph.monte_carlo_iteration(max_iterations=max_iterations)

        # 收集结果
        final_stats = individual_graph.get_network_stats()
        improvement = ((initial_energy - final_stats['avg_energy']) / initial_energy * 100)

        result = {
            'individual_id': individual_id,
            'parameters': individual_params,
            'initial_energy': initial_energy,
            'final_energy': final_stats['avg_energy'],
            'improvement': improvement,
            'compression_centers': final_stats['compression_centers'],
            'migration_bridges': final_stats['migration_bridges'],
            'concept_centers': list(individual_graph.concept_centers.keys())
        }

        population_results.append(result)

        print(f"{individual_id} 结果:")
        print(f"  能耗降低: {improvement:.1f}%")
        print(f"  压缩中心: {result['compression_centers']}个")
        print(f"  迁移桥梁: {result['migration_bridges']}个")

    # 分析群体统计
    analyze_population_results(population_results)

    return population_results


def analyze_population_results(results):
    """分析群体结果"""
    print(f"\n=== 群体统计结果 ===")

    improvements = [r['improvement'] for r in results]
    compressions = [r['compression_centers'] for r in results]
    migrations = [r['migration_bridges'] for r in results]

    print(f"能耗降低统计:")
    print(f"  平均: {np.mean(improvements):.1f}%")
    print(f"  标准差: {np.std(improvements):.1f}%")
    print(f"  范围: {min(improvements):.1f}% - {max(improvements):.1f}%")

    print(f"概念压缩统计:")
    print(f"  平均: {np.mean(compressions):.1f}个")
    print(f"  范围: {min(compressions)} - {max(compressions)}个")

    print(f"迁移桥梁统计:")
    print(f"  平均: {np.mean(migrations):.1f}个")
    print(f"  范围: {min(migrations)} - {max(migrations)}个")

    # 分析参数与结果的相关性
    print(f"\n=== 个体差异分析 ===")

    # 找出最优和最差个体
    best_individual = max(results, key=lambda x: x['improvement'])
    worst_individual = min(results, key=lambda x: x['improvement'])

    print(f"最优个体: {best_individual['individual_id']} (能耗降低: {best_individual['improvement']:.1f}%)")
    print(f"最差个体: {worst_individual['individual_id']} (能耗降低: {worst_individual['improvement']:.1f}%)")

    return {
        'mean_improvement': np.mean(improvements),
        'std_improvement': np.std(improvements),
        'mean_compressions': np.mean(compressions),
        'mean_migrations': np.mean(migrations)
    }


class CognitiveState(Enum):
    """认知状态枚举"""
    FOCUSED = "专注状态"  # 高主观能耗，适合深度思考
    EXPLORATORY = "探索状态"  # 中等主观能耗，适合广度探索
    FATIGUED = "疲劳状态"  # 低主观能耗，认知受限
    INSPIRED = "灵感状态"  # 可变主观能耗，创造性思维


class SubjectiveCognitiveGraph(StochasticCognitiveGraph):
    """带主观认知状态的认知图"""

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        super().__init__(individual_params, network_seed)

        # 主观认知状态参数
        self.current_state = CognitiveState.FOCUSED
        self.subjective_energy = 1.0  # 当前主观认知能耗预算 (0-1范围)
        self.energy_history = []  # 主观能耗历史

        # 状态转移参数
        self.state_transition_matrix = {
            CognitiveState.FOCUSED: {
                CognitiveState.EXPLORATORY: 0.3,
                CognitiveState.FATIGUED: 0.1,  # 从0.2降低到0.1
                CognitiveState.INSPIRED: 0.2,  # 从0.1提高到0.2
                CognitiveState.FOCUSED: 0.4
            },
            CognitiveState.EXPLORATORY: {
                CognitiveState.FOCUSED: 0.3,
                CognitiveState.FATIGUED: 0.2,  # 从0.3降低到0.2
                CognitiveState.INSPIRED: 0.2,
                CognitiveState.EXPLORATORY: 0.3  # 从0.2提高到0.3
            },
            CognitiveState.FATIGUED: {
                CognitiveState.FOCUSED: 0.3,  # 从0.1提高到0.3
                CognitiveState.EXPLORATORY: 0.4,  # 从0.2提高到0.4
                CognitiveState.INSPIRED: 0.1,  # 从0.05提高到0.1
                CognitiveState.FATIGUED: 0.2  # 从0.65大幅降低到0.2
            },
            CognitiveState.INSPIRED: {
                CognitiveState.FOCUSED: 0.4,
                CognitiveState.EXPLORATORY: 0.3,
                CognitiveState.FATIGUED: 0.1,
                CognitiveState.INSPIRED: 0.2
            }
        }

        # 状态对应的主观能耗范围
        self.state_energy_ranges = {
            CognitiveState.FOCUSED: (1.5, 2.5),  # 大幅提高专注状态能耗
            CognitiveState.EXPLORATORY: (1.0, 1.8),  # 提高探索状态能耗
            CognitiveState.FATIGUED: (0.8, 1.2),  # 大幅提高疲劳状态能耗
            CognitiveState.INSPIRED: (2.0, 3.0)  # 大幅提高灵感状态能耗
        }

        # 硬遍历和软遍历的能耗分配策略
        self.hard_traversal_energy_ratio = 0.6  # 硬遍历专注方向能耗占比
        self.soft_traversal_energy_ratio = 0.4  # 软遍历扩散方向能耗占比

    def update_cognitive_state(self):
        """更新主观认知状态"""
        current_state = self.current_state
        transition_probs = self.state_transition_matrix[current_state]

        # 基于概率进行状态转移
        rand_val = random.random()
        cumulative_prob = 0

        for new_state, prob in transition_probs.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                if new_state != current_state:
                    self.current_state = new_state
                    # 状态改变时更新主观能耗
                    self._update_subjective_energy()
                break

        # 记录状态和能耗
        self.energy_history.append({
            'iteration': self.iteration_count,
            'state': self.current_state,
            'energy': self.subjective_energy
        })

    def _update_subjective_energy(self):
        """根据当前状态更新主观认知能耗"""
        energy_range = self.state_energy_ranges[self.current_state]
        self.subjective_energy = random.uniform(energy_range[0], energy_range[1])

        # 添加个体差异
        energy_variation = self.individual_params.get('energy_variation', 0.1)
        self.subjective_energy *= random.uniform(1 - energy_variation, 1 + energy_variation)
        self.subjective_energy = max(0.1, min(1.5, self.subjective_energy))  # 合理范围

    def can_traverse_edge(self, edge_energy, traversal_type):
        """检查是否可以遍历某条边（考虑主观认知能耗）"""
        if traversal_type == "hard":
            # 降低硬遍历的能耗要求
            required_energy = edge_energy * 0.8  # 从1.2降低到0.8
            available_energy = self.subjective_energy * self.hard_traversal_energy_ratio
        else:
            # 降低软遍历的能耗要求
            required_energy = edge_energy * 0.6  # 从0.8降低到0.6
            available_energy = self.subjective_energy * self.soft_traversal_energy_ratio

        return available_energy >= required_energy, available_energy - required_energy

    def traverse_path(self, path, traversal_type="hard"):
        """改进的遍历函数 - 考虑主观认知状态"""
        # 更新认知状态
        if random.random() < 0.1:  # 10%的概率状态变化
            self.update_cognitive_state()

        # 检查整个路径是否可遍历
        total_required_energy = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                edge_energy = self.G[u][v]['weight']
                total_required_energy += edge_energy

        # 基于路径总能耗和当前状态决定是否遍历
        can_traverse, energy_balance = self.can_traverse_edge(total_required_energy, traversal_type)

        if not can_traverse:
            # 认知阻塞：主观能耗不足，无法完成遍历
            if random.random() < 0.3:  # 30%的概率触发状态恶化
                self.current_state = CognitiveState.FATIGUED
                self._update_subjective_energy()
            return  # 无法遍历，直接返回

        # 可以遍历，执行原有逻辑
        super().traverse_path(path, traversal_type)

        # 遍历后可能的状态变化
        self._post_traversal_state_update(traversal_type, energy_balance)

    def _post_traversal_state_update(self, traversal_type, energy_balance):
        """遍历后的状态更新"""
        if energy_balance > 0.3:
            # 能耗余额充足，可能进入更好的状态
            if traversal_type == "hard" and random.random() < 0.4:
                self.current_state = CognitiveState.FOCUSED
            elif traversal_type == "soft" and random.random() < 0.3:
                self.current_state = CognitiveState.EXPLORATORY
        elif energy_balance < -0.2:
            # 能耗透支，可能进入疲劳状态
            if random.random() < 0.5:
                self.current_state = CognitiveState.FATIGUED

        self._update_subjective_energy()

    def find_available_paths(self, start_node, traversal_type="hard", max_path_length=3):
        """基于当前主观状态找到可用的路径"""
        available_paths = []

        if traversal_type == "hard":
            # 硬遍历：深度优先，专注一个方向
            self._hard_traversal_search(start_node, [], available_paths, max_path_length)
        else:
            # 软遍历：广度优先，探索多个方向
            self._soft_traversal_search(start_node, [], available_paths, max_path_length)

        return available_paths

    def _hard_traversal_search(self, current_node, current_path, available_paths, max_length):
        """硬遍历路径搜索"""
        if len(current_path) >= max_length:
            available_paths.append(current_path.copy())
            return

        current_path.append(current_node)

        # 找到能耗最低的边作为硬遍历方向
        neighbors = list(self.G.neighbors(current_node))
        if not neighbors:
            available_paths.append(current_path.copy())
            return

        # 按能耗排序，选择最低能耗的边
        neighbors.sort(key=lambda n: self.G[current_node][n]['weight'])

        for neighbor in neighbors[:2]:  # 只考虑前2个最低能耗的邻居
            if neighbor not in current_path:
                edge_energy = self.G[current_node][neighbor]['weight']
                can_traverse, _ = self.can_traverse_edge(edge_energy, "hard")
                if can_traverse:
                    self._hard_traversal_search(neighbor, current_path, available_paths, max_length)

        current_path.pop()

    def _soft_traversal_search(self, current_node, current_path, available_paths, max_length):
        """软遍历路径搜索"""
        if len(current_path) >= max_length:
            available_paths.append(current_path.copy())
            return

        current_path.append(current_node)

        neighbors = list(self.G.neighbors(current_node))
        if not neighbors:
            available_paths.append(current_path.copy())
            return

        # 软遍历考虑所有方向，但受主观能耗限制
        for neighbor in neighbors:
            if neighbor not in current_path:
                edge_energy = self.G[current_node][neighbor]['weight']
                can_traverse, _ = self.can_traverse_edge(edge_energy, "soft")
                if can_traverse:
                    self._soft_traversal_search(neighbor, current_path, available_paths, max_length)

        current_path.pop()

    def monte_carlo_iteration(self, max_iterations=5000):
        """改进的蒙特卡洛模拟 - 考虑主观认知状态"""
        print(f"初始认知状态: {self.current_state.value}, 主观能耗: {self.subjective_energy:.2f}")

        for iteration in range(max_iterations):
            self.iteration_count += 1

            # 每100次迭代更新一次状态
            if iteration % 100 == 0:
                self.update_cognitive_state()

            # 应用遗忘机制
            self._apply_forgetting()

            # 记录当前状态
            current_avg_energy = self.get_average_energy()
            self.energy_history.append({
                'iteration': self.iteration_count,
                'state': self.current_state,
                'energy': self.subjective_energy,
                'network_energy': current_avg_energy
            })

            # 基于当前状态选择操作类型
            operation = self._select_operation_based_on_state()

            if operation == "hard_traversal":
                self._state_based_hard_traversal()
            elif operation == "soft_traversal":
                self._state_based_soft_traversal()
            elif operation == "compression":
                self._random_compression()
            elif operation == "migration":
                self._random_migration()

            # 进度报告
            if iteration % 500 == 0:
                stats = self.get_network_stats()
                print(f"迭代 {iteration}, 状态: {self.current_state.value}, "
                      f"主观能耗: {self.subjective_energy:.2f}, 网络能耗: {current_avg_energy:.3f}")

    def _select_operation_based_on_state(self):
        """基于认知状态选择操作类型 - 调整概率"""
        state_operations = {
            CognitiveState.FOCUSED: {
                "hard_traversal": 0.5,  # 从0.6降低
                "soft_traversal": 0.3,  # 从0.2提高
                "compression": 0.1,
                "migration": 0.1
            },
            CognitiveState.EXPLORATORY: {
                "hard_traversal": 0.3,  # 从0.2提高
                "soft_traversal": 0.4,  # 从0.5降低
                "compression": 0.1,
                "migration": 0.2
            },
            CognitiveState.FATIGUED: {
                "hard_traversal": 0.2,  # 从0.1提高
                "soft_traversal": 0.4,  # 从0.3提高
                "compression": 0.2,  # 从0.3降低
                "migration": 0.2  # 从0.3降低
            },
            CognitiveState.INSPIRED: {
                "hard_traversal": 0.3,
                "soft_traversal": 0.4,
                "compression": 0.1,
                "migration": 0.2
            }
        }
    def _state_based_hard_traversal(self):
        """基于状态的硬遍历"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return

        start_node = random.choice(available_nodes)
        available_paths = self.find_available_paths(start_node, "hard", 3)

        if available_paths:
            selected_path = random.choice(available_paths)
            if len(selected_path) >= 2:
                super().traverse_path(selected_path, "hard")

    def _state_based_soft_traversal(self):
        """基于状态的软遍历"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return

        start_node = random.choice(available_nodes)
        available_paths = self.find_available_paths(start_node, "soft", 2)

        if available_paths:
            selected_path = random.choice(available_paths)
            if len(selected_path) >= 2:
                super().traverse_path(selected_path, "soft")

    def visualize_cognitive_states(self):
        """可视化认知状态变化"""
        if not self.energy_history:
            return

        iterations = [e['iteration'] for e in self.energy_history]
        energies = [e['energy'] for e in self.energy_history]
        network_energies = [e.get('network_energy', 0) for e in self.energy_history]
        states = [e['state'] for e in self.energy_history]

        # 创建颜色映射
        state_colors = {
            CognitiveState.FOCUSED: 'green',
            CognitiveState.EXPLORATORY: 'blue',
            CognitiveState.FATIGUED: 'red',
            CognitiveState.INSPIRED: 'purple'
        }

        colors = [state_colors[state] for state in states]

        plt.figure(figsize=(12, 8))

        # 主观能耗曲线
        plt.subplot(2, 1, 1)
        plt.scatter(iterations, energies, c=colors, alpha=0.6)
        plt.plot(iterations, energies, 'gray', alpha=0.3)
        plt.ylabel('主观认知能耗')
        plt.title('主观认知状态与能耗变化')

        # 添加状态图例
        for state, color in state_colors.items():
            plt.plot([], [], 'o', color=color, label=state.value)
        plt.legend()

        # 网络能耗曲线
        plt.subplot(2, 1, 2)
        plt.plot(iterations, network_energies, 'b-', alpha=0.7)
        plt.xlabel('迭代次数')
        plt.ylabel('网络平均能耗')
        plt.title('认知网络能耗演化')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印状态统计
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1

        print("\n=== 认知状态统计 ===")
        for state, count in state_counts.items():
            percentage = (count / len(states)) * 100
            print(f"{state.value}: {count}次 ({percentage:.1f}%)")


# 扩展的个体参数
def create_enhanced_individual_params(base_params):
    """创建增强的个体参数（包含认知状态相关参数）"""
    enhanced_params = base_params.copy()

    # 添加认知状态相关参数
    enhanced_params.update({
        'energy_variation': random.uniform(0.05, 0.15),
        'focus_bias': random.uniform(-0.1, 0.1),
        'exploration_bias': random.uniform(-0.1, 0.1),
        'fatigue_resistance': random.uniform(0.1, 0.3),
        'inspiration_frequency': random.uniform(0.05, 0.2)
    })

    return enhanced_params


# 运行增强实验
def run_enhanced_population_experiment(num_individuals=5, max_iterations=2000):
    """运行增强的群体实验"""
    base_parameters = {
        'forgetting_rate': 0.002,
        'base_learning_rate': 0.85,
        'hard_traversal_bias': 0.0,
        'soft_traversal_bias': 0.0,
        'compression_bias': 0.0,
        'migration_bias': 0.0,
        'learning_rate_variation': 0.1
    }

    variation_ranges = {
        'forgetting_rate': 0.2,
        'base_learning_rate': 0.1,
        'hard_traversal_bias': (-0.1, 0.1),
        'soft_traversal_bias': (-0.1, 0.1),
        'compression_bias': (-0.03, 0.03),
        'migration_bias': (-0.05, 0.05),
        'learning_rate_variation': (0.05, 0.15)
    }

    variation_simulator = IndividualVariation(base_parameters, variation_ranges)
    population_results = []

    print(f"=== 开始增强群体实验：{num_individuals}个个体 ===")

    for i in range(num_individuals):
        individual_id = f"个体_{i + 1}"
        print(f"\n--- 模拟 {individual_id} ---")

        # 生成基础参数
        base_individual_params = variation_simulator.generate_individual(individual_id)
        # 增强参数
        individual_params = create_enhanced_individual_params(base_individual_params)

        # 创建个体认知网络
        individual_graph = AdjustedSubjectiveCognitiveGraph(individual_params)

        # 初始化网络（使用更多节点）
        extended_nodes = create_extended_network_nodes()
        extended_edges = create_extended_network_edges(extended_nodes)
        individual_graph.initialize_graph(extended_nodes, extended_edges)

        # 运行模拟
        initial_energy = individual_graph.get_average_energy()
        individual_graph.monte_carlo_iteration(max_iterations=max_iterations)

        # 收集结果
        final_stats = individual_graph.get_network_stats()
        improvement = ((initial_energy - final_stats['avg_energy']) / initial_energy * 100)

        result = {
            'individual_id': individual_id,
            'parameters': individual_params,
            'initial_energy': initial_energy,
            'final_energy': final_stats['avg_energy'],
            'improvement': improvement,
            'compression_centers': final_stats['compression_centers'],
            'migration_bridges': final_stats['migration_bridges'],
            'concept_centers': list(individual_graph.concept_centers.keys()),
            'cognitive_states': individual_graph.energy_history
        }

        population_results.append(result)

        print(f"{individual_id} 结果:")
        print(f"  能耗降低: {improvement:.1f}%")
        print(f"  压缩中心: {result['compression_centers']}个")
        print(f"  迁移桥梁: {result['migration_bridges']}个")

        # 可视化认知状态
        individual_graph.visualize_cognitive_states()

    return population_results


def create_extended_network_nodes():
    """创建扩展的网络节点"""
    # 基础节点
    physics_nodes = ["牛顿定律", "运动学", "力学", "能量守恒", "动量", "热力学", "电磁学", "光学"]
    math_nodes = ["微积分", "线性代数", "概率论", "优化理论", "数论", "几何学", "拓扑学", "统计学"]
    cs_nodes = ["算法", "数据结构", "机器学习", "神经网络", "计算机视觉", "自然语言处理", "数据库", "操作系统"]
    principle_nodes = ["优化", "变换", "迭代", "抽象", "模式识别", "对称", "递归", "归纳"]

    return physics_nodes + math_nodes + cs_nodes + principle_nodes


def create_extended_network_edges(nodes):
    """创建扩展的网络边"""
    np.random.seed(42)
    initial_edges = []

    # 简单的领域分类函数
    def get_domain(node):
        physics_keywords = ["定律", "运动", "力学", "能量", "动量", "热", "电磁", "光学"]
        math_keywords = ["积分", "代数", "概率", "优化", "数论", "几何", "拓扑", "统计"]
        cs_keywords = ["算法", "数据", "学习", "网络", "视觉", "语言", "数据库", "系统"]

        for keyword in physics_keywords:
            if keyword in node:
                return "physics"
        for keyword in math_keywords:
            if keyword in node:
                return "math"
        for keyword in cs_keywords:
            if keyword in node:
                return "cs"
        return "principle"

    # 创建连接
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            domain_u = get_domain(u)
            domain_v = get_domain(v)

            # 基于领域关系设置初始能耗
            if domain_u == domain_v:
                # 同一领域内：低能耗
                weight = round(np.random.uniform(0.5, 1.2), 2)
            elif domain_u == "principle" or domain_v == "principle":
                # 与原理节点的连接：中等能耗
                weight = round(np.random.uniform(0.7, 1.5), 2)
            else:
                # 跨领域：高能耗
                weight = round(np.random.uniform(1.3, 2.2), 2)

            initial_edges.append((u, v, weight))

    return initial_edges


class AdjustedSubjectiveCognitiveGraph(SubjectiveCognitiveGraph):
    """调整参数后的主观认知图"""

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        super().__init__(individual_params, network_seed)

        # 调整状态能耗范围
        self.state_energy_ranges = {
            CognitiveState.FOCUSED: (1.5, 2.5),
            CognitiveState.EXPLORATORY: (1.0, 1.8),
            CognitiveState.FATIGUED: (0.8, 1.2),
            CognitiveState.INSPIRED: (2.0, 3.0)
        }

        # 调整状态转移概率
        self.state_transition_matrix = {
            CognitiveState.FOCUSED: {
                CognitiveState.EXPLORATORY: 0.3,
                CognitiveState.FATIGUED: 0.1,
                CognitiveState.INSPIRED: 0.2,
                CognitiveState.FOCUSED: 0.4
            },
            CognitiveState.EXPLORATORY: {
                CognitiveState.FOCUSED: 0.3,
                CognitiveState.FATIGUED: 0.2,
                CognitiveState.INSPIRED: 0.2,
                CognitiveState.EXPLORATORY: 0.3
            },
            CognitiveState.FATIGUED: {
                CognitiveState.FOCUSED: 0.3,
                CognitiveState.EXPLORATORY: 0.4,
                CognitiveState.INSPIRED: 0.1,
                CognitiveState.FATIGUED: 0.2
            },
            CognitiveState.INSPIRED: {
                CognitiveState.FOCUSED: 0.4,
                CognitiveState.EXPLORATORY: 0.3,
                CognitiveState.FATIGUED: 0.1,
                CognitiveState.INSPIRED: 0.2
            }
        }

        # 调整能耗分配
        self.hard_traversal_energy_ratio = 0.6
        self.soft_traversal_energy_ratio = 0.4

        # 重新初始化主观能耗
        self._update_subjective_energy()

    def can_traverse_edge(self, edge_energy, traversal_type):
        """调整能耗要求计算"""
        if traversal_type == "hard":
            required_energy = edge_energy * 0.8
            available_energy = self.subjective_energy * self.hard_traversal_energy_ratio
        else:
            required_energy = edge_energy * 0.6
            available_energy = self.subjective_energy * self.soft_traversal_energy_ratio

        return available_energy >= required_energy, available_energy - required_energy

    def traverse_path(self, path, traversal_type="hard"):
        """添加强制学习机制的遍历函数"""
        # 更新认知状态
        if random.random() < 0.1:
            self.update_cognitive_state()

        # 检查路径能耗
        total_required_energy = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                edge_energy = self.G[u][v]['weight']
                total_required_energy += edge_energy

        # 决定是否遍历
        can_traverse, energy_balance = self.can_traverse_edge(total_required_energy, traversal_type)

        # 强制学习机制
        if not can_traverse and random.random() < 0.2 and self.current_state != CognitiveState.FATIGUED:
            can_traverse = True
            energy_balance = -0.5

        if not can_traverse:
            if random.random() < 0.3:
                self.current_state = CognitiveState.FATIGUED
                self._update_subjective_energy()
            return

        # 执行遍历
        super().traverse_path(path, traversal_type)
        self._post_traversal_state_update(traversal_type, energy_balance)

    def _select_operation_based_on_state(self):
        """调整状态基于的操作选择"""
        state_operations = {
            CognitiveState.FOCUSED: {
                "hard_traversal": 0.5,
                "soft_traversal": 0.3,
                "compression": 0.1,
                "migration": 0.1
            },
            CognitiveState.EXPLORATORY: {
                "hard_traversal": 0.3,
                "soft_traversal": 0.4,
                "compression": 0.1,
                "migration": 0.2
            },
            CognitiveState.FATIGUED: {
                "hard_traversal": 0.2,
                "soft_traversal": 0.4,
                "compression": 0.2,
                "migration": 0.2
            },
            CognitiveState.INSPIRED: {
                "hard_traversal": 0.3,
                "soft_traversal": 0.4,
                "compression": 0.1,
                "migration": 0.2
            }
        }

        probs = state_operations[self.current_state]
        rand_val = random.random()
        cumulative = 0

        for op, prob in probs.items():
            cumulative += prob
            if rand_val <= cumulative:
                return op

        return "hard_traversal"
# 运行示例
if __name__ == "__main__":
    # 运行小规模增强实验

    enhanced_results = run_enhanced_population_experiment(
        num_individuals=5,
        max_iterations=10000
    )

# if __name__ == "__main__":
#     # 运行小规模群体测试
#     population_results = run_population_experiment(
#         num_individuals=5,
#         max_iterations=10000
#     )
#
#     # 可选：可视化某个个体的结果
#     if population_results:
#         # 重新创建最优个体进行可视化
#         best_individual = max(population_results, key=lambda x: x['improvement'])
#         print(f"\n=== 可视化最优个体: {best_individual['individual_id']} ===")
#
#         best_graph = create_individual_network(
#             best_individual['individual_id'],
#             best_individual['parameters']
#         )
#         best_graph.monte_carlo_iteration(max_iterations=10000)
#         best_graph.visualize_graph(f"最优个体: {best_individual['individual_id']}")
#         best_graph.visualize_energy_convergence()