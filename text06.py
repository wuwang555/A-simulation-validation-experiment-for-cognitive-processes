import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from collections import defaultdict
import random
import math
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedCognitiveGraph:
    def __init__(self, forgetting_rate=0.0005, base_learning_rate=0.85):
        self.G = nx.Graph()
        self.traversal_history = []
        self.concept_centers = {}
        self.iteration_count = 0
        self.energy_history = []

        # 改进的参数
        self.forgetting_rate = forgetting_rate  # 遗忘率
        self.base_learning_rate = base_learning_rate
        self.semantic_similarity_matrix = {}  # 语义相似度缓存
        self.last_activation_time = {}  # 记录边最后激活时间

        # 领域分组定义
        self.physics_nodes = ["牛顿定律", "运动学", "力学", "能量守恒", "动量"]
        self.math_nodes = ["微积分", "线性代数", "概率论", "优化理论"]
        self.cs_nodes = ["算法", "数据结构", "机器学习", "神经网络"]
        self.principle_nodes = ["优化", "变换", "迭代", "抽象", "模式识别"]

        self.all_nodes = self.physics_nodes + self.math_nodes + self.cs_nodes + self.principle_nodes

    import math

    def forgetting_function(self, current_time, last_activation_time, current_energy, similarity):
        """
        基于指数衰减的遗忘时间函数

        Args:
            current_time: 当前迭代次数
            last_activation_time: 最后激活时间
            current_energy: 当前边的能耗
            similarity: 节点间的语义相似度

        Returns:
            遗忘因子 (0-1之间的值，表示能耗增加的比例)
        """
        # 计算时间间隔（以迭代次数为单位）
        time_gap = current_time - last_activation_time

        # 基础遗忘率：时间间隔越长，遗忘越快
        base_forgetting = 1 - math.exp(-time_gap / 1000)  # 时间尺度参数

        # 当前能耗影响：高能耗边更容易被遗忘
        energy_factor = 0.5 + 0.5 * (current_energy / 2.0)  # 归一化到0.5-1.0

        # 相似度影响：相似概念遗忘更慢
        similarity_protection = 1 - (similarity * 0.7)  # 相似度提供保护

        # 综合遗忘因子
        forgetting_factor = (base_forgetting * energy_factor * similarity_protection *
                             self.forgetting_rate)

        return min(forgetting_factor, 0.3)  # 限制最大遗忘率

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

    def monte_carlo_iteration(self, max_iterations=2000,
                              hard_traversal_prob=0.5,
                              soft_traversal_prob=0.3,
                              compression_prob=0.1,
                              migration_prob=0.1):
        """
        蒙特卡洛模拟主循环
        """
        print(f"开始蒙特卡洛模拟，最大迭代次数: {max_iterations}")

        for iteration in range(max_iterations):
            self.iteration_count += 1

            # 应用遗忘机制
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

            # 概念压缩 - 只在后期触发
            cumulative_prob += compression_prob
            if rand_val <= cumulative_prob and self.iteration_count > max_iterations * 0.3:
                self._random_compression()
                continue

            # 第一性原理迁移 - 只在后期触发
            cumulative_prob += migration_prob
            if rand_val <= cumulative_prob and self.iteration_count > max_iterations * 0.4:
                self._random_migration()
                continue

            # 每100次迭代打印进度
            if iteration % 200 == 0:
                stats = self.get_network_stats()
                print(f"迭代 {iteration}, 平均能耗: {current_avg_energy:.3f}, "
                      f"压缩中心: {stats['compression_centers']}, 迁移桥梁: {stats['migration_bridges']}")

    def _apply_forgetting(self):
        """应用遗忘机制到所有边 - 使用改进的时间函数"""
        current_time = self.iteration_count

        for u, v in self.G.edges():
            time_since_last_activation = current_time - self.last_activation_time.get((u, v), 0)
            if time_since_last_activation > 0:
                current_energy = self.G[u][v]['weight']
                similarity = self.calculate_semantic_similarity(u, v)

                # 使用新的遗忘时间函数
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
        """随机概念压缩尝试 - 改进的触发条件"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 3:
            return

        # 只有5%的概率真正触发压缩
        if random.random() > 0.05:
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
        """随机第一性原理迁移尝试 - 改进的触发条件"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 4:
            return

        # 只有5%的概率真正触发迁移
        if random.random() > 0.05:
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
        plt.plot(self.energy_history, 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('迭代次数')
        plt.ylabel('平均认知能耗')
        plt.title('改进模型 - 认知能耗收敛过程')
        plt.grid(True, alpha=0.3)

        # 标记重要事件
        colors = ['red', 'green', 'orange', 'purple']
        for i, (center, info) in enumerate(self.concept_centers.items()):
            iteration = info['iteration']
            if iteration < len(self.energy_history):
                color = colors[i % len(colors)]
                plt.axvline(x=iteration, color=color, alpha=0.5, linestyle='--',
                            label=f'压缩: {center}' if i < 4 else "")

        if len(self.concept_centers) > 0:
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
        for node in self.G.nodes():
            if node in self.concept_centers:
                node_colors.append('red')  # 概念压缩中心
                node_sizes.append(2000)
            elif any('migration_bridges' in self.G.nodes[n] for n in self.G.nodes()):
                node_colors.append('orange')  # 迁移桥梁节点
                node_sizes.append(1500)
            else:
                node_colors.append('lightblue')
                node_sizes.append(800)

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
        nx.draw_networkx_labels(self.G, pos, font_size=8,
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
                weight = round(np.random.uniform(1.3, 2.2), 2)

            initial_edges.append((u, v, weight))

    return all_nodes, initial_edges


def run_improved_experiment():
    """运行改进后的实验"""
    # 创建改进的认知图实例
    cog_graph = ImprovedCognitiveGraph(
        forgetting_rate=0.0005,  # 较小的遗忘率
        base_learning_rate=0.85  # 基础学习速率
    )

    # 初始化网络
    nodes, edges = create_sample_network()
    cog_graph.initialize_graph(nodes, edges)

    print("=== 初始状态 ===")
    initial_stats = cog_graph.get_network_stats()
    print(f"初始平均能耗: {initial_stats['avg_energy']:.3f}")
    cog_graph.visualize_graph("初始认知图")

    # 运行蒙特卡洛模拟
    print("\n=== 开始改进的蒙特卡洛模拟 ===")
    cog_graph.monte_carlo_iteration(
        max_iterations=10000,
        hard_traversal_prob=0.5,  # 50%概率硬遍历
        soft_traversal_prob=0.3,  # 30%概率软遍历
        compression_prob=0.08,  # 8%概率概念压缩（但实际触发概率更低）
        migration_prob=0.12  # 12%概率第一性原理迁移（但实际触发概率更低）
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
    cog_graph.visualize_graph("改进模型后的认知图")
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


if __name__ == "__main__":
    run_improved_experiment()