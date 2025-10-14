import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from collections import defaultdict

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class CognitiveGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.traversal_history = []  # 记录遍历历史
        self.concept_centers = {}  # 记录概念压缩的中心节点

    def initialize_graph(self, nodes, initial_edges):
        """初始化认知图"""
        self.G.add_nodes_from(nodes)

        # 添加边并设置初始认知能耗（权重越大表示能耗越高）
        for edge in initial_edges:
            u, v, weight = edge
            self.G.add_edge(u, v, weight=weight, traversal_count=0, original_weight=weight)

    def traverse_path(self, path, traversal_type="hard"):
        """模拟遍历路径，降低相关边的能耗

        Args:
            path: 遍历的节点路径
            traversal_type: "hard"表示硬遍历（领域内），"soft"表示软遍历（跨领域探索）
        """
        self.traversal_history.append((path, traversal_type))

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                # 遍历次数增加
                self.G[u][v]['traversal_count'] += 1

                # 根据遍历类型调整学习速率
                if traversal_type == "hard":
                    learning_rate = 0.8  # 硬遍历学习更快
                else:
                    learning_rate = 0.9  # 软遍历学习较慢

                # 能耗随着遍历次数增加而降低（学习效应）
                current_weight = self.G[u][v]['weight']
                new_weight = max(0.1, current_weight * learning_rate)
                self.G[u][v]['weight'] = new_weight

    def conceptual_compression(self, center_node, related_nodes, compression_strength=0.5):
        """
        概念压缩：强化中心节点与相关节点的连接

        Args:
            center_node: 中心节点（概念压缩的焦点）
            related_nodes: 相关节点列表
            compression_strength: 压缩强度（0-1之间）
        """
        print(f"执行概念压缩: 中心节点 '{center_node}', 相关节点 {related_nodes}")

        # 1. 强化中心节点与每个相关节点的连接（大幅降低能耗）
        for node in related_nodes:
            if self.G.has_edge(center_node, node):
                current_energy = self.G[center_node][node]['weight']
                # 使用压缩强度调整能耗降低程度
                compressed_energy = max(0.05, current_energy * compression_strength)
                self.G[center_node][node]['weight'] = compressed_energy
                print(f"  强化连接 {center_node} - {node}: {current_energy:.2f} -> {compressed_energy:.2f}")

        # 2. 可选：适度弱化相关节点之间的直接连接，引导通过中心节点
        for i in range(len(related_nodes)):
            for j in range(i + 1, len(related_nodes)):
                if self.G.has_edge(related_nodes[i], related_nodes[j]):
                    current = self.G[related_nodes[i]][related_nodes[j]]['weight']
                    # 适度增加能耗，但不完全切断
                    new_energy = min(2.0, current * 1.2)
                    self.G[related_nodes[i]][related_nodes[j]]['weight'] = new_energy

        # 记录概念压缩关系
        self.concept_centers[center_node] = {
            'related_nodes': related_nodes,
            'compression_strength': compression_strength
        }

        return True

    def first_principles_migration(self, start_node, end_node, principle_nodes, exploration_bonus=0.1):
        """
        第一性原理迁移：通过基础原理节点发现跨领域低能耗路径

        Args:
            start_node: 起始节点
            end_node: 目标节点
            principle_nodes: 可能的基础原理节点列表
            exploration_bonus: 探索奖励，鼓励使用原理节点
        """
        print(f"寻找第一性原理迁移路径: {start_node} -> {end_node}")

        best_path = None
        best_energy = float('inf')
        found_via_principles = False

        # 检查直接连接
        if self.G.has_edge(start_node, end_node):
            direct_energy = self.G[start_node][end_node]['weight']
            best_path = [start_node, end_node]
            best_energy = direct_energy
            print(f"  直接路径能耗: {direct_energy:.2f}")

        # 尝试通过每个原理节点建立连接
        for principle in principle_nodes:
            if (self.G.has_edge(start_node, principle) and
                    self.G.has_edge(principle, end_node)):

                path_energy = (self.G[start_node][principle]['weight'] +
                               self.G[principle][end_node]['weight'])

                # 应用探索奖励：使用原理节点的路径获得额外奖励
                principle_bonus = exploration_bonus if principle in principle_nodes else 0
                adjusted_energy = path_energy - principle_bonus

                print(f"  通过 '{principle}' 的路径能耗: {path_energy:.2f} (调整后: {adjusted_energy:.2f})")

                if adjusted_energy < best_energy:
                    best_energy = adjusted_energy
                    best_path = [start_node, principle, end_node]
                    found_via_principles = True

        # 如果找到比直接连接更好的路径，则强化该路径
        if best_path and len(best_path) > 2 and found_via_principles:
            print(f"  🎯 发现优化路径: {best_path}, 能耗: {best_energy:.2f}")

            # 强化迁移路径上的连接
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                current = self.G[u][v]['weight']
                new_energy = max(0.1, current * 0.7)  # 显著强化
                self.G[u][v]['weight'] = new_energy
                print(f"    强化连接 {u} - {v}: {current:.2f} -> {new_energy:.2f}")

            # 记录迁移关系
            principle_node = best_path[1]
            if 'migration_bridges' not in self.G.nodes[principle_node]:
                self.G.nodes[principle_node]['migration_bridges'] = []

            self.G.nodes[principle_node]['migration_bridges'].append({
                'from': start_node,
                'to': end_node,
                'energy_saving': best_energy
            })

            # 模拟遍历这条新发现的优化路径
            self.traverse_path(best_path, traversal_type="soft")

            return best_path

        return None

    def auto_detect_compression_candidates(self, min_traversal=3, similarity_threshold=0.3):
        """
        自动检测可能的概念压缩候选

        Args:
            min_traversal: 最小共同遍历次数
            similarity_threshold: 连接相似度阈值
        """
        print("自动检测概念压缩候选...")
        compression_candidates = []

        # 分析遍历历史，找出频繁共同遍历的节点组
        co_traversal = defaultdict(int)

        for path, traversal_type in self.traversal_history:
            if traversal_type == "hard":  # 只关注硬遍历
                for i in range(len(path)):
                    for j in range(i + 1, len(path)):
                        node_pair = tuple(sorted([path[i], path[j]]))
                        co_traversal[node_pair] += 1

        # 找出频繁共同遍历的节点对
        frequent_pairs = [(pair, count) for pair, count in co_traversal.items()
                          if count >= min_traversal]

        # 基于频繁共同遍历构建候选组
        for (node1, node2), count in frequent_pairs:
            print(f"  频繁共同遍历: {node1} - {node2} ({count}次)")

            # 寻找与这两个节点都有较强连接的其他节点
            candidate_group = {node1, node2}
            for node in self.G.nodes():
                if node != node1 and node != node2:
                    edges_to_group = []
                    if self.G.has_edge(node, node1):
                        edges_to_group.append(self.G[node][node1]['weight'])
                    if self.G.has_edge(node, node2):
                        edges_to_group.append(self.G[node][node2]['weight'])

                    if (len(edges_to_group) == 2 and
                            max(edges_to_group) < similarity_threshold):
                        candidate_group.add(node)

            if len(candidate_group) >= 3:
                # 选择连接最紧密的节点作为中心节点
                center = self._find_center_node(candidate_group)
                related_nodes = [n for n in candidate_group if n != center]
                compression_candidates.append((center, related_nodes))

        return compression_candidates

    def _find_center_node(self, node_group):
        """在节点组中找到连接最紧密的中心节点"""
        best_center = None
        best_avg_energy = float('inf')

        for candidate in node_group:
            total_energy = 0
            connection_count = 0

            for other in node_group:
                if candidate != other and self.G.has_edge(candidate, other):
                    total_energy += self.G[candidate][other]['weight']
                    connection_count += 1

            if connection_count > 0:
                avg_energy = total_energy / connection_count
                if avg_energy < best_avg_energy:
                    best_avg_energy = avg_energy
                    best_center = candidate

        return best_center

    def calculate_path_energy(self, path):
        """计算路径的总认知能耗"""
        total_energy = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                total_energy += self.G[u][v]['weight']
        return total_energy

    def visualize_graph(self, title="认知图", highlight_nodes=None, highlight_edges=None, figsize=(12, 8)):
        """可视化认知图"""
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42)

        # 设置节点颜色
        node_colors = []
        for node in self.G.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('red')  # 高亮节点
            elif node in self.concept_centers:
                node_colors.append('orange')  # 概念压缩中心
            else:
                node_colors.append('lightblue')

        # 设置边颜色和宽度
        edge_colors = []
        edge_widths = []
        for u, v in self.G.edges():
            if highlight_edges and (u, v) in highlight_edges:
                edge_colors.append('red')
                edge_widths.append(3.0)
            else:
                edge_colors.append('gray')
                # 边宽度与能耗成反比（能耗越低，边越粗）
                energy = self.G[u][v]['weight']
                edge_widths.append(max(1, 3 - energy))

        # 绘制图形
        nx.draw_networkx_nodes(self.G, pos, node_size=1500,
                               node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(self.G, pos, width=edge_widths,
                               alpha=0.7, edge_color=edge_colors)
        nx.draw_networkx_labels(self.G, pos, font_size=10,
                                font_family='SimHei')

        # 添加边权重标签
        edge_labels = {(u, v): f"{self.G[u][v]['weight']:.2f}"
                       for u, v in self.G.edges()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels,
                                     font_size=8)

        plt.title(title, fontsize=16, fontfamily='SimHei')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # 打印网络统计信息
        self.print_network_stats()

    def print_network_stats(self):
        """打印网络统计信息"""
        print(f"网络统计: {len(self.G.nodes())}个节点, {len(self.G.edges())}条边")
        energies = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        avg_energy = np.mean(energies)
        print(f"平均边能耗: {avg_energy:.3f} (范围: {min(energies):.3f} - {max(energies):.3f})")

        if self.concept_centers:
            print(f"概念压缩中心: {list(self.concept_centers.keys())}")


# 使用示例和演示
def demo_cognitive_graph():
    """演示认知图的功能"""
    # 创建认知图实例
    cg = CognitiveGraph()

    # 定义节点 - 包含多个知识领域
    nodes = [
        # 物理学领域
        "牛顿定律", "运动学", "力学", "能量守恒", "动量",
        # 数学领域
        "微积分", "线性代数", "概率论", "优化理论",
        # 计算机领域
        "算法", "数据结构", "机器学习", "神经网络",
        # 基础原理节点（用于第一性原理迁移）
        "优化", "变换", "迭代", "抽象"
    ]

    # 初始化边（基于语义相关性设置初始能耗）
    # 同一领域内连接较强（能耗较低），跨领域连接较弱（能耗较高）
    np.random.seed(42)
    initial_edges = []

    # 领域分组
    physics_nodes = ["牛顿定律", "运动学", "力学", "能量守恒", "动量"]
    math_nodes = ["微积分", "线性代数", "概率论", "优化理论"]
    cs_nodes = ["算法", "数据结构", "机器学习", "神经网络"]
    principle_nodes = ["优化", "变换", "迭代", "抽象"]

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
                weight = round(np.random.uniform(0.3, 0.8), 2)
            elif (u in principle_nodes or v in principle_nodes):
                # 与原理节点的连接：中等能耗
                weight = round(np.random.uniform(0.5, 1.2), 2)
            else:
                # 跨领域：高能耗
                weight = round(np.random.uniform(1.0, 2.0), 2)

            initial_edges.append((u, v, weight))

    cg.initialize_graph(nodes, initial_edges)

    print("=== 初始认知图 ===")
    cg.visualize_graph("初始认知图")

    # 模拟学习过程 - 硬遍历（领域内学习）
    print("\n=== 模拟硬遍历学习过程 ===")
    hard_traversal_paths = [
        # 物理学领域学习
        ["牛顿定律", "运动学", "力学"],
        ["力学", "能量守恒", "动量"],
        ["牛顿定律", "力学", "动量"],
        # 数学领域学习
        ["微积分", "线性代数", "优化理论"],
        ["概率论", "优化理论"],
        # 计算机领域学习
        ["算法", "数据结构", "机器学习"],
        ["机器学习", "神经网络"]
    ]

    for path in hard_traversal_paths:
        cg.traverse_path(path, traversal_type="hard")
        path_energy = cg.calculate_path_energy(path)
        print(f"硬遍历路径 {path}，总能耗: {path_energy:.2f}")

    print("\n=== 学习后的认知图 ===")
    cg.visualize_graph("硬遍历学习后的认知图")

    # 自动检测并执行概念压缩
    print("\n=== 自动概念压缩 ===")
    candidates = cg.auto_detect_compression_candidates(min_traversal=2)
    for center, related_nodes in candidates:
        cg.conceptual_compression(center, related_nodes, compression_strength=0.4)

    if candidates:
        cg.visualize_graph("概念压缩后的认知图",
                           highlight_nodes=[c[0] for c in candidates])

    # 模拟第一性原理迁移 - 软遍历（跨领域创新）
    print("\n=== 模拟第一性原理迁移 ===")
    migration_attempts = [
        # 从物理学优化概念到计算机算法
        ("优化理论", "算法", ["优化", "迭代"]),
        # 从数学变换到机器学习
        ("变换", "机器学习", ["抽象", "迭代"]),
        # 从力学到数据结构
        ("力学", "数据结构", ["变换", "抽象"])
    ]

    discovered_paths = []
    for start, end, principles in migration_attempts:
        path = cg.first_principles_migration(start, end, principles)
        if path:
            discovered_paths.append(path)
            print(f"  发现创新路径: {' -> '.join(path)}")

    # 可视化最终结果
    print("\n=== 最终认知图 ===")
    highlight_edges = []
    for path in discovered_paths:
        for i in range(len(path) - 1):
            highlight_edges.append((path[i], path[i + 1]))

    cg.visualize_graph("最终认知图（包含概念压缩和第一性原理迁移）",
                       highlight_edges=highlight_edges)

    # 性能对比
    print("\n=== 性能对比 ===")
    if discovered_paths:
        test_path = discovered_paths[0]
        original_energy = sum(cg.G[u][v]['original_weight'] for u, v in zip(test_path[:-1], test_path[1:])
                              if cg.G.has_edge(u, v))
        current_energy = cg.calculate_path_energy(test_path)
        improvement = ((original_energy - current_energy) / original_energy * 100)
        print(f"路径 {test_path} 的改进:")
        print(f"  原始预估能耗: {original_energy:.2f}")
        print(f"  当前实际能耗: {current_energy:.2f}")
        print(f"  能耗降低: {improvement:.1f}%")


if __name__ == "__main__":
    demo_cognitive_graph()