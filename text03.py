import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class CognitiveGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.node_history = {}  # 记录节点的遍历历史

    def initialize_graph(self, nodes, initial_edges):
        """初始化认知图"""
        self.G.add_nodes_from(nodes)

        # 添加边并设置初始认知能耗（权重越大表示能耗越高）
        for edge in initial_edges:
            u, v, weight = edge
            self.G.add_edge(u, v, weight=weight, traversal_count=0)

        # 初始化节点历史
        for node in nodes:
            self.node_history[node] = []

    def traverse_path(self, path):
        """模拟遍历路径，降低相关边的能耗"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                # 遍历次数增加
                self.G[u][v]['traversal_count'] += 1
                # 能耗随着遍历次数增加而降低（学习效应）
                current_weight = self.G[u][v]['weight']
                new_weight = max(0.1, current_weight * 0.9)  # 最低能耗为0.1
                self.G[u][v]['weight'] = new_weight

                # 记录遍历历史
                self.node_history[u].append(v)
                self.node_history[v].append(u)

    def conceptual_compression(self, node_group, new_node_name):
        """概念压缩：将一组节点压缩为一个新节点"""
        if len(node_group) < 2:
            return False

        # 1. 计算新节点与外部节点的连接能耗（取最小值）
        external_connections = {}

        for node in node_group:
            for neighbor in self.G.neighbors(node):
                if neighbor not in node_group:  # 只考虑外部连接
                    weight = self.G[node][neighbor]['weight']
                    if neighbor not in external_connections:
                        external_connections[neighbor] = weight
                    else:
                        # 取最小值，代表最优路径
                        external_connections[neighbor] = min(
                            external_connections[neighbor], weight
                        )

        # 2. 添加新节点
        self.G.add_node(new_node_name)

        # 3. 添加新节点与外部节点的连接
        for external_node, min_weight in external_connections.items():
            self.G.add_edge(new_node_name, external_node,
                            weight=min_weight, traversal_count=0)

        # 4. 移除原节点（在实际应用中可能选择隐藏而不是移除）
        # self.G.remove_nodes_from(node_group)

        # 记录压缩历史
        self.node_history[new_node_name] = f"压缩自: {node_group}"

        return True

    def conceptual_compression_v2(self, center_node, related_nodes):
        """
        新版本的概念压缩：强化中心节点与相关节点的连接
        """
        # 强化中心节点与每个相关节点的连接
        for node in related_nodes:
            if self.G.has_edge(center_node, node):
                # 显著降低能耗，模拟压缩效果
                current_energy = self.G[center_node][node]['weight']
                compressed_energy = max(0.1, current_energy * 0.3)  # 大幅降低
                self.G[center_node][node]['weight'] = compressed_energy

                # 记录压缩关系
                if 'compressed_concepts' not in self.G.nodes[center_node]:
                    self.G.nodes[center_node]['compressed_concepts'] = []
                self.G.nodes[center_node]['compressed_concepts'].append(node)

        # 可选：弱化相关节点之间的直接连接，突出中心节点的枢纽地位
        for i in range(len(related_nodes)):
            for j in range(i + 1, len(related_nodes)):
                if self.G.has_edge(related_nodes[i], related_nodes[j]):
                    # 适度增加能耗，引导通过中心节点
                    current = self.G[related_nodes[i]][related_nodes[j]]['weight']
                    self.G[related_nodes[i]][related_nodes[j]]['weight'] = current * 1.5

    def first_principles_migration(self, start_domain, end_domain, principle_nodes):
        """
        第一性原理迁移：通过基础原理节点连接两个领域
        """
        best_path = None
        best_energy = float('inf')

        # 尝试通过每个原理节点建立连接
        for principle in principle_nodes:
            if (self.G.has_edge(start_domain, principle) and
                    self.G.has_edge(principle, end_domain)):

                path_energy = (self.G[start_domain][principle]['weight'] +
                               self.G[principle][end_domain]['weight'])

                if path_energy < best_energy:
                    best_energy = path_energy
                    best_path = [start_domain, principle, end_domain]

        # 如果找到比直接连接更好的路径，则强化该路径
        if best_path and self.G.has_edge(start_domain, end_domain):
            direct_energy = self.G[start_domain][end_domain]['weight']
            if best_energy < direct_energy:
                # 强化迁移路径
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i + 1]
                    current = self.G[u][v]['weight']
                    self.G[u][v]['weight'] = max(0.1, current * 0.7)

                # 记录迁移关系
                self.G.nodes[best_path[1]]['migration_bridge'] = {
                    'from': start_domain,
                    'to': end_domain,
                    'energy_saving': direct_energy - best_energy
                }

                return best_path

        return None
    def calculate_path_energy(self, path):
        """计算路径的总认知能耗"""
        total_energy = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                total_energy += self.G[u][v]['weight']
        return total_energy

    def visualize_graph(self, title="认知图", node_size=2000, figsize=(12, 8)):
        """可视化认知图"""
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42)

        # 绘制节点和边
        edge_weights = [self.G[u][v]['weight'] * 2 for u, v in self.G.edges()]
        node_colors = ['lightblue' for _ in self.G.nodes()]

        nx.draw_networkx_nodes(self.G, pos, node_size=node_size,
                               node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(self.G, pos, width=edge_weights,
                               alpha=0.7, edge_color='gray')
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
        print(f"网络统计: {len(self.G.nodes())}个节点, {len(self.G.edges())}条边")
        avg_energy = np.mean([self.G[u][v]['weight'] for u, v in self.G.edges()])
        print(f"平均边能耗: {avg_energy:.3f}")


# 使用示例
if __name__ == "__main__":
    # 创建认知图实例
    cg = CognitiveGraph()

    # 定义初始节点和边（全连接图）
    nodes = ["数学", "物理", "化学", "生物", "计算机"]

    # 初始化边（随机生成初始能耗）
    np.random.seed(42)
    initial_edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            weight = round(np.random.uniform(0.5, 2.0), 2)  # 初始能耗
            initial_edges.append((nodes[i], nodes[j], weight))

    cg.initialize_graph(nodes, initial_edges)

    print("=== 初始认知图 ===")
    cg.visualize_graph("初始认知图")

    # 模拟一些学习过程（路径遍历）
    learning_paths = [
        ["数学", "物理", "计算机"],
        ["物理", "数学", "计算机"],
        ["化学", "生物", "物理"],
        ["数学", "计算机"]
    ]

    for path in learning_paths:
        cg.traverse_path(path)
        path_energy = cg.calculate_path_energy(path)
        print(f"遍历路径 {path}，总能耗: {path_energy:.2f}")

    print("\n=== 学习后的认知图 ===")
    cg.visualize_graph("学习后的认知图")

    # 进行概念压缩：将数学和物理压缩为"数理基础"
    compression_success = cg.conceptual_compression(
        ["数学", "物理"], "数理基础"
    )

    if compression_success:
        print("\n=== 概念压缩后的认知图 ===")
        cg.visualize_graph("概念压缩后的认知图")

        # 测试压缩后的路径能耗
        new_path = ["数理基础", "计算机"]
        new_energy = cg.calculate_path_energy(new_path)
        print(f"压缩后路径 {new_path} 的能耗: {new_energy:.2f}")

        # 对比原始路径能耗
        old_path = ["数学", "物理", "计算机"]
        old_energy = cg.calculate_path_energy(old_path)
        print(f"原始路径 {old_path} 的能耗: {old_energy:.2f}")
        print(f"能耗降低: {((old_energy - new_energy) / old_energy * 100):.1f}%")