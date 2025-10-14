import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class CognitiveGraph:
    def __init__(self, nodes):
        self.G = nx.Graph()
        self.node_history = {}  # 记录节点遍历历史
        self.compression_history = []  # 记录压缩历史

        # 初始化节点和全连接
        for node in nodes:
            self.G.add_node(node, energy_cost=1.0)  # 初始能耗为1
            self.node_history[node] = 0

        # 创建全连接，初始能耗基于随机相似度
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j:
                    # 相似度越高，能耗越低
                    similarity = np.random.uniform(0.1, 0.9)
                    energy_cost = 1 - similarity
                    self.G.add_edge(node1, node2, energy_cost=energy_cost,
                                    similarity=similarity, traversal_count=0)

    def traverse_path(self, path):
        """模拟遍历路径，降低相关边的能耗"""
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            if self.G.has_edge(node1, node2):
                current_cost = self.G[node1][node2]['energy_cost']
                # 遍历次数增加，能耗降低（学习效应）
                self.G[node1][node2]['traversal_count'] += 1
                new_cost = current_cost * 0.9  # 每次遍历降低10%能耗
                self.G[node1][node2]['energy_cost'] = max(new_cost, 0.1)

                # 更新节点遍历历史
                self.node_history[node1] += 1
                self.node_history[node2] += 1

    def should_compress(self, nodes, threshold=0.8):
        """判断一组节点是否应该被压缩"""
        if len(nodes) < 2:
            return False

        # 计算内部连接的平均相似度
        total_similarity = 0
        count = 0
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j and self.G.has_edge(node1, node2):
                    total_similarity += self.G[node1][node2]['similarity']
                    count += 1

        if count == 0:
            return False

        avg_similarity = total_similarity / count
        return avg_similarity > threshold

    def compress_concepts(self, nodes, new_node_name):
        """执行概念压缩"""
        if not self.should_compress(nodes):
            return False

        # 1. 创建新节点
        self.G.add_node(new_node_name, energy_cost=0.5)  # 压缩后节点基础能耗更低
        self.node_history[new_node_name] = 0

        # 2. 计算新节点与外部节点的连接能耗（回答您的问题！）
        external_nodes = set(self.G.nodes()) - set(nodes) - {new_node_name}

        for ext_node in external_nodes:
            # 计算原节点群与此外部节点的平均能耗
            total_energy = 0
            count = 0
            for node in nodes:
                if self.G.has_edge(node, ext_node):
                    total_energy += self.G[node][ext_node]['energy_cost']
                    count += 1

            if count > 0:
                avg_energy = total_energy / count
                # 压缩后的连接能耗更低（因为经过了抽象）
                compressed_energy = avg_energy * 0.7
                similarity = 1 - compressed_energy

                self.G.add_edge(new_node_name, ext_node,
                                energy_cost=compressed_energy,
                                similarity=similarity,
                                traversal_count=0)

        # 3. 移除原节点（这会自动移除相关的边）
        for node in nodes:
            self.G.remove_node(node)

        # 记录压缩历史
        self.compression_history.append({
            'compressed_nodes': nodes,
            'new_node': new_node_name,
            'step': len(self.compression_history) + 1
        })

        return True

    def visualize(self, title="认知图"):
        """可视化当前的认知图"""
        plt.figure(figsize=(12, 8))

        # 节点颜色基于能耗
        node_colors = [self.G.nodes[node]['energy_cost'] for node in self.G.nodes()]

        # 边颜色和宽度基于能耗
        edge_colors = [self.G[u][v]['energy_cost'] for u, v in self.G.edges()]
        edge_widths = [2 / (self.G[u][v]['energy_cost'] + 0.1) for u, v in self.G.edges()]

        pos = nx.spring_layout(self.G, seed=42)

        # 绘制边
        edges = nx.draw_networkx_edges(self.G, pos,
                                       edge_color=edge_colors,
                                       edge_cmap=plt.cm.Reds,
                                       width=edge_widths,
                                       alpha=0.6)

        # 绘制节点
        nodes = nx.draw_networkx_nodes(self.G, pos,
                                       node_color=node_colors,
                                       node_size=800,
                                       cmap=plt.cm.Blues,
                                       alpha=0.7)

        # 添加标签
        nx.draw_networkx_labels(self.G, pos, font_size=10)

        # 添加颜色条
        plt.colorbar(nodes, label='节点能耗')
        if edges:
            plt.colorbar(edges, label='边能耗')

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def get_global_energy(self):
        """计算全局总能耗"""
        total_energy = 0
        for node in self.G.nodes():
            total_energy += self.G.nodes[node]['energy_cost']
        for u, v in self.G.edges():
            total_energy += self.G[u][v]['energy_cost']
        return total_energy


# 演示代码
if __name__ == "__main__":
    # 初始化认知图
    initial_nodes = ['数学', '物理', '化学', '生物', '计算机', '心理学', '经济学']
    cog_graph = CognitiveGraph(initial_nodes)

    print("初始状态:")
    print(f"节点数: {cog_graph.G.number_of_nodes()}")
    print(f"边数: {cog_graph.G.number_of_edges()}")
    print(f"全局能耗: {cog_graph.get_global_energy():.2f}")
    cog_graph.visualize("初始认知图")

    # 模拟一些学习路径
    science_path = ['数学', '物理', '化学', '生物']
    tech_path = ['数学', '计算机', '经济学']

    for _ in range(5):
        cog_graph.traverse_path(science_path)
        cog_graph.traverse_path(tech_path)

    print("\n经过学习后:")
    cog_graph.visualize("学习后的认知图")

    # 尝试概念压缩
    science_nodes = ['数学', '物理', '化学', '生物']
    if cog_graph.compress_concepts(science_nodes, '自然科学'):
        print("\n压缩成功!")
        print(f"新节点数: {cog_graph.G.number_of_nodes()}")
        print(f"新边数: {cog_graph.G.number_of_edges()}")
        print(f"压缩后全局能耗: {cog_graph.get_global_energy():.2f}")
        cog_graph.visualize("概念压缩后的认知图")

        # 显示压缩历史
        print("\n压缩历史:")
        for record in cog_graph.compression_history:
            print(f"步骤{record['step']}: {record['compressed_nodes']} -> {record['new_node']}")
    else:
        print("\n压缩条件未满足")