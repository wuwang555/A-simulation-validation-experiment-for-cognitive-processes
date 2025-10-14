import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib as mpl
import random
# ====================== 解决中文乱码问题 ======================
# 配置中文字体
# 设置中文字体 - 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class CognitiveGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.concept_vectors = {}  # 存储概念的向量表示
        self.energy_costs = {}  # 存储边的能耗
        self.traversal_history = []  # 记录遍历历史
        self.compressed_concepts = {}  # 存储压缩后的概念

    def add_concept(self, concept, vector):
        """添加概念节点"""
        self.graph.add_node(concept)
        self.concept_vectors[concept] = np.array(vector)

    def calculate_similarity(self, concept1, concept2):
        """计算两个概念的相似度（余弦相似度）"""
        vec1 = self.concept_vectors[concept1].reshape(1, -1)
        vec2 = self.concept_vectors[concept2].reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    def traverse(self, path):
        """遍历概念路径，更新能耗"""
        self.traversal_history.extend(path)

        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]

            # 如果边不存在，创建它
            if not self.graph.has_edge(node1, node2):
                similarity = self.calculate_similarity(node1, node2)
                # 相似度越高，能耗越低
                energy_cost = 1 - similarity
                self.graph.add_edge(node1, node2, weight=energy_cost)
                self.energy_costs[(node1, node2)] = energy_cost
            else:
                # 已存在的边，随着遍历降低能耗（学习效应）
                current_cost = self.graph[node1][node2]['weight']
                new_cost = current_cost * 0.95  # 每次遍历降低5%能耗
                self.graph[node1][node2]['weight'] = new_cost
                self.energy_costs[(node1, node2)] = new_cost

    def find_compressible_groups(self, similarity_threshold=0.7, co_occurrence_threshold=3):
        """找出可以压缩的概念组"""
        from collections import defaultdict

        # 统计概念共现
        co_occurrence = defaultdict(int)
        for i in range(len(self.traversal_history) - 1):
            pair = tuple(sorted([self.traversal_history[i], self.traversal_history[i + 1]]))
            co_occurrence[pair] += 1

        # 找出频繁共现且相似的概念组
        compressible_groups = []
        concepts = list(self.graph.nodes())

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                concept1, concept2 = concepts[i], concepts[j]

                # 检查相似度和共现频率
                similarity = self.calculate_similarity(concept1, concept2)
                co_occur_count = co_occurrence.get(tuple(sorted([concept1, concept2])), 0)

                if similarity > similarity_threshold and co_occur_count > co_occurrence_threshold:
                    compressible_groups.append([concept1, concept2])

        return compressible_groups

    def compress_concepts(self, concept_group):
        """压缩一组概念为一个新概念"""
        if len(concept_group) < 2:
            return

        # 创建新概念名称
        new_concept = f"Compressed_{'_'.join(concept_group)}"

        # 新概念的向量是原概念向量的平均
        vectors = [self.concept_vectors[concept] for concept in concept_group]
        new_vector = np.mean(vectors, axis=0)

        # 添加新概念
        self.add_concept(new_concept, new_vector)
        self.compressed_concepts[new_concept] = concept_group

        # 移除原概念
        for concept in concept_group:
            self.graph.remove_node(concept)
            del self.concept_vectors[concept]

        print(f"概念压缩完成: {concept_group} -> {new_concept}")

    def visualize(self, title="Cognitive Graph"):
        """可视化当前的概念网络"""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)

        # 绘制节点和边
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue',
                               node_size=500, alpha=0.9)
        nx.draw_networkx_labels(self.graph, pos)

        # 绘制边，宽度反映能耗（能耗越低，边越粗）
        edges = self.graph.edges()
        weights = [self.graph[u][v]['weight'] for u, v in edges]

        nx.draw_networkx_edges(self.graph, pos, width=[max(1, 5 * (1 - w)) for w in weights],
                               alpha=0.7, edge_color='gray')

        # 绘制边的权重标签
        edge_labels = {(u, v): f'{self.graph[u][v]["weight"]:.2f}'
                       for u, v in self.graph.edges()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def get_network_stats(self):
        """获取网络统计信息"""
        total_energy = sum(self.energy_costs.values())
        avg_energy = total_energy / len(self.energy_costs) if self.energy_costs else 0

        return {
            "节点数量": len(self.graph.nodes()),
            "边数量": len(self.graph.edges()),
            "总能耗": total_energy,
            "平均能耗": avg_energy,
            "压缩概念数": len(self.compressed_concepts)
        }


# ===== 演示如何使用 =====
if __name__ == "__main__":
    # 创建认知图
    cg = CognitiveGraph()

    # 添加一些初始概念（用随机向量模拟）
    # 现实中这些向量应该来自词嵌入模型如Word2Vec
    concepts = {
        "牛顿第一定律": [0.9, 0.8, 0.1, 0.2],
        "惯性定律": [0.85, 0.75, 0.15, 0.25],  # 与牛顿第一定律高度相似
        "F=ma": [0.7, 0.6, 0.8, 0.3],
        "力学": [0.6, 0.5, 0.7, 0.4],
        "运动": [0.3, 0.4, 0.2, 0.9],
        "静止": [0.2, 0.3, 0.1, 0.85],  # 与运动相关但不同
    }

    for concept, vector in concepts.items():
        cg.add_concept(concept, vector)

    print("=== 初始状态 ===")
    cg.visualize("初始认知网络")
    print(cg.get_network_stats())

    # 模拟一些遍历路径（学习过程）
    print("\n=== 模拟学习过程 ===")
    traversals = [
        ["牛顿第一定律", "惯性定律", "力学"],
        ["牛顿第一定律", "惯性定律", "F=ma"],
        ["运动", "静止", "力学"],
        ["牛顿第一定律", "惯性定律", "F=ma", "力学"],
        ["牛顿第一定律", "惯性定律"]  # 多次强化这个路径
    ]

    for i, path in enumerate(traversals):
        print(f"遍历 {i + 1}: {path}")
        cg.traverse(path)

    print("\n=== 学习后的网络 ===")
    cg.visualize("学习后的认知网络")
    print(cg.get_network_stats())

    # 尝试概念压缩
    print("\n=== 尝试概念压缩 ===")
    compressible_groups = cg.find_compressible_groups()
    print(f"发现可压缩的概念组: {compressible_groups}")

    if compressible_groups:
        for group in compressible_groups:
            cg.compress_concepts(group)

        print("\n=== 压缩后的网络 ===")
        cg.visualize("概念压缩后的认知网络")
        print(cg.get_network_stats())
        print(f"压缩映射: {cg.compressed_concepts}")