"""
增强认知图模型模块
包含基于语义的认知图和能耗优化认知图。
"""

from core.cognitive_graph import BaseCognitiveGraph
from core.semantic_network import EnhancedSemanticConceptNetwork
from typing import Dict, Any, List


class SemanticEnhancedCognitiveGraph(BaseCognitiveGraph):
    """语义增强认知图 - 精简版本

    基于语义相似度初始化网络，并提供基于语义的概念压缩功能。
    """

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42, num_concepts: int = None):
        """初始化语义增强认知图。

        Args:
            individual_params (Dict[str, Any]): 个体参数。
            network_seed (int): 随机种子。
            num_concepts (int, optional): 概念节点数量。如果为None，使用默认数量。
        """
        super().__init__(individual_params, network_seed)
        self.semantic_network = EnhancedSemanticConceptNetwork(num_concepts=num_concepts)
        self.semantic_network.build_comprehensive_network()

    def calculate_semantic_similarity(self, node1: str, node2: str) -> float:
        """计算两个节点之间的语义相似度。

        根据论文公式（3）中的 sim(v_i, v_j) 计算，用于能耗函数。

        Args:
            node1 (str): 第一个节点名称。
            node2 (str): 第二个节点名称。

        Returns:
            float: 语义相似度，范围[0,1]。
        """
        return self.semantic_network.calculate_enhanced_similarity(node1, node2, "adaptive")

    def initialize_semantic_graph(self):
        """基于语义初始化认知图。

        根据语义相似度创建边，边权重初始化为 2.0 - 相似度*1.5，
        模拟论文中能耗函数 E_ij = α(1-sim) + ... 的初始状态。
        """
        nodes = list(self.semantic_network.concept_definitions.keys())
        self.G.add_nodes_from(nodes)

        edge_count = 0
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1:], i + 1):
                similarity = self.calculate_semantic_similarity(node1, node2)
                if similarity > 0.1:
                    energy = 2.0 - similarity * 1.5
                    energy = max(0.3, min(2.0, energy))

                    self.G.add_edge(node1, node2, weight=energy,
                                    original_weight=energy, traversal_count=0)
                    self.last_activation_time[(node1, node2)] = 0
                    edge_count += 1

        print(f"语义初始化: {len(nodes)}节点, {edge_count}条边")

    def conceptual_compression_based_on_semantics(self, compression_threshold: float = 0.3):
        """基于语义的概念压缩。

        根据论文中概念压缩的触发条件（公式4），寻找语义相似的节点集群并压缩。

        Args:
            compression_threshold (float): 相似度阈值，高于此值认为相似。

        Returns:
            list: 压缩后的组列表，每个元素为 (center, related_nodes)。
        """
        compressed_groups = []
        processed_nodes = set()

        for node in self.G.nodes():
            if node in processed_nodes:
                continue

            # 寻找语义相似的节点
            similar_nodes = [node]
            for other_node in self.G.nodes():
                if (other_node != node and other_node not in processed_nodes and
                        self.calculate_semantic_similarity(node, other_node) > compression_threshold):
                    similar_nodes.append(other_node)

            if len(similar_nodes) > 1:
                # 选择中心节点
                center = self._select_compression_center(similar_nodes)
                if center:
                    related_nodes = [n for n in similar_nodes if n != center]
                    compressed_groups.append((center, related_nodes))
                    processed_nodes.update(similar_nodes)

                    # 执行压缩
                    self.conceptual_compression(center, related_nodes, 0.4)
                    print(f"语义压缩: {center} <- {related_nodes}")

        return compressed_groups

    def _select_compression_center(self, nodes: List[str]) -> str:
        """选择压缩中心节点，即与其他节点平均相似度最高的节点。

        Args:
            nodes (List[str]): 候选节点列表。

        Returns:
            str: 中心节点名称。
        """
        best_center = None
        best_avg_similarity = 0

        for candidate in nodes:
            total_similarity = 0
            count = 0

            for other in nodes:
                if candidate != other:
                    similarity = self.calculate_semantic_similarity(candidate, other)
                    total_similarity += similarity
                    count += 1

            if count > 0:
                avg_similarity = total_similarity / count
                if avg_similarity > best_avg_similarity:
                    best_avg_similarity = avg_similarity
                    best_center = candidate

        return best_center


class EnergyOptimizedCognitiveGraph(SemanticEnhancedCognitiveGraph):
    """能耗优化认知图 - 精简版本

    在语义增强的基础上，增加能耗优化的遍历和智能概念压缩功能。
    """

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42, num_concepts: int = None):
        """初始化能耗优化认知图。

        Args:
            individual_params (Dict[str, Any]): 个体参数。
            network_seed (int): 随机种子。
            num_concepts (int, optional): 概念节点数量。
        """
        super().__init__(individual_params, network_seed, num_concepts)
        self.energy_optimization_threshold = 0.3

    def energy_efficient_traversal(self, start_node: str, target_node: str, max_depth: int = 3):
        """能耗优化的遍历（软遍历/硬遍历的混合实现）。

        根据论文中能量最小化原理，优先选择低能耗路径。

        Args:
            start_node (str): 起始节点。
            target_node (str): 目标节点。
            max_depth (int): 最大搜索深度。

        Returns:
            tuple: (路径列表, 总能耗)
        """
        def find_path(current, target, path, current_energy, visited, depth):
            if depth > max_depth or current == target:
                return path, current_energy

            best_path, best_energy = None, float('inf')
            neighbors = list(self.G.neighbors(current))

            # 按能耗排序，优先探索低能耗路径（硬遍历）
            neighbors.sort(key=lambda n: self.G[current][n]['weight'])

            for neighbor in neighbors[:4]:  # 考虑前4个低能耗邻居
                if neighbor not in visited:
                    edge_energy = self.G[current][neighbor]['weight']
                    new_energy = current_energy + edge_energy

                    if new_energy < best_energy * 1.5:  # 剪枝
                        visited.add(neighbor)
                        candidate_path, candidate_energy = find_path(
                            neighbor, target, path + [neighbor], new_energy, visited, depth + 1
                        )
                        visited.remove(neighbor)

                        if candidate_energy < best_energy:
                            best_energy = candidate_energy
                            best_path = candidate_path

            return best_path, best_energy

        visited = {start_node}
        return find_path(start_node, target_node, [start_node], 0, visited, 0)

    def smart_concept_compression(self, compression_threshold: float = 0.4):
        """智能概念压缩，基于预期能耗节省决定是否压缩。

        对应论文中概念压缩的触发条件：当压缩后能耗降低超过阈值时执行。

        Args:
            compression_threshold (float): 压缩阈值。

        Returns:
            list: 成功压缩的组列表，每个元素为 (center, nodes, expected_saving)。
        """
        compressed_groups = []

        # 找出高能耗集群
        high_energy_clusters = self._find_high_energy_clusters()

        for center, nodes in high_energy_clusters:
            if len(nodes) >= 2:
                expected_saving = self._calculate_compression_saving(center, nodes)

                if expected_saving > self.energy_optimization_threshold:
                    success = self.conceptual_compression(center, nodes, 0.5)
                    if success:
                        compressed_groups.append((center, nodes, expected_saving))
                        print(f"智能压缩: {center}, 预期节省: {expected_saving:.3f}")

        return compressed_groups

    def _find_high_energy_clusters(self):
        """找出高能耗集群，即中心节点与邻居边能耗较高的节点集合。

        Returns:
            list: 每个元素为 (center, neighbors) 的列表。
        """
        clusters = []
        processed = set()

        for node in self.G.nodes():
            if node in processed:
                continue

            high_energy_neighbors = []
            for neighbor in self.G.neighbors(node):
                energy = self.G[node][neighbor]['weight']
                if energy > 1.0:  # 高能耗阈值
                    high_energy_neighbors.append(neighbor)

            if len(high_energy_neighbors) >= 2:
                clusters.append((node, high_energy_neighbors))
                processed.update(high_energy_neighbors)
                processed.add(node)

        return clusters

    def _calculate_compression_saving(self, center: str, nodes: List[str]) -> float:
        """计算压缩预期节省的能耗百分比。

        根据公式：节省 = (当前总能耗 - 压缩后总能耗) / 当前总能耗。
        这里假设压缩后能耗降低40%作为近似。

        Args:
            center (str): 中心节点。
            nodes (List[str]): 相关节点列表。

        Returns:
            float: 预期节省比例。
        """
        current_total = sum(self.G[center][n]['weight'] for n in nodes if self.G.has_edge(center, n))
        compressed_total = current_total * 0.6  # 假设压缩后能耗降低40%
        return (current_total - compressed_total) / current_total if current_total > 0 else 0


if __name__ == "__main__":
    # 简单测试：创建一个小型认知图并运行基本功能
    print("测试 EnhancedCognitiveGraph 模块...")
    params = {
        'forgetting_rate': 0.002,
        'base_learning_rate': 0.85,
        'hard_traversal_bias': 0.0,
        'soft_traversal_bias': 0.0,
        'compression_bias': 0.0,
        'migration_bias': 0.0,
        'learning_rate_variation': 0.1
    }
    graph = EnergyOptimizedCognitiveGraph(params, num_concepts=51)
    graph.initialize_semantic_graph()
    print("初始化完成，节点数:", len(graph.G.nodes()))
    print("边数:", len(graph.G.edges()))
    print("测试通过。")