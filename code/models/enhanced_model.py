from core.cognitive_graph import BaseCognitiveGraph
from core.semantic_network import EnhancedSemanticConceptNetwork
from typing import Dict, Any
import random
import numpy as np

class SemanticEnhancedCognitiveGraph(BaseCognitiveGraph):
    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        super().__init__(individual_params, network_seed)
        self.semantic_network = EnhancedSemanticConceptNetwork()
        self.semantic_network.build_comprehensive_network()

    def calculate_semantic_similarity(self, node1, node2):
        """计算语义相似度"""
        return self.semantic_network.calculate_enhanced_similarity(node1, node2, "adaptive")

    def initialize_semantic_graph(self):
        """基于语义相似度初始化认知图"""
        nodes = list(self.semantic_network.concept_definitions.keys())
        self.G.add_nodes_from(nodes)

        initial_edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1:], i + 1):
                similarity = self.calculate_semantic_similarity(node1, node2)

                if similarity > 0.1:
                    energy = 2.0 - similarity * 1.5
                    energy = max(0.3, min(2.0, energy))

                    initial_edges.append((node1, node2, energy))

        for edge in initial_edges:
            u, v, weight = edge
            self.G.add_edge(u, v, weight=weight, traversal_count=0, original_weight=weight)
            self.last_activation_time[(u, v)] = 0

        print(f"基于语义初始化完成: {len(nodes)}个节点, {len(initial_edges)}条边")

    def conceptual_compression_based_on_semantics(self, compression_threshold=0.3):
        """基于语义相似度的概念压缩"""
        compressed_groups = []
        processed_nodes = set()

        all_nodes = list(self.G.nodes())

        for node in all_nodes:
            if node in processed_nodes:
                continue

            similar_nodes = [node]
            for other_node in all_nodes:
                if other_node != node and other_node not in processed_nodes:
                    similarity = self.calculate_semantic_similarity(node, other_node)
                    if similarity > compression_threshold:
                        similar_nodes.append(other_node)

            if len(similar_nodes) > 1:
                best_center = None
                best_avg_similarity = 0

                for candidate in similar_nodes:
                    total_similarity = 0
                    count = 0

                    for other in similar_nodes:
                        if candidate != other:
                            similarity = self.calculate_semantic_similarity(candidate, other)
                            total_similarity += similarity
                            count += 1

                    if count > 0:
                        avg_similarity = total_similarity / count
                        if avg_similarity > best_avg_similarity:
                            best_avg_similarity = avg_similarity
                            best_center = candidate

                if best_center:
                    related_nodes = [n for n in similar_nodes if n != best_center]
                    compressed_groups.append((best_center, related_nodes))
                    processed_nodes.update(similar_nodes)

        for center, related_nodes in compressed_groups:
            self.conceptual_compression(center, related_nodes, compression_strength=0.4)
            print(f"语义压缩: {center} <- {related_nodes}")

        return compressed_groups


class EnergyOptimizedCognitiveGraph(SemanticEnhancedCognitiveGraph):
    """能耗优化认知图"""

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        super().__init__(individual_params, network_seed)
        # 能耗优化特定参数
        self.energy_optimization_threshold = 0.3
        self.min_energy_threshold = 0.1
        self.max_energy_threshold = 2.0

    def energy_efficient_traversal(self, start_node, target_node, max_depth=3):
        """能耗优化的遍历算法"""

        def find_low_energy_path(current, target, path, current_energy, visited, depth):
            if depth > max_depth:
                return None, float('inf')

            if current == target:
                return path, current_energy

            best_path = None
            best_energy = float('inf')

            neighbors = list(self.G.neighbors(current))
            # 按能耗排序邻居
            neighbors.sort(key=lambda n: self.G[current][n]['weight'])

            for neighbor in neighbors[:5]:  # 只考虑前5个低能耗邻居
                if neighbor not in visited:
                    edge_energy = self.G[current][neighbor]['weight']
                    new_energy = current_energy + edge_energy

                    # 剪枝：如果当前路径能耗已经过高，提前终止
                    if new_energy > best_energy * 1.5:
                        continue

                    visited.add(neighbor)
                    candidate_path, candidate_energy = find_low_energy_path(
                        neighbor, target, path + [neighbor], new_energy, visited, depth + 1
                    )
                    visited.remove(neighbor)

                    if candidate_energy < best_energy:
                        best_energy = candidate_energy
                        best_path = candidate_path

            return best_path, best_energy

        visited = {start_node}
        path, total_energy = find_low_energy_path(start_node, target_node, [start_node], 0, visited, 0)

        return path, total_energy

    def adaptive_learning_rate(self, current_energy, similarity, traversal_type):
        """自适应学习率，基于当前能耗和相似度"""
        base_rate = self.base_learning_rate

        # 能耗越低，学习率越高（精力充沛时学习效果好）
        energy_factor = 1.5 - (current_energy / 2.0)

        # 相似度越高，学习率越高（关联性强的内容容易学）
        similarity_factor = 0.5 + similarity * 0.5

        # 遍历类型影响
        if traversal_type == "hard":
            traversal_factor = 0.8
        else:
            traversal_factor = 1.2

        adaptive_rate = base_rate * energy_factor * similarity_factor * traversal_factor
        return max(0.1, min(1.0, adaptive_rate))

    def smart_concept_compression(self, compression_threshold=0.4):
        """智能概念压缩，基于能耗优化"""
        compressed_groups = []

        # 找出高能耗的密集区域
        high_energy_clusters = self._find_high_energy_clusters()

        for cluster_center, cluster_nodes in high_energy_clusters:
            if len(cluster_nodes) >= 2:
                # 计算压缩后的预期能耗节省
                expected_saving = self._calculate_compression_saving(cluster_center, cluster_nodes)

                if expected_saving > self.energy_optimization_threshold:
                    success = self.conceptual_compression(cluster_center, cluster_nodes, compression_strength=0.5)
                    if success:
                        compressed_groups.append((cluster_center, cluster_nodes, expected_saving))
                        print(f"智能压缩: {cluster_center} <- {cluster_nodes}, 预期节省: {expected_saving:.3f}")

        return compressed_groups

    def improved_smart_concept_compression(self, compression_threshold=0.4, max_group_size=6):
        """改进的智能概念压缩"""
        compressed_groups = []
        high_energy_clusters = self._find_high_energy_clusters()

        for cluster_center, cluster_nodes in high_energy_clusters:
            # 限制压缩组大小
            if len(cluster_nodes) > max_group_size:
                cluster_nodes = cluster_nodes[:max_group_size]

            # 基于语义相似度进一步筛选
            filtered_nodes = []
            for node in cluster_nodes:
                similarity = self.calculate_semantic_similarity(cluster_center, node)
                if similarity > 0.2:  # 语义相似度阈值
                    filtered_nodes.append(node)

            if len(filtered_nodes) >= 2:  # 至少2个相关节点
                expected_saving = self._calculate_realistic_compression_saving(
                    cluster_center, filtered_nodes
                )

                if expected_saving > compression_threshold:
                    success = self.conceptual_compression(
                        cluster_center, filtered_nodes,
                        compression_strength=random.uniform(0.3, 0.7)  # 可变压缩强度
                    )
                    if success:
                        compressed_groups.append((cluster_center, filtered_nodes, expected_saving))

        return compressed_groups

    def _find_high_energy_clusters(self):
        """找出高能耗的节点集群"""
        clusters = []
        processed = set()

        for node in self.G.nodes():
            if node in processed:
                continue

            # 找出与当前节点相连的高能耗边
            high_energy_neighbors = []
            total_energy = 0

            for neighbor in self.G.neighbors(node):
                energy = self.G[node][neighbor]['weight']
                if energy > 1.0:  # 高能耗阈值
                    high_energy_neighbors.append(neighbor)
                    total_energy += energy

            if len(high_energy_neighbors) >= 2 and total_energy > 2.5:
                # 这是一个高能耗集群
                clusters.append((node, high_energy_neighbors))
                processed.update(high_energy_neighbors)
                processed.add(node)

        return clusters

    def _calculate_realistic_compression_saving(self, center, nodes):
        """更真实的能耗节省计算"""
        current_total_energy = 0
        compressed_total_energy = 0

        for node in nodes:
            if self.G.has_edge(center, node):
                current_energy = self.G[center][node]['weight']
                current_total_energy += current_energy

                # 基于相似度的可变压缩效果
                similarity = self.calculate_semantic_similarity(center, node)
                compression_factor = 0.3 + similarity * 0.4  # 相似度越高，压缩效果越好
                compressed_energy = current_energy * compression_factor
                compressed_total_energy += compressed_energy

        if current_total_energy > 0:
            saving = (current_total_energy - compressed_total_energy) / current_total_energy
            return min(saving, 0.8)  # 设置最大节省上限

        return 0.0

    def _calculate_compression_saving(self, center, nodes):
        """计算压缩节省（简化版本）"""
        # 简化的计算，假设每个连接的压缩节省为固定比例
        return len(nodes) * 0.15

    def evaluate_compression_quality(self, center, nodes):
        """评估压缩质量"""
        semantic_cohesion = 0
        for node in nodes:
            semantic_cohesion += self.calculate_semantic_similarity(center, node)
        semantic_cohesion /= len(nodes)

        # 基于语义凝聚度和节点数量评估质量
        size_penalty = max(0, (len(nodes) - 4) * 0.1)  # 节点过多惩罚
        quality_score = semantic_cohesion - size_penalty

        return max(0, quality_score)