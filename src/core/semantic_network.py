"""
语义网络模块
-------------
定义语义概念网络及相关类，基于关键词和元结构计算概念间的相似度，
并支持构建综合网络、跨领域路径搜索等功能。
"""

import time
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any

import jieba
import numpy as np

from config import (
    CORE_CONCEPT_DEFINITIONS, CONCEPT_DOMAINS,
    META_STRUCTURE_MAP, NETWORK_CONFIG, KEYWORD_CONFIG
)


class SemanticConceptNetwork:
    """基于定义关键词的语义概念网络。

    该类维护概念定义、关键词列表，并通过关键词Jaccard相似度及领域相似度计算概念间的语义相似度。
    支持构建综合网络、递归扩展及跨领域路径搜索。
    """

    def __init__(self):
        self.concept_definitions: Dict[str, Dict] = {}      # 概念 -> {定义,来源,时间戳}
        self.concept_keywords: Dict[str, List[str]] = {}    # 概念 -> 关键词列表
        self.semantic_network: Dict[str, Dict[str, float]] = defaultdict(dict)  # 概念->邻居概念->相似度

    def add_concept_definition(self, concept: str, definition: str, source: str = "manual") -> None:
        """添加一个概念的定义，并自动提取关键词。

        :param concept: 概念名称
        :param definition: 定义文本
        :param source: 来源标识
        """
        self.concept_definitions[concept] = {
            'definition': definition,
            'source': source,
            'timestamp': time.time()
        }

        keywords = self.extract_keywords(definition)
        self.concept_keywords[concept] = keywords

        print(f"添加概念 '{concept}': {definition}")
        print(f"  提取关键词: {keywords}")

    def extract_keywords(self, text: str, top_k: Optional[int] = None) -> List[str]:
        """从文本中提取关键词，使用jieba分词并过滤停用词。

        :param text: 输入文本
        :param top_k: 返回的关键词数量，默认从配置中读取
        :return: 关键词列表
        """
        if top_k is None:
            top_k = KEYWORD_CONFIG['top_k']

        words = jieba.cut(text)
        filtered_words = [
            word for word in words
            if len(word) > KEYWORD_CONFIG['min_word_length'] and word not in KEYWORD_CONFIG['stop_words']
        ]

        word_freq = defaultdict(int)
        for word in filtered_words:
            word_freq[word] += 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]

    def expand_concept_network(self, concept: str, max_depth: Optional[int] = None, current_depth: int = 0) -> None:
        """递归扩展概念网络，将概念的关键词（若也是已定义概念）连接起来。

        :param concept: 当前概念
        :param max_depth: 最大递归深度
        :param current_depth: 当前深度
        """
        if max_depth is None:
            max_depth = NETWORK_CONFIG['max_expansion_depth']

        if current_depth >= max_depth or concept not in self.concept_keywords:
            return

        keywords = self.concept_keywords[concept]

        for keyword in keywords:
            if keyword in self.concept_definitions:
                similarity = self.calculate_semantic_similarity(concept, keyword)
                self.semantic_network[concept][keyword] = similarity
                self.semantic_network[keyword][concept] = similarity

                if current_depth < max_depth - 1:
                    self.expand_concept_network(keyword, max_depth, current_depth + 1)

    def calculate_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """计算两个概念的语义相似度，采用Jaccard相似度（关键词交集/并集）与领域相似度的加权和。

        :param concept1: 第一个概念
        :param concept2: 第二个概念
        :return: 相似度值 (0~1)
        """
        if concept1 not in self.concept_keywords or concept2 not in self.concept_keywords:
            return 0.0

        keywords1 = set(self.concept_keywords[concept1])
        keywords2 = set(self.concept_keywords[concept2])

        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))

        if union == 0:
            return 0.0

        jaccard_similarity = intersection / union

        domain_similarity = self._calculate_domain_similarity(concept1, concept2)

        combined_similarity = 0.7 * jaccard_similarity + 0.3 * domain_similarity
        return min(combined_similarity, 1.0)

    def _calculate_domain_similarity(self, concept1: str, concept2: str) -> float:
        """根据概念所属领域计算相似度。

        :return: 领域相似度
        """
        domain1 = self.get_domain(concept1)
        domain2 = self.get_domain(concept2)

        if domain1 == domain2:
            return 1.0
        elif (domain1 == "principles" or domain2 == "principles"):
            return 0.6
        else:
            return 0.2

    def build_comprehensive_network(self, num_concepts: Optional[int] = None) -> None:
        """构建综合概念网络。

        根据核心概念定义构建所有概念节点，计算两两相似度，并递归扩展。

        :param num_concepts: 可选，指定使用的核心概念数量（从配置中截取前N个）
        """
        self._predefine_core_concepts(num_concepts)

        all_concepts = list(self.concept_definitions.keys())
        print(f"开始构建语义网络，共有 {len(all_concepts)} 个概念")

        threshold = NETWORK_CONFIG['similarity_threshold']
        for i, concept1 in enumerate(all_concepts):
            for j, concept2 in enumerate(all_concepts[i + 1:], i + 1):
                similarity = self.calculate_semantic_similarity(concept1, concept2)
                if similarity > threshold:
                    self.semantic_network[concept1][concept2] = similarity
                    self.semantic_network[concept2][concept1] = similarity

        for concept in all_concepts:
            self.expand_concept_network(concept)

        print(f"语义网络构建完成! 包含 {len(self.semantic_network)} 个概念节点")
        total_connections = sum(len(neighbors) for neighbors in self.semantic_network.values()) // 2
        print(f"网络连接数: {total_connections}")

    def _predefine_core_concepts(self, num_concepts: Optional[int] = None) -> None:
        """从配置中加载核心概念定义。

        :param num_concepts: 可选，指定加载的概念数量
        """
        if num_concepts is not None:
            all_concepts = list(CORE_CONCEPT_DEFINITIONS.keys())
            if num_concepts > len(all_concepts):
                print(f"警告：请求的概念数{num_concepts}超过最大概念数{len(all_concepts)}，使用最大概念数")
                num_concepts = len(all_concepts)

            selected_concepts = all_concepts[:num_concepts]
            for concept in selected_concepts:
                definition = CORE_CONCEPT_DEFINITIONS[concept]
                self.add_concept_definition(concept, definition, "predefined")
            print(f"使用前 {num_concepts} 个核心概念构建语义网络")
        else:
            for concept, definition in CORE_CONCEPT_DEFINITIONS.items():
                self.add_concept_definition(concept, definition, "predefined")
            print(f"使用全部 {len(CORE_CONCEPT_DEFINITIONS)} 个核心概念构建语义网络")

    def find_cross_domain_paths(self, start_concept: str, end_concept: str,
                                 max_path_length: Optional[int] = None) -> List[Tuple[List[str], float]]:
        """寻找两个概念间的跨领域路径（广度优先搜索）。

        :param start_concept: 起始概念
        :param end_concept: 目标概念
        :param max_path_length: 最大路径长度
        :return: 列表，每个元素为 (路径节点列表, 路径累积相似度)
        """
        if max_path_length is None:
            max_path_length = NETWORK_CONFIG['max_path_length']

        if start_concept not in self.semantic_network or end_concept not in self.semantic_network:
            return []

        max_paths = NETWORK_CONFIG['max_paths_to_find']
        queue = [(start_concept, [start_concept], 1.0)]
        visited = {start_concept}
        found_paths = []

        while queue and len(found_paths) < max_paths:
            current, path, path_similarity = queue.pop(0)

            if current == end_concept and len(path) > 1:
                found_paths.append((path, path_similarity))
                continue

            if len(path) >= max_path_length:
                continue

            for neighbor, similarity in self.semantic_network[current].items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_similarity = path_similarity * similarity
                    queue.append((neighbor, path + [neighbor], new_similarity))

        found_paths.sort(key=lambda x: x[1], reverse=True)
        return found_paths

    def get_domain(self, concept: str) -> str:
        """返回概念所属的领域，若未定义则返回 'other'。"""
        for domain, concepts in CONCEPT_DOMAINS.items():
            if concept in concepts:
                return domain
        return "other"

    def visualize_semantic_network(self, highlight_concepts: Optional[List[str]] = None) -> None:
        """可视化语义网络（需安装matplotlib）。"""
        from utils.visualization import visualize_semantic_network
        visualize_semantic_network(self.semantic_network, self.concept_definitions, highlight_concepts)

        import matplotlib.pyplot as plt
        import os
        from datetime import datetime
        fig_dir = "results/semantic_network"
        os.makedirs(fig_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"semantic_network_{timestamp}.png"),
                    dpi=300, bbox_inches='tight')
        print(f"语义网络图已保存到 results/semantic_network/")


class MetaStructureSimilarity:
    """基于元结构的相似度计算，将概念映射到元结构空间，计算余弦相似度。"""

    def __init__(self):
        self.meta_structure_map = META_STRUCTURE_MAP
        self.concept_to_meta = {}
        for meta, concepts in self.meta_structure_map.items():
            for concept in concepts:
                self.concept_to_meta[concept] = meta

    def map_to_meta_structure(self, concept: str) -> np.ndarray:
        """将概念映射到元结构空间的向量。

        :param concept: 概念名称
        :return: 长度为元结构数量的向量，对应维度为1表示完全匹配，0.5表示关键词匹配。
        """
        meta_vector = np.zeros(len(self.meta_structure_map))
        meta_list = list(self.meta_structure_map.keys())

        if concept in self.concept_to_meta:
            meta = self.concept_to_meta[concept]
            idx = meta_list.index(meta)
            meta_vector[idx] = 1.0

        for i, meta in enumerate(meta_list):
            related_concepts = self.meta_structure_map[meta]
            for related in related_concepts:
                if related in concept:
                    meta_vector[i] = max(meta_vector[i], 0.5)

        return meta_vector

    def meta_structure_similarity(self, concept1: str, concept2: str) -> float:
        """计算两个概念在元结构空间中的余弦相似度。

        :return: 相似度 (0~1)
        """
        vec1 = self.map_to_meta_structure(concept1)
        vec2 = self.map_to_meta_structure(concept2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class EnhancedSemanticConceptNetwork(SemanticConceptNetwork):
    """增强的语义概念网络，整合元结构相似度，提供多种相似度计算方法。"""

    def __init__(self, num_concepts: Optional[int] = None):
        super().__init__()
        self.meta_similarity = MetaStructureSimilarity()
        self.concept_frequency = defaultdict(int)
        self.num_concepts = num_concepts

    def build_comprehensive_network(self) -> None:
        """重写父类方法，使用 num_concepts 构建网络。"""
        self._predefine_core_concepts(self.num_concepts)

        all_concepts = list(self.concept_definitions.keys())
        print(f"开始构建增强语义网络，共有 {len(all_concepts)} 个概念")

        threshold = NETWORK_CONFIG['similarity_threshold']
        for i, concept1 in enumerate(all_concepts):
            for j, concept2 in enumerate(all_concepts[i + 1:], i + 1):
                similarity = self.calculate_semantic_similarity(concept1, concept2)
                if similarity > threshold:
                    self.semantic_network[concept1][concept2] = similarity
                    self.semantic_network[concept2][concept1] = similarity

        for concept in all_concepts:
            self.expand_concept_network(concept)

        print(f"增强语义网络构建完成! 包含 {len(self.semantic_network)} 个概念节点")
        total_connections = sum(len(neighbors) for neighbors in self.semantic_network.values()) // 2
        print(f"网络连接数: {total_connections}")

    def calculate_enhanced_similarity(self, concept1: str, concept2: str,
                                      method: str = "combined") -> float:
        """增强的相似度计算，支持多种策略。

        :param concept1: 第一个概念
        :param concept2: 第二个概念
        :param method: 计算方法，可选 'semantic_only', 'meta_only', 'combined', 'adaptive'
        :return: 相似度值
        """
        if method == "semantic_only":
            return self.calculate_semantic_similarity(concept1, concept2)

        elif method == "meta_only":
            return self.meta_similarity.meta_structure_similarity(concept1, concept2)

        elif method == "combined":
            semantic_sim = self.calculate_semantic_similarity(concept1, concept2)
            meta_sim = self.meta_similarity.meta_structure_similarity(concept1, concept2)

            complexity1 = self._concept_complexity(concept1)
            complexity2 = self._concept_complexity(concept2)
            avg_complexity = (complexity1 + complexity2) / 2

            meta_weight = min(0.7, 0.3 + avg_complexity * 0.4)
            semantic_weight = 1 - meta_weight

            return semantic_weight * semantic_sim + meta_weight * meta_sim

        elif method == "adaptive":
            semantic_sim = self.calculate_semantic_similarity(concept1, concept2)
            meta_sim = self.meta_similarity.meta_structure_similarity(concept1, concept2)
            abstraction_level = self._abstraction_level(concept1, concept2)

            if abstraction_level > 0.6:
                return meta_sim * 0.8 + semantic_sim * 0.2
            elif abstraction_level < 0.3:
                return semantic_sim * 0.8 + meta_sim * 0.2
            else:
                return (semantic_sim + meta_sim) / 2

        return 0.0

    def _concept_complexity(self, concept: str) -> float:
        """估计概念复杂度，基于关键词数量及多样性。"""
        if concept not in self.concept_keywords:
            return 0.5

        keywords = self.concept_keywords[concept]
        num_keywords = len(keywords)
        keyword_variety = len(set(keywords)) / max(1, num_keywords)

        complexity = min(1.0, num_keywords / 15 * 0.6 + keyword_variety * 0.4)
        return complexity

    def _abstraction_level(self, concept1: str, concept2: str) -> float:
        """估计两个概念的平均抽象程度。"""
        def single_concept_abstraction(concept: str) -> float:
            meta_vec = self.meta_similarity.map_to_meta_structure(concept)
            abstraction_score = np.sum(meta_vec) / len(meta_vec)

            if concept in self.concept_definitions:
                definition_length = len(self.concept_definitions[concept]['definition'])
                length_factor = min(1.0, definition_length / 100)
                abstraction_score = 0.7 * abstraction_score + 0.3 * length_factor

            return abstraction_score

        abstraction1 = single_concept_abstraction(concept1)
        abstraction2 = single_concept_abstraction(concept2)
        return (abstraction1 + abstraction2) / 2


if __name__ == "__main__":
    # 简单测试
    net = SemanticConceptNetwork()
    net.build_comprehensive_network(num_concepts=51)
    print("语义网络构建成功")