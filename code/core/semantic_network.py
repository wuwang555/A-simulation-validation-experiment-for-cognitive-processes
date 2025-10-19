import time
from collections import defaultdict
import jieba
import numpy as np
from config import CORE_CONCEPT_DEFINITIONS

class SemanticConceptNetwork:
    """基于定义关键词的语义概念网络"""

    def __init__(self):
        self.concept_definitions = {}
        self.concept_keywords = {}
        self.semantic_network = defaultdict(dict)

    def add_concept_definition(self, concept, definition, source="manual"):
        """添加概念定义"""
        self.concept_definitions[concept] = {
            'definition': definition,
            'source': source,
            'timestamp': time.time()
        }

        keywords = self.extract_keywords(definition)
        self.concept_keywords[concept] = keywords

        print(f"添加概念 '{concept}': {definition}")
        print(f"  提取关键词: {keywords}")

    def extract_keywords(self, text, top_k=10):
        """从文本中提取关键词 - 改进版本"""
        # 使用jieba进行分词
        words = jieba.cut(text)

        # 过滤停用词和短词
        stop_words = {
            '的', '是', '在', '和', '与', '或', '等', '这个', '那个', '一种',
            '研究', '包括', '通过', '给定', '任何', '两个', '某种', '一个'
        }

        filtered_words = [
            word for word in words
            if len(word) > 1 and word not in stop_words
        ]

        # 统计词频
        word_freq = defaultdict(int)
        for word in filtered_words:
            word_freq[word] += 1

        # 返回频率最高的关键词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]

    def expand_concept_network(self, concept, max_depth=2, current_depth=0):
        """递归扩展概念网络"""
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

    def calculate_semantic_similarity(self, concept1, concept2):
        """计算两个概念的语义相似度 - 增强版本"""
        if concept1 not in self.concept_keywords or concept2 not in self.concept_keywords:
            return 0.0

        keywords1 = set(self.concept_keywords[concept1])
        keywords2 = set(self.concept_keywords[concept2])

        # Jaccard相似度
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))

        if union == 0:
            return 0.0

        jaccard_similarity = intersection / union

        # 考虑领域相似性
        domain_similarity = self._calculate_domain_similarity(concept1, concept2)

        # 综合相似度
        combined_similarity = 0.7 * jaccard_similarity + 0.3 * domain_similarity

        return min(combined_similarity, 1.0)

    def _calculate_domain_similarity(self, concept1, concept2):
        """计算领域相似度"""
        domain1 = self.get_domain(concept1)
        domain2 = self.get_domain(concept2)

        if domain1 == domain2:
            return 1.0
        elif (domain1 == "principles" or domain2 == "principles"):
            return 0.6  # 原理节点与各领域都有一定相似性
        else:
            return 0.2  # 不同领域的基础相似度

    def build_comprehensive_network(self):
        """构建综合概念网络 - 修复版本"""
        # 预定义核心概念
        self._predefine_core_concepts()

        # 为所有概念扩展网络 - 修复：确保所有节点都被处理
        all_concepts = list(self.concept_definitions.keys())
        print(f"开始构建语义网络，共有 {len(all_concepts)} 个概念")

        # 首先建立所有概念之间的直接连接
        for i, concept1 in enumerate(all_concepts):
            for j, concept2 in enumerate(all_concepts[i + 1:], i + 1):
                similarity = self.calculate_semantic_similarity(concept1, concept2)
                if similarity > 0.1:  # 设置较低的阈值以建立更多连接
                    self.semantic_network[concept1][concept2] = similarity
                    self.semantic_network[concept2][concept1] = similarity

        # 然后进行深度扩展
        for concept in all_concepts:
            self.expand_concept_network(concept, max_depth=3)

        print(f"语义网络构建完成! 包含 {len(self.semantic_network)} 个概念节点")
        print(f"网络连接数: {sum(len(neighbors) for neighbors in self.semantic_network.values()) // 2}")
    def _predefine_core_concepts(self):
        """预定义核心概念及其关系"""
        core_definitions = {
            # 物理学概念
            "牛顿定律": "物体运动的基本定律，描述了力与运动的关系",
            "力学": "研究物体运动和受力情况的物理学分支",
            "运动学": "研究物体运动而不考虑力的物理学分支",
            "能量守恒": "能量既不会凭空产生也不会凭空消失的物理定律",
            "动量": "物体运动状态的量度，质量与速度的乘积",
            "万有引力": "任何两个有质量的物体之间相互吸引的力",
            "摩擦力": "两个接触表面之间阻碍相对运动的力",
            "静电力": "电荷之间相互作用的力",

            # 数学概念
            "微积分": "研究变化和累积的数学分支，包括微分和积分",
            "几何学": "研究空间形状大小和相对位置的数学分支",
            "拓扑学": "研究空间在连续变形下不变性质的数学分支",
            "线性代数": "研究向量空间和线性映射的数学分支",
            "概率论": "研究随机现象数量规律的数学分支",
            "统计学": "收集分析解释数据的数学科学",
            "代数": "研究数学符号和运算规则的数学分支",
            "离散数学": "研究离散结构的数学分支",

            # 计算机科学概念
            "算法": "解决问题的一系列明确的计算步骤",
            "数据结构": "计算机中组织和存储数据的方式",
            "机器学习": "让计算机通过经验自动改进性能的人工智能分支",
            "神经网络": "模仿生物神经网络的计算模型",
            "计算机视觉": "让计算机理解和分析视觉信息的技术",
            "自然语言处理": "计算机与人类自然语言交互的技术",
            "数据库": "结构化信息或数据的有组织集合",
            "操作系统": "管理计算机硬件与软件资源的系统软件",

            # 基础原理概念
            "优化": "在给定约束下找到最佳解决方案的过程",
            "变换": "从一个形式或状态转换为另一个的过程",
            "迭代": "重复反馈过程的活动",
            "抽象": "提取主要特征忽略次要细节的思维过程",
            "模式识别": "通过算法识别数据中模式的过程",
            "对称": "物体在某种变换下保持不变的性质",
            "递归": "通过函数调用自身来解决问题的方法",
            "归纳": "从特殊到一般的推理方法"
        }

        for concept, definition in core_definitions.items():
            self.add_concept_definition(concept, definition, "predefined")

    def find_cross_domain_paths(self, start_concept, end_concept, max_path_length=4):
        """寻找跨领域的概念路径"""
        if start_concept not in self.semantic_network or end_concept not in self.semantic_network:
            return []

        queue = [(start_concept, [start_concept], 1.0)]
        visited = {start_concept}
        found_paths = []

        while queue and len(found_paths) < 10:
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

    def get_domain(self, concept):
        """获取概念所属领域"""
        domains = {
            "physics": ["牛顿定律", "力学", "运动学", "能量守恒", "动量", "万有引力", "摩擦力", "静电力"],
            "math": ["微积分", "几何学", "拓扑学", "线性代数", "概率论", "统计学", "代数", "离散数学"],
            "cs": ["算法", "数据结构", "机器学习", "神经网络", "计算机视觉", "自然语言处理", "数据库", "操作系统"],
            "principles": ["优化", "变换", "迭代", "抽象", "模式识别", "对称", "递归", "归纳"]
        }

        for domain, concepts in domains.items():
            if concept in concepts:
                return domain
        return "other"

    def visualize_semantic_network(self, highlight_concepts=None):
        """可视化语义网络"""
        from utils.visualization import visualize_semantic_network
        visualize_semantic_network(self.semantic_network, self.concept_definitions, highlight_concepts)

class MetaStructureSimilarity:
    """基于元结构的相似度计算"""

    def __init__(self):
        # 定义元结构映射
        self.meta_structure_map = {
            "信息": ["数据", "知识", "信号", "消息", "情报"],
            "几何": ["结构", "形状", "关系", "形式", "布局", "拓扑"],
            "运动": ["遍历", "变化", "过程", "流动", "迁移", "转换"],
            "方法": ["算法", "策略", "技术", "途径", "手段", "方法论"],
            "历史": ["迭代", "回归", "演化", "发展", "进程", "时间"],
            "能量": ["能耗", "功率", "动力", "资源", "消耗"],
            "抽象": ["概念", "思想", "理论", "原理", "范式"],
            "系统": ["网络", "集合", "整体", "组织", "架构"]
        }

        # 反向映射：从具体概念到元结构
        self.concept_to_meta = {}
        for meta, concepts in self.meta_structure_map.items():
            for concept in concepts:
                self.concept_to_meta[concept] = meta

    def map_to_meta_structure(self, concept):
        """将概念映射到元结构空间"""
        meta_vector = np.zeros(len(self.meta_structure_map))
        meta_list = list(self.meta_structure_map.keys())

        # 直接映射
        if concept in self.concept_to_meta:
            meta = self.concept_to_meta[concept]
            idx = meta_list.index(meta)
            meta_vector[idx] = 1.0

        # 关键词匹配
        for i, meta in enumerate(meta_list):
            related_concepts = self.meta_structure_map[meta]
            for related in related_concepts:
                if related in concept:
                    meta_vector[i] = max(meta_vector[i], 0.5)

        return meta_vector

    def meta_structure_similarity(self, concept1, concept2):
        """基于元结构的相似度计算"""
        vec1 = self.map_to_meta_structure(concept1)
        vec2 = self.map_to_meta_structure(concept2)

        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

class EnhancedSemanticConceptNetwork(SemanticConceptNetwork):
    """增强的语义概念网络，整合元结构相似度"""

    def __init__(self):
        super().__init__()
        self.meta_similarity = MetaStructureSimilarity()
        self.concept_frequency = defaultdict(int)  # 概念出现频率

    def calculate_enhanced_similarity(self, concept1, concept2, method="combined"):
        """增强的相似度计算，整合多种方法"""

        if method == "semantic_only":
            return self.calculate_semantic_similarity(concept1, concept2)

        elif method == "meta_only":
            return self.meta_similarity.meta_structure_similarity(concept1, concept2)

        elif method == "combined":
            # 语义相似度
            semantic_sim = self.calculate_semantic_similarity(concept1, concept2)

            # 元结构相似度
            meta_sim = self.meta_similarity.meta_structure_similarity(concept1, concept2)

            # 动态权重调整（基于概念复杂度）
            complexity1 = self._concept_complexity(concept1)
            complexity2 = self._concept_complexity(concept2)
            avg_complexity = (complexity1 + complexity2) / 2

            # 复杂度越高，越依赖元结构相似度
            meta_weight = min(0.7, 0.3 + avg_complexity * 0.4)
            semantic_weight = 1 - meta_weight

            return semantic_weight * semantic_sim + meta_weight * meta_sim

        elif method == "adaptive":
            # 自适应方法：根据概念特性选择最佳相似度
            semantic_sim = self.calculate_semantic_similarity(concept1, concept2)
            meta_sim = self.meta_similarity.meta_structure_similarity(concept1, concept2)

            # 如果概念很抽象，偏向元结构；如果很具体，偏向语义
            abstraction_level = self._abstraction_level(concept1, concept2)

            if abstraction_level > 0.6:  # 高度抽象
                return meta_sim * 0.8 + semantic_sim * 0.2
            elif abstraction_level < 0.3:  # 具体概念
                return semantic_sim * 0.8 + meta_sim * 0.2
            else:  # 中等抽象
                return (semantic_sim + meta_sim) / 2

        return 0.0

    def _concept_complexity(self, concept):
        """评估概念复杂度"""
        if concept not in self.concept_keywords:
            return 0.5

        keywords = self.concept_keywords[concept]

        # 基于关键词数量和多样性评估复杂度
        num_keywords = len(keywords)
        keyword_variety = len(set(keywords)) / max(1, num_keywords)

        complexity = min(1.0, num_keywords / 15 * 0.6 + keyword_variety * 0.4)
        return complexity

    def _abstraction_level(self, concept1, concept2):
        """评估概念的抽象程度"""

        def single_concept_abstraction(concept):
            # 基于元结构映射判断抽象程度
            meta_vec = self.meta_similarity.map_to_meta_structure(concept)
            abstraction_score = np.sum(meta_vec) / len(meta_vec)

            # 基于定义长度调整（通常抽象概念定义更长）
            if concept in self.concept_definitions:
                definition_length = len(self.concept_definitions[concept]['definition'])
                length_factor = min(1.0, definition_length / 100)
                abstraction_score = 0.7 * abstraction_score + 0.3 * length_factor

            return abstraction_score

        abstraction1 = single_concept_abstraction(concept1)
        abstraction2 = single_concept_abstraction(concept2)

        return (abstraction1 + abstraction2) / 2

