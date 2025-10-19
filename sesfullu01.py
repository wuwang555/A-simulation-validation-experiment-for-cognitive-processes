import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from collections import defaultdict
import random
from collections import defaultdict
import math
import json
import time
from enum import Enum
import jieba
from typing import Dict, Any

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化jieba
jieba.initialize()


class CognitiveState(Enum):
    """认知状态枚举"""
    FOCUSED = "专注状态"
    EXPLORATORY = "探索状态"
    FATIGUED = "疲劳状态"
    INSPIRED = "灵感状态"


class IndividualVariation:
    """个体差异模拟器"""

    def __init__(self, base_parameters: Dict[str, Any], variation_ranges: Dict[str, Any]):
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
                    min_val = base_value * (1 - variation)
                    max_val = base_value * (1 + variation)
                    params[param] = np.random.uniform(min_val, max_val)
                elif isinstance(variation, tuple) and len(variation) == 2:
                    params[param] = np.random.uniform(variation[0], variation[1])
                else:
                    params[param] = base_value
            else:
                params[param] = base_value

        self.individual_parameters[individual_id] = params
        return params


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
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            G = nx.Graph()

            for concept in self.semantic_network:
                domain = self.get_domain(concept)
                G.add_node(concept, domain=domain)

            for concept1, neighbors in self.semantic_network.items():
                for concept2, similarity in neighbors.items():
                    if similarity > 0.1:
                        G.add_edge(concept1, concept2, weight=similarity)

            domain_colors = {
                "physics": "lightblue",
                "math": "lightgreen",
                "cs": "lightcoral",
                "principles": "gold",
                "other": "lightgray"
            }

            node_colors = [domain_colors[G.nodes[node].get('domain', 'other')] for node in G.nodes()]

            plt.figure(figsize=(16, 12))
            pos = nx.spring_layout(G, k=3, iterations=50)

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)

            edges = G.edges()
            weights = [G[u][v]['weight'] * 3 for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color='gray')

            nx.draw_networkx_labels(G, pos, font_size=8, font_family='SimHei')

            plt.title("语义概念网络", fontsize=16, fontfamily='SimHei')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("需要安装networkx和matplotlib来可视化网络")


class AdjustedSubjectiveCognitiveGraph:
    """调整参数后的主观认知图"""

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        self.G = nx.Graph()
        self.traversal_history = []
        self.concept_centers = {}
        self.iteration_count = 0
        self.energy_history = []

        # 主观认知状态参数
        self.current_state = CognitiveState.FOCUSED
        self.subjective_energy = 1.5
        self.cognitive_energy_history = []

        # 状态转移参数
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

        # 状态对应的主观能耗范围
        self.state_energy_ranges = {
            CognitiveState.FOCUSED: (1.5, 2.5),
            CognitiveState.EXPLORATORY: (1.0, 1.8),
            CognitiveState.FATIGUED: (0.8, 1.2),
            CognitiveState.INSPIRED: (2.0, 3.0)
        }

        # 个体参数
        self.individual_params = individual_params
        self.forgetting_rate = individual_params.get('forgetting_rate', 0.002)
        self.base_learning_rate = individual_params.get('base_learning_rate', 0.85)
        self.hard_traversal_bias = individual_params.get('hard_traversal_bias', 0.0)
        self.soft_traversal_bias = individual_params.get('soft_traversal_bias', 0.0)
        self.compression_bias = individual_params.get('compression_bias', 0.0)
        self.migration_bias = individual_params.get('migration_bias', 0.0)
        self.learning_rate_variation = individual_params.get('learning_rate_variation', 0.1)

        # 硬遍历和软遍历的能耗分配策略
        self.hard_traversal_energy_ratio = 0.6
        self.soft_traversal_energy_ratio = 0.4

        self.last_activation_time = {}
        self.network_seed = network_seed

    def update_cognitive_state(self):
        """更新主观认知状态"""
        current_state = self.current_state
        transition_probs = self.state_transition_matrix[current_state]

        rand_val = random.random()
        cumulative_prob = 0

        for new_state, prob in transition_probs.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                if new_state != current_state:
                    self.current_state = new_state
                    self._update_subjective_energy()
                break

        self.cognitive_energy_history.append({
            'iteration': self.iteration_count,
            'state': self.current_state,
            'energy': self.subjective_energy
        })

    def _update_subjective_energy(self):
        """根据当前状态更新主观认知能耗"""
        energy_range = self.state_energy_ranges[self.current_state]
        self.subjective_energy = random.uniform(energy_range[0], energy_range[1])

        energy_variation = self.individual_params.get('energy_variation', 0.1)
        self.subjective_energy *= random.uniform(1 - energy_variation, 1 + energy_variation)
        self.subjective_energy = max(0.1, min(3.0, self.subjective_energy))

    def can_traverse_edge(self, edge_energy, traversal_type):
        """检查是否可以遍历某条边（考虑主观认知能耗）"""
        if traversal_type == "hard":
            required_energy = edge_energy * 0.8
            available_energy = self.subjective_energy * self.hard_traversal_energy_ratio
        else:
            required_energy = edge_energy * 0.6
            available_energy = self.subjective_energy * self.soft_traversal_energy_ratio

        return available_energy >= required_energy, available_energy - required_energy

    def traverse_path(self, path, traversal_type="hard"):
        """改进的遍历函数 - 考虑主观认知状态"""
        if random.random() < 0.1:
            self.update_cognitive_state()

        total_required_energy = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                edge_energy = self.G[u][v]['weight']
                total_required_energy += edge_energy

        can_traverse, energy_balance = self.can_traverse_edge(total_required_energy, traversal_type)

        if not can_traverse:
            if random.random() < 0.2 and self.current_state != CognitiveState.FATIGUED:
                can_traverse = True
                energy_balance = -0.5

        if not can_traverse:
            if random.random() < 0.3:
                self.current_state = CognitiveState.FATIGUED
                self._update_subjective_energy()
            return

        self.traversal_history.append((path, traversal_type, self.iteration_count))
        current_time = self.iteration_count

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                self.last_activation_time[(u, v)] = current_time
                self.G[u][v]['traversal_count'] += 1

                similarity = 0.5
                base_rate = self.base_learning_rate

                individual_learning_variation = np.random.uniform(
                    1 - self.learning_rate_variation,
                    1 + self.learning_rate_variation
                )

                if traversal_type == "hard":
                    learning_rate = base_rate * (0.7 + 0.3 * similarity) * individual_learning_variation
                else:
                    learning_rate = base_rate * 0.9 * individual_learning_variation

                current_weight = self.G[u][v]['weight']
                learning_effect = learning_rate * (current_weight / 2.0)

                new_weight = max(0.05, current_weight * (1 - learning_effect))
                self.G[u][v]['weight'] = new_weight

        self._post_traversal_state_update(traversal_type, energy_balance)

    def _post_traversal_state_update(self, traversal_type, energy_balance):
        """遍历后的状态更新"""
        if energy_balance > 0.3:
            if traversal_type == "hard" and random.random() < 0.4:
                self.current_state = CognitiveState.FOCUSED
            elif traversal_type == "soft" and random.random() < 0.3:
                self.current_state = CognitiveState.EXPLORATORY
        elif energy_balance < -0.2:
            if random.random() < 0.5:
                self.current_state = CognitiveState.FATIGUED

        self._update_subjective_energy()

    def _apply_forgetting(self):
        """应用遗忘机制到所有边"""
        current_time = self.iteration_count

        for u, v in self.G.edges():
            time_since_last_activation = current_time - self.last_activation_time.get((u, v), 0)
            if time_since_last_activation > 0:
                current_energy = self.G[u][v]['weight']
                similarity = 0.5

                forget_factor = self.forgetting_function(
                    current_time,
                    self.last_activation_time.get((u, v), 0),
                    current_energy,
                    similarity
                )

                new_weight = self.G[u][v]['weight'] * (1 + forget_factor)
                original = self.G[u][v].get('original_weight', 2.0)
                self.G[u][v]['weight'] = min(new_weight, original)

    def forgetting_function(self, current_time, last_activation_time, current_energy, similarity):
        """基于指数衰减的遗忘时间函数"""
        time_gap = current_time - last_activation_time

        base_forgetting = 1 - math.exp(-time_gap / 500)
        energy_factor = 0.5 + 0.5 * (current_energy / 2.0)
        similarity_protection = 1 - (similarity * 0.5)

        forgetting_factor = (base_forgetting * energy_factor *
                             similarity_protection * self.forgetting_rate)

        return min(forgetting_factor, 0.1)

    def monte_carlo_iteration(self, max_iterations=5000):
        """改进的蒙特卡洛模拟 - 考虑主观认知状态"""
        print(f"初始认知状态: {self.current_state.value}, 主观能耗: {self.subjective_energy:.2f}")

        for iteration in range(max_iterations):
            self.iteration_count += 1

            if iteration % 100 == 0:
                self.update_cognitive_state()

            self._apply_forgetting()

            current_avg_energy = self.get_average_energy()
            self.energy_history.append(current_avg_energy)

            operation = self._select_operation_based_on_state()

            if operation == "hard_traversal":
                self._state_based_hard_traversal()
            elif operation == "soft_traversal":
                self._state_based_soft_traversal()
            elif operation == "compression":
                self._random_compression()
            elif operation == "migration":
                self._random_migration()

            if iteration % 500 == 0:
                stats = self.get_network_stats()
                print(f"迭代 {iteration}, 状态: {self.current_state.value}, "
                      f"主观能耗: {self.subjective_energy:.2f}, 网络能耗: {current_avg_energy:.3f}")

    def _select_operation_based_on_state(self):
        """基于认知状态选择操作类型"""
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

    def _state_based_hard_traversal(self):
        """基于状态的硬遍历"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return

        start_node = random.choice(available_nodes)
        path = self._find_hard_traversal_path(start_node, 3)

        if path and len(path) >= 2:
            self.traverse_path(path, "hard")

    def _state_based_soft_traversal(self):
        """基于状态的软遍历"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return

        start_node = random.choice(available_nodes)
        path = self._find_soft_traversal_path(start_node, 2)

        if path and len(path) >= 2:
            self.traverse_path(path, "soft")

    def _find_hard_traversal_path(self, start_node, max_length):
        """硬遍历路径搜索"""
        path = [start_node]
        current_node = start_node

        for _ in range(max_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break

            neighbors.sort(key=lambda n: self.G[current_node][n]['weight'])

            found_next = False
            for neighbor in neighbors[:3]:
                if neighbor not in path:
                    edge_energy = self.G[current_node][neighbor]['weight']
                    can_traverse, _ = self.can_traverse_edge(edge_energy, "hard")
                    if can_traverse:
                        path.append(neighbor)
                        current_node = neighbor
                        found_next = True
                        break

            if not found_next:
                break

        return path if len(path) >= 2 else None

    def _find_soft_traversal_path(self, start_node, max_length):
        """软遍历路径搜索"""
        path = [start_node]
        current_node = start_node

        for _ in range(max_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break

            random.shuffle(neighbors)

            found_next = False
            for neighbor in neighbors:
                if neighbor not in path:
                    edge_energy = self.G[current_node][neighbor]['weight']
                    can_traverse, _ = self.can_traverse_edge(edge_energy, "soft")
                    if can_traverse:
                        path.append(neighbor)
                        current_node = neighbor
                        found_next = True
                        break

            if not found_next:
                break

        return path if len(path) >= 2 else None

    def _random_compression(self):
        """随机概念压缩尝试"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 3:
            return

        if random.random() > 0.10:
            return

        if self.iteration_count < 2000:
            return

        center_candidate = random.choice(available_nodes)

        good_neighbors = []
        for neighbor in self.G.neighbors(center_candidate):
            if (self.G[center_candidate][neighbor]['weight'] < 1.0 and self.calculate_semantic_similarity(center_candidate, neighbor) > 0.4):
                good_neighbors.append(neighbor)

        if len(good_neighbors) >= 3:
            num_to_compress = random.randint(2, min(3, len(good_neighbors)))
            nodes_to_compress = random.sample(good_neighbors, num_to_compress)

            compression_strength = random.uniform(0.4, 0.6)
            self.conceptual_compression(center_candidate, nodes_to_compress, compression_strength)

    def conceptual_compression(self, center_node, related_nodes, compression_strength=0.5):
        """概念压缩：强化中心节点与相关节点的连接"""
        if len(related_nodes) < 2:
            return False

        for node in related_nodes:
            if self.G.has_edge(center_node, node):
                current_energy = self.G[center_node][node]['weight']
                compressed_energy = max(0.05, current_energy * compression_strength)
                self.G[center_node][node]['weight'] = compressed_energy

        self.concept_centers[center_node] = {
            'related_nodes': related_nodes,
            'compression_strength': compression_strength,
            'iteration': self.iteration_count
        }

        return True

    def _random_migration(self):
        """随机第一性原理迁移尝试"""
        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 4:
            return

        if random.random() > 0.05:
            return

        start_node, end_node = random.sample(available_nodes, 2)

        principle_candidates = [n for n in available_nodes
                                if n not in [start_node, end_node]]

        if not principle_candidates:
            return

        num_principles = random.randint(1, min(2, len(principle_candidates)))
        selected_principles = random.sample(principle_candidates, num_principles)

        exploration_bonus = random.uniform(0.05, 0.15)
        self.first_principles_migration(start_node, end_node, selected_principles, exploration_bonus)

    def first_principles_migration(self, start_node, end_node, principle_nodes, exploration_bonus=0.1):
        """第一性原理迁移 - 修复版本"""
        best_path = None
        best_energy = float('inf')

        # 修复：确保direct_energy始终有值
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

        # 要求新路径必须明显优于直接路径
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
        plt.title('认知能耗收敛过程')
        plt.grid(True, alpha=0.3)

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

    def visualize_cognitive_states(self):
        """可视化认知状态变化"""
        if not self.cognitive_energy_history:
            return

        iterations = [e['iteration'] for e in self.cognitive_energy_history]
        energies = [e['energy'] for e in self.cognitive_energy_history]
        network_energies = self.energy_history[:len(iterations)]
        states = [e['state'] for e in self.cognitive_energy_history]

        state_colors = {
            CognitiveState.FOCUSED: 'green',
            CognitiveState.EXPLORATORY: 'blue',
            CognitiveState.FATIGUED: 'red',
            CognitiveState.INSPIRED: 'purple'
        }

        colors = [state_colors[state] for state in states]

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.scatter(iterations, energies, c=colors, alpha=0.6)
        plt.plot(iterations, energies, 'gray', alpha=0.3)
        plt.ylabel('主观认知能耗')
        plt.title('主观认知状态与能耗变化')

        for state, color in state_colors.items():
            plt.plot([], [], 'o', color=color, label=state.value)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(iterations, network_energies, 'b-', alpha=0.7)
        plt.xlabel('迭代次数')
        plt.ylabel('网络平均能耗')
        plt.title('认知网络能耗演化')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1

        print("\n=== 认知状态统计 ===")
        for state, count in state_counts.items():
            percentage = (count / len(states)) * 100
            print(f"{state.value}: {count}次 ({percentage:.1f}%)")

    def visualize_graph(self, title="认知图", figsize=(12, 8)):
        """可视化认知图"""
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42)

        node_colors = []
        node_sizes = []
        for node in self.G.nodes():
            if node in self.concept_centers:
                node_colors.append('red')
                node_sizes.append(2000)
            elif any('migration_bridges' in self.G.nodes[n] for n in self.G.nodes()):
                node_colors.append('orange')
                node_sizes.append(1500)
            else:
                node_colors.append('lightblue')
                node_sizes.append(800)

        edge_colors = []
        edge_widths = []
        for u, v in self.G.edges():
            energy = self.G[u][v]['weight']
            edge_widths.append(max(0.5, 3 - energy * 1.5))

            if energy < 0.3:
                edge_colors.append('green')
            elif energy < 0.7:
                edge_colors.append('blue')
            else:
                edge_colors.append('gray')

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

        stats = self.get_network_stats()
        print(f"网络统计:")
        print(f"  节点: {stats['nodes']}, 边: {stats['edges']}")
        print(f"  迭代次数: {stats['iterations']}")
        print(f"  平均能耗: {stats['avg_energy']:.3f}")
        print(f"  概念压缩中心: {stats['compression_centers']}")
        print(f"  迁移桥梁: {stats['migration_bridges']}")


class SemanticEnhancedCognitiveGraph(AdjustedSubjectiveCognitiveGraph):
    """基于语义增强的认知图"""

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        super().__init__(individual_params, network_seed)

        self.semantic_network = SemanticConceptNetwork()
        self.semantic_network.build_comprehensive_network()

    def calculate_semantic_similarity(self, node1, node2):
        """基于语义网络计算相似度"""
        return self.semantic_network.calculate_semantic_similarity(node1, node2)

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

    def find_semantic_migration_paths(self, start_node, end_node):
        """基于语义网络寻找迁移路径"""
        return self.semantic_network.find_cross_domain_paths(start_node, end_node)

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


class EnergyOptimizedCognitiveGraph(SemanticEnhancedCognitiveGraph):
    """能耗优化的认知图"""

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        super().__init__(individual_params, network_seed)

        # 替换为增强的语义网络
        self.semantic_network = EnhancedSemanticConceptNetwork()
        self.semantic_network.build_comprehensive_network()

        # 能耗优化参数
        self.energy_optimization_threshold = 0.3
        self.min_energy_threshold = 0.1
        self.max_energy_threshold = 2.0

    def calculate_semantic_similarity(self, node1, node2):
        """使用增强的相似度计算"""
        return self.semantic_network.calculate_enhanced_similarity(node1, node2, method="adaptive")

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
            if len(cluster_nodes) > 8:  # 最大8个节点
                cluster_nodes = cluster_nodes[:8]

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


def create_enhanced_individual_params(base_params):
    """创建增强的个体参数（包含认知状态相关参数）"""
    enhanced_params = base_params.copy()

    enhanced_params.update({
        'energy_variation': random.uniform(0.05, 0.15),
        'focus_bias': random.uniform(-0.1, 0.1),
        'exploration_bias': random.uniform(-0.1, 0.1),
        'fatigue_resistance': random.uniform(0.1, 0.3),
        'inspiration_frequency': random.uniform(0.05, 0.2)
    })

    return enhanced_params


def run_semantic_enhanced_experiment(num_individuals=3, max_iterations=10000):
    """运行语义增强的群体实验"""
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

    print(f"=== 开始语义增强群体实验：{num_individuals}个个体 ===")

    for i in range(num_individuals):
        individual_id = f"个体_{i + 1}"
        print(f"\n--- 模拟 {individual_id} ---")

        base_individual_params = variation_simulator.generate_individual(individual_id)
        individual_params = create_enhanced_individual_params(base_individual_params)

        individual_graph = SemanticEnhancedCognitiveGraph(individual_params)
        individual_graph.initialize_semantic_graph()

        initial_energy = individual_graph.get_average_energy()
        individual_graph.monte_carlo_iteration(max_iterations=max_iterations)

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
            'cognitive_states': individual_graph.cognitive_energy_history
        }

        population_results.append(result)

        print(f"{individual_id} 结果:")
        print(f"  能耗降低: {improvement:.1f}%")
        print(f"  压缩中心: {result['compression_centers']}个")
        print(f"  迁移桥梁: {result['migration_bridges']}个")

        individual_graph.visualize_cognitive_states()

    analyze_population_results(population_results)

    return population_results


def test_enhanced_features():
    """测试增强功能"""
    print("=== 测试增强的认知图模型 ===")

    # 创建个体参数
    base_params = {
        'forgetting_rate': 0.002,
        'base_learning_rate': 0.85,
        'hard_traversal_bias': 0.0,
        'soft_traversal_bias': 0.0,
        'compression_bias': 0.0,
        'migration_bias': 0.0,
        'learning_rate_variation': 0.1
    }

    # 创建能耗优化的认知图
    cognitive_graph = EnergyOptimizedCognitiveGraph(base_params)
    cognitive_graph.initialize_semantic_graph()

    print("初始网络统计:")
    stats = cognitive_graph.get_network_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试相似度计算
    test_pairs = [
        ("算法", "数据结构"),
        ("牛顿定律", "优化"),
        ("几何学", "拓扑学")
    ]

    print("\n=== 相似度计算测试 ===")
    for concept1, concept2 in test_pairs:
        similarity = cognitive_graph.calculate_semantic_similarity(concept1, concept2)
        print(f"{concept1} <-> {concept2}: {similarity:.3f}")

    # 运行蒙特卡洛模拟
    print("\n=== 开始能耗优化模拟 ===")
    cognitive_graph.monte_carlo_iteration(max_iterations=10000)

    # 智能概念压缩
    print("\n=== 智能概念压缩 ===")
    compressed_groups = cognitive_graph.improved_smart_concept_compression(
        compression_threshold=0.3,
        max_group_size=6
    )
    print(f"完成 {len(compressed_groups)} 个智能压缩")

    # 最终统计
    final_stats = cognitive_graph.get_network_stats()
    improvement = ((stats['avg_energy'] - final_stats['avg_energy']) / stats['avg_energy'] * 100)

    print(f"\n=== 最终结果 ===")
    print(f"能耗降低: {improvement:.1f}%")
    print(f"压缩中心: {final_stats['compression_centers']}个")
    print(f"迁移桥梁: {final_stats['migration_bridges']}个")

    return cognitive_graph


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

    print(f"\n=== 个体差异分析 ===")

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


def demo_semantic_network():
    """演示语义网络功能"""
    semantic_net = SemanticConceptNetwork()
    semantic_net.build_comprehensive_network()

    print("\n=== 语义网络演示 ===")

    # 显示一些概念的关键词
    sample_concepts = ["牛顿定律", "微积分", "算法", "优化"]
    for concept in sample_concepts:
        if concept in semantic_net.concept_keywords:
            print(f"{concept}: {semantic_net.concept_keywords[concept]}")

    # 寻找跨领域路径
    print("\n=== 跨领域路径示例 ===")
    domain_pairs = [
        ("牛顿定律", "算法"),
        ("微积分", "机器学习"),
        ("几何学", "计算机视觉")
    ]

    for start, end in domain_pairs:
        paths = semantic_net.find_cross_domain_paths(start, end)
        if paths:
            best_path, similarity = paths[0]
            print(f"{start} -> {end}:")
            print(f"  路径: {' -> '.join(best_path)}")
            print(f"  语义相似度: {similarity:.3f}")
        else:
            print(f"{start} -> {end}: 未找到路径")

    # 可视化语义网络
    semantic_net.visualize_semantic_network()


if __name__ == "__main__":
    # 确保安装了必要的库
    try:
        import jieba
        import networkx as nx
    except ImportError:
        print("请安装所需库: pip install jieba networkx matplotlib")
        exit(1)

    print("=== 语义增强认知图模型 ===")

    # 演示语义网络
    demo_semantic_network()

    # 运行语义增强的群体实验
    enhanced_results = run_semantic_enhanced_experiment(
        num_individuals=3,
        max_iterations=10000
    )

    # 可视化最优个体
    if enhanced_results:
        best_individual = max(enhanced_results, key=lambda x: x['improvement'])
        print(f"\n=== 可视化最优个体: {best_individual['individual_id']} ===")

        best_graph = SemanticEnhancedCognitiveGraph(best_individual['parameters'])
        best_graph.initialize_semantic_graph()
        best_graph.monte_carlo_iteration(max_iterations=10000)
        best_graph.visualize_graph(f"最优个体: {best_individual['individual_id']}")
        best_graph.visualize_energy_convergence()

        # 基于语义进行概念压缩
        print("\n=== 语义概念压缩 ===")
        compressed_groups = best_graph.conceptual_compression_based_on_semantics()
        print(f"发现 {len(compressed_groups)} 个语义压缩组")

    # 新增：测试增强功能
    print("\n" + "=" * 50)
    print("测试增强功能")
    print("=" * 50)

    enhanced_graph = test_enhanced_features()

    # 可视化结果
    enhanced_graph.visualize_graph("能耗优化认知图")
    enhanced_graph.visualize_energy_convergence()
    enhanced_graph.visualize_cognitive_states()