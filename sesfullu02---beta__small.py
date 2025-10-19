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
from typing import Dict, Any, List, Tuple
import heapq
from matplotlib.colors import LinearSegmentedColormap
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化jieba
jieba.initialize()


class CognitiveVisualization:
    """认知图可视化工具"""

    def __init__(self):
        self.color_schemes = {
            'physics': 'lightblue',
            'math': 'lightgreen',
            'cs': 'lightcoral',
            'principles': 'gold',
            'other': 'lightgray'
        }

        self.state_colors = {
            CognitiveState.FOCUSED: 'green',
            CognitiveState.EXPLORATORY: 'blue',
            CognitiveState.FATIGUED: 'red',
            CognitiveState.INSPIRED: 'purple'
        }

    def visualize_energy_convergence(self, cognitive_graph, title="认知能耗收敛过程"):
        """可视化能耗收敛过程"""
        plt.figure(figsize=(10, 6))
        plt.plot(cognitive_graph.energy_history, 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('迭代次数')
        plt.ylabel('平均认知能耗')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # 标记概念压缩事件
        colors = ['red', 'green', 'orange', 'purple']
        for i, (center, info) in enumerate(cognitive_graph.concept_centers.items()):
            iteration = info['iteration']
            if iteration < len(cognitive_graph.energy_history):
                color = colors[i % len(colors)]
                plt.axvline(x=iteration, color=color, alpha=0.5, linestyle='--',
                            label=f'压缩: {center}' if i < 4 else "")

        if len(cognitive_graph.concept_centers) > 0:
            plt.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.show()

    def visualize_cognitive_states(self, cognitive_graph, title="认知状态与能耗变化"):
        """可视化认知状态变化"""
        if not cognitive_graph.cognitive_energy_history:
            print("没有认知状态历史数据")
            return

        iterations = [e['iteration'] for e in cognitive_graph.cognitive_energy_history]
        energies = [e['energy'] for e in cognitive_graph.cognitive_energy_history]

        # 确保网络能耗历史长度匹配
        network_energies = cognitive_graph.energy_history[:len(iterations)]
        states = [e['state'] for e in cognitive_graph.cognitive_energy_history]

        colors = [self.state_colors[state] for state in states]

        plt.figure(figsize=(12, 8))

        # 子图1: 主观认知状态与能耗
        plt.subplot(2, 1, 1)
        plt.scatter(iterations, energies, c=colors, alpha=0.6)
        plt.plot(iterations, energies, 'gray', alpha=0.3)
        plt.ylabel('主观认知能耗')
        plt.title(title)

        # 创建图例
        for state, color in self.state_colors.items():
            plt.plot([], [], 'o', color=color, label=state.value)
        plt.legend()

        # 子图2: 网络平均能耗
        plt.subplot(2, 1, 2)
        plt.plot(iterations, network_energies, 'b-', alpha=0.7)
        plt.xlabel('迭代次数')
        plt.ylabel('网络平均能耗')
        plt.title('认知网络能耗演化')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印状态统计
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1

        print("\n=== 认知状态统计 ===")
        for state, count in state_counts.items():
            percentage = (count / len(states)) * 100
            print(f"{state.value}: {count}次 ({percentage:.1f}%)")

    def visualize_cognitive_graph(self, cognitive_graph, title="认知图", figsize=(14, 10)):
        """可视化认知图结构"""
        if cognitive_graph.G.number_of_nodes() == 0:
            print("图中没有节点")
            return

        plt.figure(figsize=figsize)

        # 使用spring布局，但增加k值使节点更分散
        pos = nx.spring_layout(cognitive_graph.G, seed=42, k=2, iterations=50)

        # 准备节点颜色和大小
        node_colors = []
        node_sizes = []
        node_labels = {}

        for i, node in enumerate(cognitive_graph.G.nodes()):
            # 确定节点颜色
            if node in cognitive_graph.concept_centers:
                node_colors.append('red')  # 压缩中心节点
                node_sizes.append(2000)
            elif any('migration_bridges' in cognitive_graph.G.nodes[n]
                     for n in cognitive_graph.G.nodes() if n == node):
                node_colors.append('orange')  # 迁移桥梁节点
                node_sizes.append(1500)
            else:
                # 根据领域着色
                if hasattr(cognitive_graph, 'semantic_network'):
                    domain = cognitive_graph.semantic_network.get_domain(node)
                    node_colors.append(self.color_schemes.get(domain, 'lightgray'))
                else:
                    node_colors.append('lightblue')
                node_sizes.append(800)

            node_labels[node] = node

        # 准备边颜色和宽度
        edge_colors = []
        edge_widths = []

        for u, v in cognitive_graph.G.edges():
            energy = cognitive_graph.G[u][v]['weight']
            edge_widths.append(max(0.5, 4 - energy * 2))  # 能耗越低，边越粗

            # 根据能耗着色
            if energy < 0.5:
                edge_colors.append('green')  # 低能耗
            elif energy < 1.0:
                edge_colors.append('blue')  # 中能耗
            else:
                edge_colors.append('gray')  # 高能耗

        # 绘制节点
        nx.draw_networkx_nodes(cognitive_graph.G, pos,
                               node_size=node_sizes,
                               node_color=node_colors,
                               alpha=0.9,
                               edgecolors='black',
                               linewidths=0.5)

        # 绘制边
        nx.draw_networkx_edges(cognitive_graph.G, pos,
                               width=edge_widths,
                               alpha=0.6,
                               edge_color=edge_colors)

        # 绘制标签
        nx.draw_networkx_labels(cognitive_graph.G, pos,
                                font_size=8,
                                font_family='SimHei',
                                font_weight='bold')

        plt.title(title, fontsize=16, fontfamily='SimHei')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_probabilistic_network(self, semantic_network, top_edges=50):
        """可视化语义网络的重点概率连接"""
        if not hasattr(semantic_network, 'probabilistic_enhancement'):
            print("语义网络没有概率增强数据")
            return

        G = nx.Graph()

        # 添加节点
        for concept in semantic_network.concept_definitions:
            domain = semantic_network.get_domain(concept)
            G.add_node(concept, domain=domain)

        # 添加高互信息的边
        edges_with_mi = []
        for concept1 in semantic_network.probabilistic_enhancement.mutual_info:
            for concept2, mi in semantic_network.probabilistic_enhancement.mutual_info[concept1].items():
                if mi > 0.5:  # 只显示高互信息的连接
                    edges_with_mi.append((concept1, concept2, mi))

        # 按互信息排序，取前top_edges个
        edges_with_mi.sort(key=lambda x: x[2], reverse=True)
        for concept1, concept2, mi in edges_with_mi[:top_edges]:
            G.add_edge(concept1, concept2, weight=mi, mutual_info=mi)

        if G.number_of_nodes() == 0:
            print("没有找到足够的节点连接")
            return

        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=3, iterations=50)

        # 节点颜色
        node_colors = [self.color_schemes[G.nodes[node].get('domain', 'other')]
                       for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=800, alpha=0.9)

        # 边宽度基于互信息
        edge_widths = [G[u][v]['mutual_info'] * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths,
                               alpha=0.6, edge_color='darkblue')

        nx.draw_networkx_labels(G, pos, font_size=8, font_family='SimHei')

        plt.title("基于互信息的语义网络重点连接", fontsize=16, fontfamily='SimHei')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # 打印高互信息连接
        print("\n=== 高互信息概念对 ===")
        for concept1, concept2, mi in edges_with_mi[:10]:
            cond_prob = semantic_network.probabilistic_enhancement.conditional_probs[concept1].get(concept2, 0)
            print(f"{concept1} <-> {concept2}: 互信息={mi:.3f}, 条件概率={cond_prob:.3f}")

    def compare_multiple_simulations(self, simulation_results):
        """比较多个模拟结果"""
        if not simulation_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 能耗收敛比较
        for i, result in enumerate(simulation_results):
            axes[0, 0].plot(result['energy_history'],
                            label=result['individual_id'],
                            alpha=0.7)
        axes[0, 0].set_title('能耗收敛过程比较')
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('平均认知能耗')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 最终能耗比较
        final_energies = [result['final_stats']['avg_energy']
                          for result in simulation_results]
        individual_ids = [result['individual_id']
                          for result in simulation_results]
        axes[0, 1].bar(individual_ids, final_energies, alpha=0.7)
        axes[0, 1].set_title('最终平均能耗比较')
        axes[0, 1].set_ylabel('平均认知能耗')

        # 压缩中心数量比较
        compression_centers = [result['final_stats']['compression_centers']
                               for result in simulation_results]
        axes[1, 0].bar(individual_ids, compression_centers, alpha=0.7, color='orange')
        axes[1, 0].set_title('概念压缩中心数量比较')
        axes[1, 0].set_ylabel('压缩中心数量')

        # 迁移桥梁数量比较
        migration_bridges = [result['final_stats']['migration_bridges']
                             for result in simulation_results]
        axes[1, 1].bar(individual_ids, migration_bridges, alpha=0.7, color='green')
        axes[1, 1].set_title('迁移桥梁数量比较')
        axes[1, 1].set_ylabel('迁移桥梁数量')

        plt.tight_layout()
        plt.show()

        # 打印统计摘要
        print("\n=== 多模拟比较统计 ===")
        print(f"平均最终能耗: {np.mean(final_energies):.3f} ± {np.std(final_energies):.3f}")
        print(f"平均压缩中心: {np.mean(compression_centers):.1f} ± {np.std(compression_centers):.1f}")
        print(f"平均迁移桥梁: {np.mean(migration_bridges):.1f} ± {np.std(migration_bridges):.1f}")


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


class ProbabilisticGraphEnhancement:
    """概率图增强模块"""

    def __init__(self):
        self.conditional_probs = defaultdict(dict)  # P(B|A)
        self.mutual_info = defaultdict(dict)  # I(A;B)
        self.co_occurrence = defaultdict(lambda: defaultdict(int))
        self.total_pairs = 0
        self.concept_frequencies = defaultdict(int)

    def update_co_occurrence(self, concept1, concept2, count=1):
        """更新概念共现统计"""
        self.co_occurrence[concept1][concept2] += count
        self.co_occurrence[concept2][concept1] += count
        self.concept_frequencies[concept1] += count
        self.concept_frequencies[concept2] += count
        self.total_pairs += count

    def compute_conditional_probability(self, concept1, concept2):
        """计算条件概率 P(concept2|concept1)"""
        if concept1 not in self.co_occurrence or concept2 not in self.co_occurrence[concept1]:
            return 0.1  # 默认概率

        count_concept1_concept2 = self.co_occurrence[concept1][concept2]
        total_concept1 = sum(self.co_occurrence[concept1].values())

        return count_concept1_concept2 / max(total_concept1, 1)

    def compute_mutual_information(self, concept1, concept2):
        """计算互信息 I(concept1; concept2)"""
        if (concept1 not in self.concept_frequencies or
                concept2 not in self.concept_frequencies or
                self.total_pairs == 0):
            return 0.0

        # 联合概率 P(concept1, concept2)
        P_AB = self.co_occurrence[concept1][concept2] / self.total_pairs

        # 边缘概率
        P_A = self.concept_frequencies[concept1] / (self.total_pairs * 2)  # 因为每条边被计数两次
        P_B = self.concept_frequencies[concept2] / (self.total_pairs * 2)

        if P_AB > 0 and P_A > 0 and P_B > 0:
            return math.log2(P_AB / (P_A * P_B))
        return 0.0

    def find_high_mi_clusters(self, threshold=1.0):
        """找出高互信息的节点簇 - 修复版本"""
        clusters = []
        processed = set()

        # 只考虑概念节点，而不是所有共现词汇
        concepts = list(self.concept_frequencies.keys())  # 所有出现过的概念

        for concept1 in concepts:
            if concept1 in processed:
                continue

            cluster = [concept1]
            for concept2 in concepts:
                if (concept2 not in processed and
                        concept2 != concept1 and
                        concept1 in self.mutual_info and  # 确保concept1在mutual_info中
                        concept2 in self.mutual_info[concept1] and  # 确保concept2在concept1的互信息中
                        self.mutual_info[concept1][concept2] > threshold):
                    cluster.append(concept2)

            if len(cluster) > 1:
                clusters.append(cluster)
                processed.update(cluster)

        return clusters

class SemanticConceptNetwork:
    """基于定义关键词的语义概念网络"""

    def __init__(self):
        self.concept_definitions = {}
        self.concept_keywords = {}
        self.semantic_network = defaultdict(dict)
        self.probabilistic_enhancement = ProbabilisticGraphEnhancement()
        self.meta_structure = MetaStructureSimilarity()  # 新增元结构
    def add_concept_definition(self, concept, definition, source="manual"):
        """添加概念定义"""
        self.concept_definitions[concept] = {
            'definition': definition,
            'source': source,
            'timestamp': time.time()
        }

        keywords = self.extract_keywords(definition)
        self.concept_keywords[concept] = keywords

        # 更新共现统计（概念与其关键词）
        for keyword in keywords:
            self.probabilistic_enhancement.update_co_occurrence(concept, keyword)

        print(f"添加概念 '{concept}': {definition}")
        print(f"  提取关键词: {keywords}")

    def extract_keywords(self, text, top_k=10):
        """从文本中提取关键词 - 改进版本"""
        words = jieba.cut(text)

        stop_words = {
            '的', '是', '在', '和', '与', '或', '等', '这个', '那个', '一种',
            '研究', '包括', '通过', '给定', '任何', '两个', '某种', '一个'
        }

        filtered_words = [
            word for word in words
            if len(word) > 1 and word not in stop_words
        ]

        word_freq = defaultdict(int)
        for word in filtered_words:
            word_freq[word] += 1

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

                # 更新概率信息
                self.probabilistic_enhancement.update_co_occurrence(concept, keyword)

                if current_depth < max_depth - 1:
                    self.expand_concept_network(keyword, max_depth, current_depth + 1)

    def calculate_semantic_similarity(self, concept1, concept2):
        """计算两个概念的语义相似度 - 集成元结构版本"""
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
        meta_similarity = self.meta_structure.calculate_meta_similarity(concept1, concept2)  # 新增元结构相似度

        # 综合三种相似度
        combined_similarity = (0.5 * jaccard_similarity +
                              0.3 * domain_similarity +
                              0.2 * meta_similarity)

        return min(combined_similarity, 1.0)

    def _calculate_domain_similarity(self, concept1, concept2):
        """计算领域相似度 - 增强版本"""
        domain1 = self.get_domain(concept1)
        domain2 = self.get_domain(concept2)

        if domain1 == domain2:
            return 1.0
        elif (domain1 == "principles" or domain2 == "principles"):
            return 0.6  # 原理节点与各领域都有一定相似性
        elif (domain1.startswith("math") and domain2.startswith("math")):
            return 0.7  # 数学内部子领域高度相似
        elif (domain1.startswith("cs") and domain2.startswith("cs")):
            return 0.7  # 计算机科学内部子领域高度相似
        elif (domain1.startswith("biology") and domain2.startswith("biology")):
            return 0.7  # 生物学内部子领域高度相似
        else:
            return 0.2  # 不同领域的基础相似度

    def build_comprehensive_network(self):
        """构建综合概念网络 - 概率增强版本"""
        self._predefine_core_concepts()

        all_concepts = list(self.concept_definitions.keys())
        print(f"开始构建语义网络，共有 {len(all_concepts)} 个概念")

        # 首先建立所有概念之间的直接连接
        for i, concept1 in enumerate(all_concepts):
            for j, concept2 in enumerate(all_concepts[i + 1:], i + 1):
                similarity = self.calculate_semantic_similarity(concept1, concept2)
                if similarity > 0.1:
                    self.semantic_network[concept1][concept2] = similarity
                    self.semantic_network[concept2][concept1] = similarity

                    # 更新概率信息
                    count = int(similarity * 10)  # 基于相似度的共现次数
                    self.probabilistic_enhancement.update_co_occurrence(concept1, concept2, count)

        # 计算所有概率信息
        self._compute_all_probabilistic_measures()

        # 然后进行深度扩展
        for concept in all_concepts:
            self.expand_concept_network(concept, max_depth=3)

        print(f"语义网络构建完成! 包含 {len(self.semantic_network)} 个概念节点")
        print(f"网络连接数: {sum(len(neighbors) for neighbors in self.semantic_network.values()) // 2}")

    def _compute_all_probabilistic_measures(self):
        """计算所有概念对的概率度量"""
        concepts = list(self.concept_definitions.keys())
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i + 1:]:
                # 计算条件概率
                cond_prob = self.probabilistic_enhancement.compute_conditional_probability(concept1, concept2)
                self.probabilistic_enhancement.conditional_probs[concept1][concept2] = cond_prob
                self.probabilistic_enhancement.conditional_probs[concept2][concept1] = cond_prob

                # 计算互信息
                mi = self.probabilistic_enhancement.compute_mutual_information(concept1, concept2)
                self.probabilistic_enhancement.mutual_info[concept1][concept2] = mi
                self.probabilistic_enhancement.mutual_info[concept2][concept1] = mi

    def _predefine_core_concepts(self):
        """预定义核心概念及其关系 - 完整版本"""
        core_definitions = {
            # 物理概念
            "牛顿定律": "物体运动的基本定律，描述了力与运动的关系",
            "力学": "研究物体运动和受力情况的物理学分支",
            "运动": "物体位置随时间的变化过程",  # 新增运动概念
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

            # 计算机概念
            "算法": "解决问题的一系列明确的计算步骤",
            "数据结构": "计算机中组织和存储数据的方式",
            "机器学习": "让计算机通过经验自动改进性能的人工智能分支",
            "神经网络": "模仿生物神经网络的计算模型",
            "计算机视觉": "让计算机理解和分析视觉信息的技术",
            "自然语言处理": "计算机与人类自然语言交互的技术",
            "数据库": "结构化信息或数据的有组织集合",
            "操作系统": "管理计算机硬件与软件资源的系统软件",

            # 原理概念
            "优化": "在给定约束下找到最佳解决方案的过程",
            "变换": "从一个形式或状态转换为另一个的过程",
            "迭代": "重复反馈过程的活动",
            "抽象": "提取主要特征忽略次要细节的思维过程",
            "模式识别": "通过算法识别数据中模式的过程",
            "对称": "物体在某种变换下保持不变的性质",
            "递归": "通过函数调用自身来解决问题的方法",
            "归纳": "从特殊到一般的推理方法",

            # === 具体概念 ===
            # 水果类
            "苹果": "一种圆形水果，通常为红色或绿色，味道甜美",
            "香蕉": "一种长形水果，黄色外皮，味道香甜",
            "橙子": "一种圆形水果，橙色外皮，多汁酸甜",
            "葡萄": "一种小型水果，成串生长，味道甜美",

            # 编程语言类
            "C语言": "一种通用的高级编程语言，广泛用于系统编程",
            "Python": "一种解释型高级编程语言，以简洁易读著称",
            "Java": "一种面向对象的编程语言，跨平台运行",
            "JavaScript": "一种脚本语言，主要用于网页开发",

            # 算术运算类
            "加法": "将两个或多个数值合并为一个总和的数学运算",
            "减法": "从一个数值中减去另一个数值的数学运算",
            "乘法": "将相同数值重复相加的快捷数学运算",
            "除法": "将一个数值分成若干等份的数学运算",

            # 颜色类
            "红色": "一种基本颜色，波长约620-750纳米",
            "蓝色": "一种基本颜色，波长约450-495纳米",
            "绿色": "一种基本颜色，波长约495-570纳米",
            "黄色": "一种基本颜色，波长约570-590纳米",

            # 动物类
            "猫": "一种小型食肉哺乳动物，常见宠物",
            "狗": "一种犬科哺乳动物，人类最早驯化的动物",
            "鸟": "有羽毛、卵生、前肢成翼的脊椎动物",
            "鱼": "生活在水中的冷血脊椎动物，用鳃呼吸"
        }

        for concept, definition in core_definitions.items():
            self.add_concept_definition(concept, definition, "predefined")


    def get_domain(self, concept):
        """获取概念所属领域 - 完整版本"""
        domains = {
            # 物理学分支
            "physics": ["牛顿定律", "力学", "运动学", "能量守恒", "动量", "万有引力", "摩擦力", "静电力", "运动"],

            # 数学分支
            "math_advanced": ["微积分", "几何学", "拓扑学", "线性代数", "概率论", "统计学", "代数", "离散数学"],
            "math_basic": ["加法", "减法", "乘法", "除法"],

            # 计算机科学分支
            "cs_theory": ["算法", "数据结构"],
            "cs_ai": ["机器学习", "神经网络", "计算机视觉", "自然语言处理"],
            "cs_systems": ["数据库", "操作系统"],
            "cs_languages": ["C语言", "Python", "Java", "JavaScript"],

            # 原理和思维方法
            "principles": ["优化", "变换", "迭代", "抽象", "模式识别", "对称", "递归", "归纳"],

            # 生物学相关
            "biology_plants": ["苹果", "香蕉", "橙子", "葡萄"],
            "biology_animals": ["猫", "狗", "鸟", "鱼"],

            # 常识概念
            "common_colors": ["红色", "蓝色", "绿色", "黄色"],

            # === 新增：元结构概念 ===
            "meta_structure": ["信息", "历史", "运动", "几何", "能量", "方法", "系统", "抽象","物体"]
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
                "math_advanced": "lightgreen",
                "math_basic": "palegreen",  # 浅绿色表示基础数学
                "cs_theory": "lightcoral",
                "cs_ai": "coral",  # 更深的珊瑚色表示AI
                "cs_systems": "pink",  # 粉色表示系统
                "cs_languages": "hotpink",  # 亮粉色表示编程语言
                "principles": "gold",
                "biology_plants": "lightgreen",  # 植物用绿色
                "biology_animals": "bisque",  # 动物用米色
                "common_colors": "lightyellow",  # 颜色用浅黄色
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
    """调整参数后的主观认知图 - 概率增强版本"""

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

        # 🔴 优化状态转移概率 - 更符合认知规律
        self.state_transition_matrix = {
            CognitiveState.FOCUSED: {
                CognitiveState.EXPLORATORY: 0.4,  # 增加探索倾向
                CognitiveState.FATIGUED: 0.15,  # 适度疲劳
                CognitiveState.INSPIRED: 0.25,  # 增加灵感机会
                CognitiveState.FOCUSED: 0.2
            },
            CognitiveState.EXPLORATORY: {
                CognitiveState.FOCUSED: 0.35,  # 更容易进入专注
                CognitiveState.FATIGUED: 0.15,  # 适度疲劳
                CognitiveState.INSPIRED: 0.3,  # 探索容易产生灵感
                CognitiveState.EXPLORATORY: 0.2
            },
            CognitiveState.FATIGUED: {
                CognitiveState.FOCUSED: 0.25,  # 适度恢复
                CognitiveState.EXPLORATORY: 0.45,  # 疲劳时倾向于探索
                CognitiveState.INSPIRED: 0.1,  # 疲劳时不易产生灵感
                CognitiveState.FATIGUED: 0.2
            },
            CognitiveState.INSPIRED: {
                CognitiveState.FOCUSED: 0.5,  # 灵感后容易专注
                CognitiveState.EXPLORATORY: 0.3,  # 保持探索
                CognitiveState.FATIGUED: 0.05,  # 灵感时不易疲劳
                CognitiveState.INSPIRED: 0.15
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

        # 概率增强
        self.use_probabilistic_enhancement = True
        self.exploration_factor = 0.3  # 探索因子

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
        """检查是否可以遍历某条边（考虑主观认知能耗）- 优化版本"""
        # 🔴 新增：基于认知状态的能耗调节
        state_efficiency = {
            CognitiveState.FOCUSED: 1.2,  # 专注时效率提高20%
            CognitiveState.EXPLORATORY: 1.0,  # 正常
            CognitiveState.FATIGUED: 0.7,  # 疲劳时效率降低30%
            CognitiveState.INSPIRED: 1.4  # 灵感时效率提高40%
        }

        efficiency = state_efficiency[self.current_state]

        if traversal_type == "hard":
            required_energy = edge_energy * 0.7 / efficiency  # 考虑状态效率
            available_energy = self.subjective_energy * self.hard_traversal_energy_ratio
        else:
            required_energy = edge_energy * 0.5 / efficiency  # 考虑状态效率
            available_energy = self.subjective_energy * self.soft_traversal_energy_ratio

        # 🔴 新增：个体差异调节
        individual_efficiency = self.individual_params.get('cognitive_efficiency', 1.0)
        required_energy /= individual_efficiency

        energy_balance = available_energy - required_energy
        can_traverse = energy_balance >= -0.3  # 允许轻微透支

        return can_traverse, energy_balance
    def traverse_path(self, path, traversal_type="hard"):
        """改进的遍历函数 - 考虑主观认知状态和概率信息"""
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

                # 概率增强的学习率
                similarity = 0.5
                if hasattr(self, 'semantic_network'):
                    similarity = self.calculate_semantic_similarity(u, v)

                base_rate = self.base_learning_rate
                individual_learning_variation = np.random.uniform(
                    1 - self.learning_rate_variation,
                    1 + self.learning_rate_variation
                )

                # 概率增强：考虑条件概率
                prob_factor = 1.0
                if (self.use_probabilistic_enhancement and
                        hasattr(self, 'semantic_network') and
                        u in self.semantic_network.probabilistic_enhancement.conditional_probs and
                        v in self.semantic_network.probabilistic_enhancement.conditional_probs[u]):
                    cond_prob = self.semantic_network.probabilistic_enhancement.conditional_probs[u][v]
                    prob_factor = 0.8 + 0.4 * cond_prob  # 条件概率越高，学习效果越好

                if traversal_type == "hard":
                    learning_rate = base_rate * (0.8 + 0.4 * similarity) * individual_learning_variation * prob_factor
                else:
                    learning_rate = base_rate * (
                                0.7 + 0.3 * similarity) * individual_learning_variation * prob_factor  # 提高软遍历学习率

                # 🔴 新增：基于认知状态的学习率调节
                state_learning_boost = {
                    CognitiveState.FOCUSED: 1.2,
                    CognitiveState.EXPLORATORY: 1.0,
                    CognitiveState.FATIGUED: 0.8,
                    CognitiveState.INSPIRED: 1.3
                }
                learning_rate *= state_learning_boost[self.current_state]

                current_weight = self.G[u][v]['weight']
                learning_effect = learning_rate * (current_weight / 1.5)  # 增强学习效果

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

    def probabilistic_soft_traversal(self, start_node, max_length=3):
        """基于概率的智能软遍历"""
        path = [start_node]
        current_node = start_node

        for step in range(max_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break

            # 计算每个邻居的吸引力分数
            scores = []
            for neighbor in neighbors:
                if neighbor in path:  # 避免循环
                    scores.append(0)
                    continue

                # 条件概率成分
                cond_prob = 0.1
                if (hasattr(self, 'semantic_network') and
                        current_node in self.semantic_network.probabilistic_enhancement.conditional_probs and
                        neighbor in self.semantic_network.probabilistic_enhancement.conditional_probs[current_node]):
                    cond_prob = self.semantic_network.probabilistic_enhancement.conditional_probs[current_node][
                        neighbor]

                # 互信息成分
                mutual_info = 0.0
                if (hasattr(self, 'semantic_network') and
                        current_node in self.semantic_network.probabilistic_enhancement.mutual_info and
                        neighbor in self.semantic_network.probabilistic_enhancement.mutual_info[current_node]):
                    mutual_info = self.semantic_network.probabilistic_enhancement.mutual_info[current_node][neighbor]
                normalized_mi = max(0, min(1, (mutual_info + 2) / 4))  # 归一化

                # 能耗成分
                energy_cost = self.G[current_node][neighbor]['weight']
                energy_factor = 1 - energy_cost / 2.0  # 归一化

                # 综合得分
                score = (0.4 * cond_prob +
                         0.3 * normalized_mi +
                         0.3 * energy_factor)
                scores.append(score)

            if max(scores) == 0:  # 所有邻居都不合适
                break

            # 添加探索随机性
            if random.random() < self.exploration_factor:
                next_node = random.choice(neighbors)
            else:
                # 基于概率选择
                next_node = self._weighted_choice(neighbors, scores)

            path.append(next_node)
            current_node = next_node

        return path if len(path) >= 2 else None

    def _weighted_choice(self, options, weights):
        """加权随机选择"""
        total = sum(weights)
        if total == 0:
            return random.choice(options)

        r = random.uniform(0, total)
        current = 0
        for i, weight in enumerate(weights):
            current += weight
            if r <= current:
                return options[i]
        return options[-1]

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

    def monte_carlo_iteration(self, max_iterations=10000):
        """改进的蒙特卡洛模拟 - 增强压缩调试版本"""

        print(f"初始认知状态: {self.current_state.value}, 主观能耗: {self.subjective_energy:.2f}")

        compression_attempts = 0
        successful_compressions = 0

        for iteration in range(max_iterations):
            self.iteration_count += 1

            if iteration % 100 == 0:
                self.update_cognitive_state()

            self._apply_forgetting()

            current_avg_energy = self.get_average_energy()
            self.energy_history.append(current_avg_energy)

            operation = self._select_operation_based_on_state()

            if operation == "compression":
                compression_attempts += 1
                old_centers = len(self.concept_centers)
                self._probabilistic_compression()
                if len(self.concept_centers) > old_centers:
                    successful_compressions += 1
            elif operation == "hard_traversal":
                self._state_based_hard_traversal()
            elif operation == "soft_traversal":
                self._state_based_soft_traversal()
            elif operation == "migration":
                self._probabilistic_migration()

            if iteration % 500 == 0:
                stats = self.get_network_stats()
                print(f"迭代 {iteration}, 状态: {self.current_state.value}, "
                      f"主观能耗: {self.subjective_energy:.2f}, 网络能耗: {current_avg_energy:.3f}")

                # 每500次迭代显示压缩进度
                if compression_attempts > 0:
                    success_rate = successful_compressions / compression_attempts * 100
                    print(f"  压缩尝试: {compression_attempts}, 成功: {successful_compressions} ({success_rate:.1f}%)")

        # 最终压缩统计
        print(f"\n📊 压缩统计: 总尝试{compression_attempts}次, 成功{successful_compressions}次")
        if iteration % 500 == 0:
            stats = self.get_network_stats()
            # 🔴 新增压缩调试信息
            compression_info = ""
            if hasattr(self, 'semantic_network'):
                # 检查互信息统计
                mi_values = []
                for concept1 in list(self.G.nodes())[:5]:  # 采样前5个概念
                    if concept1 in self.semantic_network.probabilistic_enhancement.mutual_info:
                        for concept2, mi in list(
                                self.semantic_network.probabilistic_enhancement.mutual_info[concept1].items())[:3]:
                            if concept2 in self.G:
                                mi_values.append(mi)

                if mi_values:
                    avg_mi = sum(mi_values) / len(mi_values)
                    compression_info = f", 平均互信息: {avg_mi:.3f}"

            print(f"迭代 {iteration}, 状态: {self.current_state.value}, "
                  f"主观能耗: {self.subjective_energy:.2f}, 网络能耗: {current_avg_energy:.3f}"
                  f"{compression_info}")

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
        """基于状态的软遍历 - 概率增强版本"""
        available_nodes = list(self.G.nodes())
        if not available_nodes:
            return

        start_node = random.choice(available_nodes)

        if self.use_probabilistic_enhancement:
            path = self.probabilistic_soft_traversal(start_node, 3)
        else:
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

    def _probabilistic_compression(self):
        """基于概率的概念压缩 - 修复版本"""
        try:
            if not hasattr(self, 'semantic_network'):
                self._random_compression()
                return

            available_nodes = list(self.G.nodes())
            if len(available_nodes) < 3:
                return

            # 🔴 问题：压缩频率过低 (5%)，改为15-20%
            if random.random() > 0.15:  # 从0.05改为0.15
                return

            # 🔴 问题：压缩条件过于严格，降低要求
            if self.iteration_count < 500:  # 从1000改为500
                return

            # 只考虑认知图中实际存在的概念
            available_concepts = [node for node in available_nodes
                                  if node in self.semantic_network.concept_definitions]

            if len(available_concepts) < 3:
                return

            # 🔴 问题：互信息阈值过高 (0.3)，降低到0.1
            high_mi_clusters = []
            processed = set()

            for concept1 in available_concepts:
                if concept1 in processed:
                    continue

                cluster = [concept1]
                for concept2 in available_concepts:
                    if (concept2 not in processed and
                            concept2 != concept1 and
                            concept1 in self.semantic_network.probabilistic_enhancement.mutual_info and
                            concept2 in self.semantic_network.probabilistic_enhancement.mutual_info[concept1] and
                            self.semantic_network.probabilistic_enhancement.mutual_info[concept1][
                                concept2] > 0.1):  # 从0.3改为0.1
                        cluster.append(concept2)

                if len(cluster) > 1:
                    high_mi_clusters.append(cluster)
                    processed.update(cluster)

            # 🔴 新增：如果没有找到高互信息簇，尝试基于语义相似度的压缩
            if not high_mi_clusters:
                self._fallback_semantic_compression(available_concepts)
                return

            for cluster in high_mi_clusters:
                if len(cluster) >= 2 and len(cluster) <= 5:
                    # 检查这些概念是否属于同一语义类别
                    if self._concepts_same_category(cluster):
                        # 选择中心节点 - 改进选择策略
                        center_candidate = self._select_compression_center(cluster)

                        if center_candidate:
                            related_nodes = [node for node in cluster if node != center_candidate]

                            # 🔴 增强压缩效果
                            compression_strength = random.uniform(0.3, 0.7)  # 扩大范围
                            success = self.conceptual_compression(center_candidate, related_nodes, compression_strength)

                            if success:
                                print(f"🎯 概念压缩成功: {center_candidate} <- {related_nodes}")
                                return  # 成功一次就返回

        except Exception as e:
            print(f"压缩过程中出错: {e}")
            import traceback
            traceback.print_exc()
    def _concepts_same_category(self, concepts):
        """判断一组概念是否属于同一语义类别 - 修复版本"""
        # 确保所有概念都在认知图中
        valid_concepts = [concept for concept in concepts if concept in self.G]
        if len(valid_concepts) < 2:
            return False

        # 简单的类别判断逻辑
        categories = {
            "水果": ["苹果", "香蕉", "橙子", "葡萄"],
            "编程语言": ["C语言", "Python", "Java", "JavaScript"],
            "算术运算": ["加法", "减法", "乘法", "除法"],
            "颜色": ["红色", "蓝色", "绿色", "黄色"],
            "动物": ["猫", "狗", "鸟", "鱼"],
            # === 新增元结构类别 ===
            "物理概念": ["牛顿定律", "力学", "运动学", "能量守恒", "动量", "万有引力", "摩擦力", "静电力"],
            "数学概念": ["微积分", "几何学", "拓扑学", "线性代数", "概率论", "统计学", "代数", "离散数学"],
            "计算机概念": ["算法", "数据结构", "机器学习", "神经网络", "计算机视觉", "自然语言处理", "数据库",
                           "操作系统"]
        }

        # 找出所有概念所属的类别
        concept_categories = []
        for concept in valid_concepts:
            category_found = False
            for category, members in categories.items():
                if concept in members:
                    concept_categories.append(category)
                    category_found = True
                    break
            if not category_found:
                concept_categories.append("其他")

        # 如果所有概念都属于同一类别，返回True
        if len(set(concept_categories)) == 1 and concept_categories[0] != "其他":
            return True

        return False

    def _probabilistic_migration(self):
        """基于概率的第一性原理迁移"""
        if not hasattr(self, 'semantic_network'):
            self._random_migration()
            return

        available_nodes = list(self.G.nodes())
        if len(available_nodes) < 4:
            return

        if random.random() > 0.05:
            return

        start_node, end_node = random.sample(available_nodes, 2)

        # 寻找高互信息的中介节点
        candidate_bridges = []
        for mediator in available_nodes:
            if mediator not in [start_node, end_node]:
                mi_start_med = self.semantic_network.probabilistic_enhancement.mutual_info[start_node].get(mediator, 0)
                mi_med_end = self.semantic_network.probabilistic_enhancement.mutual_info[mediator].get(end_node, 0)

                direct_mi = self.semantic_network.probabilistic_enhancement.mutual_info[start_node].get(end_node, 0)

                # 如果中介路径的信息流强于直接路径
                if min(mi_start_med, mi_med_end) > direct_mi:
                    candidate_bridges.append((mediator, mi_start_med + mi_med_end))

        if candidate_bridges:
            # 选择信息流最强的中介
            candidate_bridges.sort(key=lambda x: x[1], reverse=True)
            selected_mediator = candidate_bridges[0][0]

            exploration_bonus = random.uniform(0.05, 0.15)
            self.first_principles_migration(start_node, end_node, [selected_mediator], exploration_bonus)

    def conceptual_compression(self, center_node, related_nodes, compression_strength=0.5):
        """概念压缩：强化中心节点与相关节点的连接 - 增强版本"""
        if len(related_nodes) < 2:
            return False

        # 记录压缩前的能耗
        pre_compression_energy = 0
        for node in related_nodes:
            if self.G.has_edge(center_node, node):
                pre_compression_energy += self.G[center_node][node]['weight']

        # 执行压缩
        for node in related_nodes:
            if self.G.has_edge(center_node, node):
                current_energy = self.G[center_node][node]['weight']
                # 更强的压缩效果
                compressed_energy = max(0.05, current_energy * compression_strength * 0.8)  # 额外降低20%
                self.G[center_node][node]['weight'] = compressed_energy

        # 计算能耗节省
        post_compression_energy = 0
        for node in related_nodes:
            if self.G.has_edge(center_node, node):
                post_compression_energy += self.G[center_node][node]['weight']

        energy_saving = pre_compression_energy - post_compression_energy

        self.concept_centers[center_node] = {
            'related_nodes': related_nodes,
            'compression_strength': compression_strength,
            'iteration': self.iteration_count,
            'energy_saving': energy_saving
        }

        print(f"💡 压缩效果: 节省能耗 {energy_saving:.3f}")
        return True

    def first_principles_migration(self, start_node, end_node, principle_nodes, exploration_bonus=0.1):
        """第一性原理迁移"""
        best_path = None
        best_energy = float('inf')

        direct_energy = float('inf')
        if self.G.has_edge(start_node, end_node):
            direct_energy = self.G[start_node][end_node]['weight']
            best_path = [start_node, end_node]
            best_energy = direct_energy

        for principle in principle_nodes:
            if (self.G.has_edge(start_node, principle) and
                    self.G.has_edge(principle, end_node)):

                path_energy = (self.G[start_node][principle]['weight'] +
                               self.G[principle][end_node]['weight'])

                adjusted_energy = path_energy - exploration_bonus

                if adjusted_energy < best_energy:
                    best_energy = adjusted_energy
                    best_path = [start_node, principle, end_node]

        improvement_threshold = 0.2
        if (best_path and len(best_path) > 2 and
                best_energy < direct_energy * (1 - improvement_threshold)):

            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                current = self.G[u][v]['weight']
                new_energy = max(0.05, current * random.uniform(0.6, 0.8))
                self.G[u][v]['weight'] = new_energy

            if 'migration_bridges' not in self.G.nodes[principle]:
                self.G.nodes[principle]['migration_bridges'] = []

            self.G.nodes[principle]['migration_bridges'].append({
                'from': start_node,
                'to': end_node,
                'energy_saving': direct_energy - best_energy,
                'iteration': self.iteration_count
            })

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

    def _fallback_semantic_compression(self, available_concepts):
        """基于语义相似度的备选压缩机制"""
        # 尝试基于简单类别进行压缩
        categories = self._group_by_category(available_concepts)

        for category, concepts in categories.items():
            if len(concepts) >= 3:
                # 选择连接度最高的节点作为中心
                center_candidate = max(concepts,
                                       key=lambda node: self.G.degree(node))
                related_nodes = [node for node in concepts
                                 if node != center_candidate and
                                 self.G.has_edge(center_candidate, node)]

                if len(related_nodes) >= 2:
                    compression_strength = random.uniform(0.4, 0.6)
                    success = self.conceptual_compression(center_candidate,
                                                          related_nodes[:3],  # 限制数量
                                                          compression_strength)
                    if success:
                        print(f"🔄 备选压缩成功: {center_candidate} <- {related_nodes[:3]}")
                        return True
        return False

    def _select_compression_center(self, cluster):
        """改进的压缩中心选择策略"""
        # 选择在认知图中连接度最高的节点
        valid_nodes = [node for node in cluster if node in self.G]
        if not valid_nodes:
            return None

        return max(valid_nodes, key=lambda node: self.G.degree(node))

    def _group_by_category(self, concepts):
        """按语义类别分组概念"""
        categories = defaultdict(list)
        for concept in concepts:
            category = self._get_broad_category(concept)
            categories[category].append(concept)
        return categories

    def _get_broad_category(self, concept):
        """获取概念的宽泛类别"""
        if concept in ["苹果", "香蕉", "橙子", "葡萄"]:
            return "水果"
        elif concept in ["猫", "狗", "鸟", "鱼"]:
            return "动物"
        elif concept in ["红色", "蓝色", "绿色", "黄色"]:
            return "颜色"
        elif concept in ["加法", "减法", "乘法", "除法"]:
            return "算术"
        elif concept in ["C语言", "Python", "Java", "JavaScript"]:
            return "编程语言"
        elif concept in ["牛顿定律", "力学", "运动学", "能量守恒", "动量"]:
            return "物理"
        elif concept in ["微积分", "几何学", "线性代数", "概率论"]:
            return "数学"
        elif concept in ["算法", "数据结构", "机器学习", "神经网络"]:
            return "计算机科学"
        else:
            return "其他"

class MetaStructureSimilarity:
    """元结构相似度计算 - 实现你的想法"""

    def __init__(self):
        self.meta_structures = {
            "信息": ["数据", "知识", "信号", "消息", "情报", "数据库", "自然语言处理"],
            "历史": ["迭代", "回归", "演化", "发展", "进程", "时间", "递归"],
            "运动": ["遍历", "变化", "过程", "流动", "迁移", "转换", "力学", "运动学"],
            "几何": ["结构", "形状", "关系", "形式", "布局", "拓扑", "几何学", "对称"],
            "能量": ["能耗", "功率", "动力", "资源", "消耗", "能量守恒", "动量"],
            "方法": ["算法", "策略", "技术", "途径", "手段", "方法论", "优化", "模式识别"],
            "系统": ["网络", "集合", "整体", "组织", "架构", "操作系统", "神经网络"],
            "抽象": ["概念", "思想", "理论", "原理", "范式", "抽象", "归纳"],
            "物体":["水果","动物","实体","颜色","性质","物理","物质"]
        }

        # 创建反向映射
        self.concept_to_meta = {}
        for meta, concepts in self.meta_structures.items():
            for concept in concepts:
                self.concept_to_meta[concept] = meta

    def get_meta_structure(self, concept):
        """获取概念的元结构"""
        return self.concept_to_meta.get(concept, "其他")

    def calculate_meta_similarity(self, concept1, concept2):
        """计算基于元结构的相似度"""
        meta1 = self.get_meta_structure(concept1)
        meta2 = self.get_meta_structure(concept2)

        if meta1 == meta2 and meta1 != "其他":
            return 0.8  # 相同元结构高度相似
        elif meta1 != "其他" and meta2 != "其他":
            return 0.3  # 不同元结构但有元结构
        else:
            return 0.1  # 至少一个没有元结构

# 其他类（SemanticEnhancedCognitiveGraph, MetaStructureSimilarity, EnhancedSemanticConceptNetwork,
# EnergyOptimizedCognitiveGraph）和辅助函数保持类似结构，但集成概率增强

class SemanticEnhancedCognitiveGraph(AdjustedSubjectiveCognitiveGraph):
    """基于语义增强的认知图 - 概率增强版本"""

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        super().__init__(individual_params, network_seed)

        self.semantic_network = SemanticConceptNetwork()
        self.semantic_network.build_comprehensive_network()

    def calculate_semantic_similarity(self, node1, node2):
        """基于语义网络计算相似度"""
        return self.semantic_network.calculate_semantic_similarity(node1, node2)

    def initialize_semantic_graph(self):
        """基于语义相似度初始化认知图 - 概率增强版本"""
        nodes = list(self.semantic_network.concept_definitions.keys())
        self.G.add_nodes_from(nodes)

        initial_edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1:], i + 1):
                similarity = self.calculate_semantic_similarity(node1, node2)

                if similarity > 0.1:
                    # 概率增强的能耗计算
                    base_energy = 2.0

                    # 语义相似度成分
                    semantic_factor = 1 - similarity

                    # 互信息成分
                    mi = self.semantic_network.probabilistic_enhancement.mutual_info[node1].get(node2, 0)
                    normalized_mi = max(0, min(1, (mi + 2) / 4))
                    mi_factor = 1 - normalized_mi * 0.5

                    # 条件概率成分
                    cond_prob = self.semantic_network.probabilistic_enhancement.conditional_probs[node1].get(node2, 0.1)
                    prob_factor = 1 - cond_prob * 0.3

                    # 综合能耗
                    energy = base_energy * semantic_factor * mi_factor * prob_factor
                    energy = max(0.3, min(2.0, energy))

                    initial_edges.append((node1, node2, energy))

        for edge in initial_edges:
            u, v, weight = edge
            self.G.add_edge(u, v, weight=weight, traversal_count=0, original_weight=weight)
            self.last_activation_time[(u, v)] = 0

        print(f"基于语义初始化完成: {len(nodes)}个节点, {len(initial_edges)}条边")


# 简化的演示函数
def demo_probabilistic_enhancement():
    """演示概率增强功能"""
    print("=== 概率增强认知图演示 ===")

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

    # 创建语义增强的认知图
    cognitive_graph = SemanticEnhancedCognitiveGraph(base_params)
    cognitive_graph.initialize_semantic_graph()

    # 显示一些概率信息
    concepts = list(cognitive_graph.G.nodes())[:5]
    print("\n=== 概念概率关系示例 ===")
    for i, concept1 in enumerate(concepts):
        for concept2 in concepts[i + 1:]:
            if cognitive_graph.G.has_edge(concept1, concept2):
                cond_prob = cognitive_graph.semantic_network.probabilistic_enhancement.conditional_probs[concept1].get(
                    concept2, 0)
                mi = cognitive_graph.semantic_network.probabilistic_enhancement.mutual_info[concept1].get(concept2, 0)
                print(f"{concept1} -> {concept2}: P={cond_prob:.3f}, MI={mi:.3f}")

    # 运行模拟
    print("\n=== 开始概率增强模拟 ===")
    cognitive_graph.monte_carlo_iteration(max_iterations=10000)

    # 显示结果
    stats = cognitive_graph.get_network_stats()
    print(f"\n=== 模拟结果 ===")
    print(f"最终平均能耗: {stats['avg_energy']:.3f}")
    print(f"概念压缩中心: {stats['compression_centers']}个")
    print(f"迁移桥梁: {stats['migration_bridges']}个")

    return cognitive_graph

def create_individual_variation_simulator():
    """创建个体差异模拟器"""
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

    return IndividualVariation(base_parameters, variation_ranges)


def run_individual_simulation(individual_id, individual_params, max_iterations=10000):
    """运行单个个体模拟"""
    print(f"\n=== 开始模拟 {individual_id} ===")

    # 创建认知图实例
    cognitive_graph = SemanticEnhancedCognitiveGraph(individual_params)
    cognitive_graph.initialize_semantic_graph()

    # 记录初始状态
    initial_energy = cognitive_graph.get_average_energy()
    print(f"初始平均能耗: {initial_energy:.3f}")

    # 运行蒙特卡洛模拟
    cognitive_graph.monte_carlo_iteration(max_iterations=max_iterations)

    # 获取最终统计
    final_stats = cognitive_graph.get_network_stats()
    improvement = ((initial_energy - final_stats['avg_energy']) / initial_energy * 100)

    print(f"{individual_id} 模拟完成:")
    print(f"  最终平均能耗: {final_stats['avg_energy']:.3f}")
    print(f"  能耗降低: {improvement:.1f}%")
    print(f"  压缩中心: {final_stats['compression_centers']}个")
    print(f"  迁移桥梁: {final_stats['migration_bridges']}个")

    return {
        'individual_id': individual_id,
        'cognitive_graph': cognitive_graph,
        'initial_energy': initial_energy,
        'final_stats': final_stats,
        'improvement': improvement,
        'energy_history': cognitive_graph.energy_history.copy(),
        'parameters': individual_params
    }


def run_multiple_simulations(num_individuals=3, max_iterations=5000):
    """运行多个个体模拟"""
    print(f"=== 开始群体模拟实验: {num_individuals}个个体 ===")

    # 创建个体差异模拟器
    variation_simulator = create_individual_variation_simulator()
    visualization = CognitiveVisualization()
    simulation_results = []

    # 运行每个个体的模拟
    for i in range(num_individuals):
        individual_id = f"个体_{i + 1}"

        # 生成个体参数
        base_params = variation_simulator.generate_individual(individual_id)
        individual_params = create_enhanced_individual_params(base_params)

        # 运行模拟
        result = run_individual_simulation(individual_id, individual_params, max_iterations)
        simulation_results.append(result)

        # 可视化单个结果（可选）
        if i == 0:  # 只可视化第一个个体以节省时间
            print(f"\n可视化 {individual_id} 的结果...")
            visualization.visualize_cognitive_graph(
                result['cognitive_graph'],
                f"{individual_id} - 认知图结构"
            )
            visualization.visualize_energy_convergence(
                result['cognitive_graph'],
                f"{individual_id} - 能耗收敛过程"
            )
            visualization.visualize_cognitive_states(
                result['cognitive_graph'],
                f"{individual_id} - 认知状态变化"
            )

    # 比较所有模拟结果
    print(f"\n=== 群体模拟结果比较 ===")
    visualization.compare_multiple_simulations(simulation_results)

    # 显示概率网络可视化
    if simulation_results:
        first_graph = simulation_results[0]['cognitive_graph']
        if hasattr(first_graph, 'semantic_network'):
            visualization.visualize_probabilistic_network(first_graph.semantic_network)

    return simulation_results


def demo_quick_simulation():
    """快速演示模拟"""
    print("=== 快速演示模拟 ===")

    # 创建单个个体
    base_params = {
        'forgetting_rate': 0.002,
        'base_learning_rate': 0.85,
        'hard_traversal_bias': 0.0,
        'soft_traversal_bias': 0.0,
        'compression_bias': 0.0,
        'migration_bias': 0.0,
        'learning_rate_variation': 0.1
    }

    individual_params = create_enhanced_individual_params(base_params)

    # 运行模拟
    result = run_individual_simulation("演示个体", individual_params, max_iterations=3000)

    # 可视化
    viz = CognitiveVisualization()
    viz.visualize_cognitive_graph(result['cognitive_graph'], "演示认知图")
    viz.visualize_energy_convergence(result['cognitive_graph'], "演示能耗收敛")
    viz.visualize_cognitive_states(result['cognitive_graph'], "演示认知状态")

    return result


# 需要补充的辅助函数
def create_enhanced_individual_params(base_params):
    """创建增强的个体参数"""
    enhanced_params = base_params.copy()
    enhanced_params.update({
        'energy_variation': random.uniform(0.05, 0.15),
        'focus_bias': random.uniform(-0.1, 0.1),
        'exploration_bias': random.uniform(-0.1, 0.1),
        'fatigue_resistance': random.uniform(0.1, 0.3),
        'inspiration_frequency': random.uniform(0.05, 0.2)
    })
    return enhanced_params


def debug_compression_info(self):
    """调试压缩相关信息"""
    if not hasattr(self, 'semantic_network'):
        print("没有语义网络")
        return

    print("\n=== 压缩调试信息 ===")
    print(f"认知图中的节点数: {self.G.number_of_nodes()}")
    print(f"语义网络中的概念数: {len(self.semantic_network.concept_definitions)}")

    # 检查互信息字典
    mi_concepts = list(self.semantic_network.probabilistic_enhancement.mutual_info.keys())
    print(f"互信息字典中的概念数: {len(mi_concepts)}")

    # 显示前几个概念
    print("前10个概念:", mi_concepts[:10])

    # 检查具体概念
    test_concepts = ["苹果", "香蕉", "物体", "运动"]
    for concept in test_concepts:
        in_graph = concept in self.G
        in_semantic = concept in self.semantic_network.concept_definitions
        in_mi = concept in self.semantic_network.probabilistic_enhancement.mutual_info
        print(f"{concept}: 图中={in_graph}, 语义={in_semantic}, 互信息={in_mi}")

if __name__ == "__main__":
    # 确保安装了必要的库
    try:
        import jieba
        import networkx as nx
    except ImportError:
        print("请安装所需库: pip install jieba networkx matplotlib numpy")
        exit(1)

    print("=== 概率增强认知图模型 ===")

    # 用户选择运行模式
    print("\n选择运行模式:")
    print("1. 快速演示模拟 (单个个体，3000次迭代)")
    print("2. 群体模拟实验 (多个个体，5000次迭代)")
    print("3. 完整实验 (多个个体，10000次迭代)")

    try:
        choice = input("请输入选择 (1/2/3, 默认为1): ").strip()

        if choice == "2":
            # 群体模拟
            results = run_multiple_simulations(num_individuals=3, max_iterations=5000)
        elif choice == "3":
            # 完整实验
            results = run_multiple_simulations(num_individuals=5, max_iterations=10000)
        else:
            # 快速演示
            result = demo_quick_simulation()

    except KeyboardInterrupt:
        print("\n用户中断模拟")
    except Exception as e:
        print(f"运行出错: {e}")



# if __name__ == "__main__":
#     # 运行演示
#     graph = demo_probabilistic_enhancement()