# universe_enhanced.py
from typing import Dict, Any
from emergence.universe import CognitiveUniverse
import random
from emergence.detector_fixed import EmergenceDetectorFixed

class CognitiveUniverseEnhanced(CognitiveUniverse):
    """增强的认知宇宙，改进涌现观察"""

    def __init__(self, individual_params: Dict[str, Any] = None, network_seed: int = 42, num_concepts: int = None):
        # 直接调用父类构造函数，传递正确的参数
        super().__init__(individual_params, network_seed)
        self.emergence_detector = EmergenceDetectorFixed()
        # 使用与emergence_study_fixed.py中一致的键名
        self.observations = {
            'natural_compressions': [],  # 改为与调用代码一致
            'natural_migrations': [],    # 改为与调用代码一致
            'energy_convergence_phases': []
        }
        self.num_concepts = num_concepts  # 保存概念数设置

    def initialize_semantic_network(self):
        """初始化语义网络 - 重写以支持num_concepts参数"""
        from core.semantic_network import SemanticConceptNetwork

        semantic_net = SemanticConceptNetwork()
        # 构建语义网络，传入num_concepts参数
        semantic_net.build_comprehensive_network(num_concepts=self.num_concepts)

        nodes = list(semantic_net.concept_definitions.keys())
        self.G.add_nodes_from(nodes)

        initial_edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1:], i + 1):
                similarity = semantic_net.calculate_semantic_similarity(node1, node2)

                if similarity > 0.1:
                    # 基于相似度设置初始能量（相似度越高，能量越低）
                    energy = 2.0 - similarity * 1.5
                    energy = max(0.3, min(2.0, energy))

                    initial_edges.append((node1, node2, {
                        'weight': energy,
                        'traversal_count': 0,
                        'original_weight': energy,
                        'similarity': similarity
                    }))
                    # 初始化激活时间为0
                    self.last_activation_time[(node1, node2)] = 0

        for u, v, attr in initial_edges:
            self.G.add_edge(u, v, **attr)

        print(f"语义网络初始化: {len(nodes)}节点, {len(initial_edges)}条边")
        print(f"初始全局能量: {self.calculate_network_energy():.3f}")

    def evolve_with_emergence_detection(self, iterations: int = 1000,
                                        detection_interval: int = 200):  # 增加检测间隔
        """带涌现检测的演化过程"""
        print(f"开始增强演化: {iterations}次迭代，检测间隔: {detection_interval}")

        initial_energy = self.calculate_network_energy()
        self.energy_history = [initial_energy]

        for i in range(iterations):
            self.iteration_count += 1

            # 基本操作
            self.basic_energy_optimization()

            if random.random() < 0.3:
                self._random_traversal()

            if i % 10 == 0:
                self.apply_basic_forgetting()

            # 记录能量
            current_energy = self.calculate_network_energy()
            self.energy_history.append(current_energy)

            # 定期检测涌现 - 减少检测频率
            if i % detection_interval == 0 and i > 200:  # 增加初始不检测的迭代数
                self._detect_emergence(i)

            if i % 500 == 0:
                improvement = ((initial_energy - current_energy) / initial_energy * 100) if initial_energy > 0 else 0
                print(f"迭代 {i}: 能量 = {current_energy:.3f} (改善: {improvement:.1f}%)")
                print(f"  检测到压缩: {len(self.observations['natural_compressions'])}")
                print(f"  检测到迁移: {len(self.observations['natural_migrations'])}")

        return self.observations

    def _detect_emergence(self, iteration: int):
        """检测涌现现象"""
        # 检测概念压缩
        compressions = self.emergence_detector.detect_spontaneous_compression(
            self.G, self.energy_history, self.traversal_history
        )

        for compression in compressions:
            if not self._is_duplicate_compression(compression):
                compression['detection_iteration'] = iteration
                self.observations['natural_compressions'].append(compression)
                print(f"🎯 迭代 {iteration}: 发现自然概念压缩!")
                print(f"   中心: {compression['center']}, 节点数: {compression['cluster_size']}")
                print(f"   强度: {compression['emergence_strength']:.3f}")

        # 检测原理迁移
        migrations = self.emergence_detector.detect_emergent_migration(
            self.G, self.traversal_history, iteration
        )

        for migration in migrations:
            if not self._is_duplicate_migration(migration):
                migration['detection_iteration'] = iteration
                self.observations['natural_migrations'].append(migration)
                print(f"🌉 迭代 {iteration}: 发现自然原理迁移!")
                print(f"   原理: {migration['principle_node']}")
                print(f"   路径: {migration['from_node']} -> {migration['to_node']}")
                print(f"   效率: {migration['efficiency_gain']:.3f}")

    def _is_duplicate_compression(self, new_compression):
        """检查重复压缩"""
        for existing in self.observations['natural_compressions']:
            if (existing['center'] == new_compression['center'] and
                    set(existing['related_nodes']) == set(new_compression['related_nodes'])):
                return True
        return False

    def _is_duplicate_migration(self, new_migration):
        """增强的重复迁移检查"""
        principle_node = new_migration['principle_node']
        from_node = new_migration['from_node']
        to_node = new_migration['to_node']

        # 1. 完全相同的迁移
        for existing in self.observations['natural_migrations']:
            if (existing['principle_node'] == principle_node and
                    existing['from_node'] == from_node and
                    existing['to_node'] == to_node):
                return True

        # 2. 同一原理节点的类似连接（方向相反）
        for existing in self.observations['natural_migrations']:
            if (existing['principle_node'] == principle_node and
                    existing['from_node'] == to_node and
                    existing['to_node'] == from_node):
                return True

        # 3. 检查近期是否出现过相同原理节点的迁移
        recent_count = 0
        for existing in self.observations['natural_migrations'][-10:]:  # 最近10个
            if existing['principle_node'] == principle_node:
                recent_count += 1
                if recent_count >= 3:  # 同一原理节点在近期出现太频繁
                    return True

        return False