"""
增强认知宇宙模块
-----------------
继承 CognitiveUniverse，增加 EmergenceDetectorFixed 进行实时涌现检测，
并支持指定概念数量构建网络。
"""

from typing import Dict, Any, Optional
import random

from emergence.universe import CognitiveUniverse
from emergence.detector_fixed import EmergenceDetectorFixed


class CognitiveUniverseEnhanced(CognitiveUniverse):
    """增强的认知宇宙，在演化过程中使用 EmergenceDetectorFixed 检测涌现现象。"""

    def __init__(self, individual_params: Optional[Dict[str, Any]] = None,
                 network_seed: int = 42, num_concepts: Optional[int] = None):
        """
        :param individual_params: 个体参数
        :param network_seed: 随机种子
        :param num_concepts: 构建语义网络时使用的概念数量
        """
        super().__init__(individual_params, network_seed)
        self.emergence_detector = EmergenceDetectorFixed()
        self.observations = {
            'natural_compressions': [],
            'natural_migrations': [],
            'energy_convergence_phases': []
        }
        self.num_concepts = num_concepts

    def initialize_semantic_network(self) -> None:
        """初始化语义网络，支持指定概念数量。"""
        from core.semantic_network import SemanticConceptNetwork

        semantic_net = SemanticConceptNetwork()
        semantic_net.build_comprehensive_network(num_concepts=self.num_concepts)

        nodes = list(semantic_net.concept_definitions.keys())
        self.G.add_nodes_from(nodes)

        initial_edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1:], i + 1):
                similarity = semantic_net.calculate_semantic_similarity(node1, node2)

                if similarity > 0.1:
                    energy = 2.0 - similarity * 1.5
                    energy = max(0.3, min(2.0, energy))

                    initial_edges.append((node1, node2, {
                        'weight': energy,
                        'traversal_count': 0,
                        'original_weight': energy,
                        'similarity': similarity
                    }))
                    self.last_activation_time[(node1, node2)] = 0

        for u, v, attr in initial_edges:
            self.G.add_edge(u, v, **attr)

        print(f"语义网络初始化: {len(nodes)}节点, {len(initial_edges)}条边")
        print(f"初始全局能量: {self.calculate_network_energy():.3f}")

    def evolve_with_emergence_detection(self, iterations: int = 1000,
                                        detection_interval: int = 200) -> Dict[str, list]:
        """带涌现检测的演化过程。

        :param iterations: 迭代次数
        :param detection_interval: 检测涌现的时间间隔
        :return: 观察到的涌现事件字典
        """
        print(f"开始增强演化: {iterations}次迭代，检测间隔: {detection_interval}")

        initial_energy = self.calculate_network_energy()
        self.energy_history = [initial_energy]

        for i in range(iterations):
            self.iteration_count += 1

            self.basic_energy_optimization()

            if random.random() < 0.3:
                self._random_traversal()

            if i % 10 == 0:
                self.apply_basic_forgetting()

            current_energy = self.calculate_network_energy()
            self.energy_history.append(current_energy)

            if i % detection_interval == 0 and i > 200:
                self._detect_emergence(i)

            if i % 500 == 0:
                improvement = ((initial_energy - current_energy) / initial_energy * 100) if initial_energy > 0 else 0
                print(f"迭代 {i}: 能量 = {current_energy:.3f} (改善: {improvement:.1f}%)")
                print(f"  检测到压缩: {len(self.observations['natural_compressions'])}")
                print(f"  检测到迁移: {len(self.observations['natural_migrations'])}")

        return self.observations

    def _detect_emergence(self, iteration: int) -> None:
        """执行涌现检测并记录新事件。"""
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

    def _is_duplicate_compression(self, new_compression: Dict) -> bool:
        """检查是否已经记录过相同的压缩事件。"""
        for existing in self.observations['natural_compressions']:
            if (existing['center'] == new_compression['center'] and
                    set(existing['related_nodes']) == set(new_compression['related_nodes'])):
                return True
        return False

    def _is_duplicate_migration(self, new_migration: Dict) -> bool:
        """增强的重复迁移检查，包括反向路径和近期频率。"""
        principle_node = new_migration['principle_node']
        from_node = new_migration['from_node']
        to_node = new_migration['to_node']

        # 完全相同的迁移
        for existing in self.observations['natural_migrations']:
            if (existing['principle_node'] == principle_node and
                    existing['from_node'] == from_node and
                    existing['to_node'] == to_node):
                return True

        # 方向相反的迁移
        for existing in self.observations['natural_migrations']:
            if (existing['principle_node'] == principle_node and
                    existing['from_node'] == to_node and
                    existing['to_node'] == from_node):
                return True

        # 同一原理节点近期出现过于频繁
        recent_count = 0
        for existing in self.observations['natural_migrations'][-10:]:
            if existing['principle_node'] == principle_node:
                recent_count += 1
                if recent_count >= 3:
                    return True

        return False


if __name__ == "__main__":
    enhanced = CognitiveUniverseEnhanced(num_concepts=51)
    enhanced.initialize_semantic_network()
    enhanced.evolve_with_emergence_detection(iterations=5000)
    print("CognitiveUniverseEnhanced 测试完成")