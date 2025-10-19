# universe_enhanced.py
from typing import Dict, Any
from emergence.universe import CognitiveUniverse
import random
from emergence.detector_fixed import EmergenceDetectorFixed
class CognitiveUniverseEnhanced(CognitiveUniverse):
    """增强的认知宇宙，改进涌现观察"""

    def __init__(self, individual_params: Dict[str, Any] = None, network_seed: int = 42):
        super().__init__(individual_params, network_seed)
        self.emergence_detector = EmergenceDetectorFixed()
        self.enhanced_observations = {
            'natural_compressions': [],
            'natural_migrations': [],
            'energy_convergence_phases': []
        }

    def evolve_with_emergence_detection(self, iterations: int = 1000,
                                        detection_interval: int = 50):
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

            # 定期检测涌现
            if i % detection_interval == 0 and i > 100:  # 前100次迭代不检测
                self._detect_emergence(i)

            if i % 500 == 0:
                improvement = ((initial_energy - current_energy) / initial_energy * 100) if initial_energy > 0 else 0
                print(f"迭代 {i}: 能量 = {current_energy:.3f} (改善: {improvement:.1f}%)")
                print(f"  检测到压缩: {len(self.enhanced_observations['natural_compressions'])}")
                print(f"  检测到迁移: {len(self.enhanced_observations['natural_migrations'])}")

        return self.enhanced_observations

    def _detect_emergence(self, iteration: int):
        """检测涌现现象"""
        # 检测概念压缩
        compressions = self.emergence_detector.detect_spontaneous_compression(
            self.G, self.energy_history, self.traversal_history
        )

        for compression in compressions:
            if not self._is_duplicate_compression(compression):
                compression['detection_iteration'] = iteration
                self.enhanced_observations['natural_compressions'].append(compression)
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
                self.enhanced_observations['natural_migrations'].append(migration)
                print(f"🌉 迭代 {iteration}: 发现自然原理迁移!")
                print(f"   原理: {migration['principle_node']}")
                print(f"   路径: {migration['from_node']} -> {migration['to_node']}")
                print(f"   效率: {migration['efficiency_gain']:.3f}")

    def _is_duplicate_compression(self, new_compression):
        """检查重复压缩"""
        for existing in self.enhanced_observations['natural_compressions']:
            if (existing['center'] == new_compression['center'] and
                    set(existing['related_nodes']) == set(new_compression['related_nodes'])):
                return True
        return False

    def _is_duplicate_migration(self, new_migration):
        """检查重复迁移"""
        for existing in self.enhanced_observations['natural_migrations']:
            if (existing['principle_node'] == new_migration['principle_node'] and
                    existing['from_node'] == new_migration['from_node'] and
                    existing['to_node'] == new_migration['to_node']):
                return True
        return False