# test_energy_optimized.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_model import EnergyOptimizedCognitiveGraph


def test_energy_optimized_methods():
    """测试能耗优化模型的方法"""
    print("=== 测试能耗优化模型方法 ===")

    base_params = {
        'forgetting_rate': 0.002,
        'base_learning_rate': 0.85,
        'hard_traversal_bias': 0.0,
        'soft_traversal_bias': 0.0,
        'compression_bias': 0.0,
        'migration_bias': 0.0,
        'learning_rate_variation': 0.1
    }

    graph = EnergyOptimizedCognitiveGraph(base_params)
    graph.initialize_semantic_graph()

    # 测试新添加的方法是否存在
    methods_to_test = [
        '_find_high_energy_clusters',
        '_calculate_realistic_compression_saving',
        'improved_smart_concept_compression',
        'adaptive_learning_rate'
    ]

    for method in methods_to_test:
        if hasattr(graph, method):
            print(f"✓ {method} 方法存在")
        else:
            print(f"✗ {method} 方法缺失")

    # 测试方法功能
    try:
        # 测试查找高能耗集群
        clusters = graph._find_high_energy_clusters()
        print(f"找到 {len(clusters)} 个高能耗集群")

        # 测试计算压缩节省
        if clusters:
            center, nodes = clusters[0]
            if len(nodes) >= 2:
                saving = graph._calculate_realistic_compression_saving(center, nodes[:2])
                print(f"压缩节省计算: {saving:.3f}")

        # 测试自适应学习率
        learning_rate = graph.adaptive_learning_rate(1.0, 0.5, "hard")
        print(f"自适应学习率: {learning_rate:.3f}")

        print("✓ 所有方法功能测试通过")

    except Exception as e:
        print(f"✗ 方法功能测试失败: {e}")


if __name__ == "__main__":
    test_energy_optimized_methods()