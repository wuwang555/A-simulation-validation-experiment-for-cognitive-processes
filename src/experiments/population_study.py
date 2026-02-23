"""
群体认知实验脚本
用于运行语义增强的群体实验，模拟多个个体的认知演化。
"""

import os
import csv
from utils.individual_variation import IndividualVariation, create_enhanced_individual_params
from models.enhanced_model import SemanticEnhancedCognitiveGraph, EnergyOptimizedCognitiveGraph
from utils.analysis import *
from core.cognitive_states import *
from utils.visualization import *


def run_semantic_enhanced_experiment(num_individuals=3, max_iterations=10000, num_concepts=None):
    """运行语义增强的群体实验。

    为多个个体生成随机参数，分别运行认知演化，并保存能量历史和分析结果。

    Args:
        num_individuals (int): 个体数量。
        max_iterations (int): 最大迭代次数。
        num_concepts (int, optional): 概念节点数量。

    Returns:
        list: 每个个体的结果字典列表。
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    population_results = []
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
    if num_concepts:
        print(f"概念数量: {num_concepts}")

    for i in range(num_individuals):
        individual_id = f"个体_{i + 1}"
        print(f"\n--- 模拟 {individual_id} ---")

        base_individual_params = variation_simulator.generate_individual(individual_id)
        individual_params = create_enhanced_individual_params(base_individual_params)

        # 创建认知图，传入num_concepts参数
        individual_graph = SemanticEnhancedCognitiveGraph(individual_params, num_concepts=num_concepts)
        individual_graph.initialize_semantic_graph()

        # 修复：使用 get_network_stats() 获取初始统计信息
        initial_stats = individual_graph.get_network_stats()
        initial_energy = initial_stats['avg_energy']

        individual_graph.monte_carlo_iteration(max_iterations=max_iterations)

        # 保存能量历史
        energy_history = individual_graph.cognitive_energy_history
        energy_file = os.path.join("results/population", f"energy_history_{individual_id}_{timestamp}.csv")
        os.makedirs("results/population", exist_ok=True)
        with open(energy_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'energy'])
            for idx, e in enumerate(energy_history):
                writer.writerow([idx, e])
        print(f"能量历史已保存: {energy_file}")

        # 修复：使用 get_network_stats() 获取最终统计信息
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


def test_enhanced_features(num_concepts=None):
    """测试增强功能，如相似度计算、蒙特卡洛迭代和智能压缩。

    Args:
        num_concepts (int, optional): 概念节点数量。

    Returns:
        EnergyOptimizedCognitiveGraph: 演化后的认知图对象。
    """
    print("=== 测试增强的认知图模型 ===")
    if num_concepts:
        print(f"概念数量: {num_concepts}")

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

    # 创建能耗优化的认知图，传入num_concepts参数
    cognitive_graph = EnergyOptimizedCognitiveGraph(base_params, num_concepts=num_concepts)
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


def demo_semantic_network(num_concepts=None):
    """演示语义网络功能，包括跨领域路径查找。

    Args:
        num_concepts (int, optional): 概念节点数量。
    """
    from core.semantic_network import SemanticConceptNetwork
    from utils.visualization import visualize_semantic_network

    semantic_net = SemanticConceptNetwork()
    # 构建语义网络，传入num_concepts参数
    semantic_net.build_comprehensive_network(num_concepts=num_concepts)

    print("\n=== 语义网络演示 ===")
    if num_concepts:
        print(f"概念数量: {num_concepts}")

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
    # 简单测试：运行一个较小规模的群体实验
    print("运行群体实验测试（2个个体，500次迭代）")
    results = run_semantic_enhanced_experiment(num_individuals=2, max_iterations=10000, num_concepts=51)
    print("测试完成。")