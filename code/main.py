import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.semantic_network import SemanticConceptNetwork
from experiments.population_study import run_semantic_enhanced_experiment, test_enhanced_features
from utils.visualization import *
from experiments.population_study import demo_semantic_network

def main():
    print("=== 语义增强认知图模型 ===")

    # 演示语义网络
    demo_semantic_network()

    # 运行群体实验
    enhanced_results = run_semantic_enhanced_experiment(
        num_individuals=3,
        max_iterations=10000
    )

    # 测试增强功能
    enhanced_graph = test_enhanced_features()

    # 可视化结果
    if enhanced_graph:
        enhanced_graph.visualize_graph("能耗优化认知图")
        enhanced_graph.visualize_energy_convergence()
        enhanced_graph.visualize_cognitive_states()


if __name__ == "__main__":
    # 检查依赖库
    try:
        import jieba
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("请安装所需库: pip install jieba networkx matplotlib numpy")
        sys.exit(1)

    main()