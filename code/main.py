import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.emergence_study import EmergenceStudy
from core.semantic_network import SemanticConceptNetwork
from experiments.population_study import run_semantic_enhanced_experiment, demo_semantic_network
from utils.visualization import *


def main():
    print("=== 认知图模型：机制设计与自然涌现对比研究 ===")
    print("选择运行模式:")
    print("1. 传统机制设计模型 (人工干预)")
    print("2. 纯粹能量模型 (自然涌现观察)")
    print("3. 对比实验 (机制 vs 涌现)")
    print("4. 语义网络演示")
    print("5. 完整涌现研究实验")

    choice = input("请选择模式 (1-5): ").strip()

    if choice == "1":
        run_traditional_mechanisms()
    elif choice == "2":
        # 使用修复版本
        from experiments.emergence_study_fixed import EmergenceStudyFixed
        study = EmergenceStudyFixed()
        study.run_pure_emergence_experiment(num_individuals=2, max_iterations=3000)
        study.visualize_emergence_results()
    elif choice == "3":
        run_comparison_study()
    elif choice == "4":
        demo_semantic_network()
    elif choice == "5":
        run_complete_emergence_study()
    else:
        print("无效选择，运行默认对比实验")
        run_comparison_study()


def run_traditional_mechanisms():
    """运行传统机制设计模型"""
    print("\n" + "=" * 60)
    print("运行传统机制设计模型")
    print("特点：人工设计的概念压缩和原理迁移机制")
    print("=" * 60)

    # 运行原有的群体实验
    enhanced_results = run_semantic_enhanced_experiment(
        num_individuals=2,  # 减少个体数以加快演示
        max_iterations=8000
    )

    # 可视化结果
    if enhanced_results and len(enhanced_results) > 0:
        # 使用第一个个体的图形进行可视化
        first_individual = enhanced_results[0]
        if 'graph' in first_individual:
            first_individual['graph'].visualize_graph("传统机制设计模型")
            first_individual['graph'].visualize_energy_convergence()
            first_individual['graph'].visualize_cognitive_states()


def run_pure_emergence():
    """运行纯粹能量模型"""
    print("\n" + "=" * 60)
    print("运行纯粹能量模型 - 自然涌现观察")
    print("特点：只实现两个公设，观察一切自然涌现")
    print("=" * 60)

    study = EmergenceStudy()

    # 运行纯粹涌现实验
    emergence_results = study.run_pure_emergence_experiment(
        num_individuals=2,
        max_iterations=8000
    )

    # 可视化结果
    study.visualize_emergence_results()

    # 生成详细报告
    study.generate_emergence_report()


def run_comparison_study():
    """运行对比实验"""
    print("\n" + "=" * 60)
    print("运行机制设计与自然涌现对比实验")
    print("=" * 60)

    study = EmergenceStudy()

    comparison_results = study.run_comparison_experiment(
        num_individuals=2,
        max_iterations=6000  # 减少迭代以加快演示
    )


def run_complete_emergence_study():
    """运行完整的涌现研究实验"""
    print("\n" + "=" * 60)
    print("运行完整涌现研究实验")
    print("=" * 60)

    study = EmergenceStudy()

    # 运行完整的涌现实验
    emergence_results = study.run_pure_emergence_experiment(
        num_individuals=3,
        max_iterations=10000
    )

    # 可视化结果
    study.visualize_emergence_results()

    # 生成详细报告
    study.generate_emergence_report()

    # 询问是否运行对比实验
    run_comparison = input("\n是否运行对比实验? (y/n): ").lower().strip()
    if run_comparison == 'y':
        comparison_results = study.run_comparison_experiment(
            num_individuals=2,
            max_iterations=8000
        )

    return study


def quick_demo():
    """快速演示功能"""
    print("\n" + "=" * 60)
    print("快速演示模式")
    print("=" * 60)

    # 演示语义网络
    print("1. 演示语义网络...")
    demo_semantic_network()

    # 快速运行一个小型涌现实验
    print("\n2. 运行小型涌现实验...")
    study = EmergenceStudy()

    emergence_results = study.run_pure_emergence_experiment(
        num_individuals=1,
        max_iterations=3000
    )

    # 简要可视化
    if hasattr(study, 'visualize_emergence_results'):
        study.visualize_emergence_results()


if __name__ == "__main__":
    # 检查依赖库
    try:
        import jieba
        import networkx as nx
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装所需库: pip install jieba networkx matplotlib numpy")
        sys.exit(1)

    # 检查emergence模块是否存在
    try:
        from emergence.detector import EmergenceDetector
        from emergence.observer import EmergenceObserver
        from emergence.universe import CognitiveUniverse
        from emergence.metrics import NaturalEmergenceMetrics

        print("✅ 涌现观察框架加载成功")
    except ImportError as e:
        print(f"❌ 涌现观察框架加载失败: {e}")
        print("请确保 emergence/ 目录及其中的文件存在")
        # 尝试列出emergence目录内容来帮助调试
        emergence_dir = os.path.join(os.path.dirname(__file__), 'emergence')
        if os.path.exists(emergence_dir):
            print(f"emergence目录内容: {os.listdir(emergence_dir)}")
        sys.exit(1)

    # 检查EmergenceStudy是否存在
    try:
        from experiments.emergence_study import EmergenceStudy

        print("✅ EmergenceStudy加载成功")
    except ImportError as e:
        print(f"❌ EmergenceStudy加载失败: {e}")
        print("请确保 experiments/emergence_study.py 文件存在")
        sys.exit(1)

    # 运行主程序
    main()