import sys
import os
from experiments.emergence_study_fixed import EmergenceStudyFixed
from core.semantic_network import SemanticConceptNetwork
from experiments.population_study import run_semantic_enhanced_experiment, demo_semantic_network
from utils.visualization import *

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class CognitiveGraphExperimentManager:
    """认知图实验管理器 - 精简版本"""

    def __init__(self):
        self.experiment_results = {}

    def run_traditional_mechanisms(self):
        """运行传统机制设计模型"""
        print("\n" + "=" * 50)
        print("传统机制设计模型")
        print("=" * 50)

        results = run_semantic_enhanced_experiment(
            num_individuals=2,
            max_iterations=10000
        )

        # 简单可视化第一个个体的结果
        if results and len(results) > 0:
            first_individual = results[0]
            if 'graph' in first_individual:
                first_individual['graph'].visualize_graph("传统机制模型")

        self.experiment_results['traditional'] = results
        return results

    def run_pure_emergence(self, num_individuals=2, max_iterations=10000):
        """运行纯粹能量模型"""
        print("\n" + "=" * 50)
        print("纯粹能量模型 - 自然涌现观察")
        print("=" * 50)

        study = EmergenceStudyFixed()
        results = study.run_pure_emergence_experiment(
            num_individuals=num_individuals,
            max_iterations=max_iterations
        )

        # 可视化结果
        study.visualize_emergence_results()

        self.experiment_results['pure_emergence'] = results
        return results

    def run_complete_study(self):
        """运行完整研究"""
        print("\n" + "=" * 50)
        print("完整认知图研究")
        print("=" * 50)

        # 运行两种模型
        traditional_results = self.run_traditional_mechanisms()
        emergence_results = self.run_pure_emergence()

        # 简单对比
        self._compare_results(traditional_results, emergence_results)
        return traditional_results, emergence_results

    def _compare_results(self, traditional, emergence):
        """简单对比结果"""
        print("\n" + "=" * 50)
        print("模型对比结果")
        print("=" * 50)

        if traditional and emergence:
            trad_improve = sum(r['improvement'] for r in traditional) / len(traditional)
            emerge_improve = sum(r.get('energy_improvement', 0) for r in emergence) / len(emergence)

            print(f"传统模型平均改善: {trad_improve:.1f}%")
            print(f"涌现模型平均改善: {emerge_improve:.1f}%")

    def quick_demo(self):
        """快速演示"""
        print("\n" + "=" * 50)
        print("快速演示模式")
        print("=" * 50)

        # 演示语义网络
        demo_semantic_network()

        # 运行小型实验
        return self.run_pure_emergence(
            num_individuals=1,
            max_iterations=2000
        )

    def show_summary(self):
        """显示实验总结"""
        print("\n" + "=" * 50)
        print("实验总结")
        print("=" * 50)

        for exp_type, results in self.experiment_results.items():
            if results:
                count = len(results) if isinstance(results, list) else 1
                print(f"{exp_type}: {count}个实验")


def check_dependencies():
    """检查依赖库"""
    required_libs = ['jieba', 'networkx', 'matplotlib', 'numpy']
    missing_libs = []

    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)

    if missing_libs:
        print(f"❌ 缺少依赖库: {', '.join(missing_libs)}")
        print("请安装: pip install " + " ".join(missing_libs))
        return False

    print("✅ 所有依赖库加载成功")
    return True


def main():
    """主函数"""
    if not check_dependencies():
        return

    print("\n=== 认知图模型实验平台 ===")
    print("选择运行模式:")
    print("1. 传统机制设计模型")
    print("2. 纯粹能量涌现模型")
    print("3. 完整对比研究")
    print("4. 语义网络演示")
    print("5. 快速演示")
    print("6. 实验总结")

    manager = CognitiveGraphExperimentManager()

    try:
        choice = input("\n请选择模式 (1-6): ").strip()

        if choice == "1":
            manager.run_traditional_mechanisms()
        elif choice == "2":
            num_individuals = input("个体数量 (默认2): ").strip() or "2"
            max_iterations = input("迭代次数 (默认8000): ").strip() or "8000"
            manager.run_pure_emergence(
                num_individuals=int(num_individuals),
                max_iterations=int(max_iterations)
            )
        elif choice == "3":
            manager.run_complete_study()
        elif choice == "4":
            demo_semantic_network()
        elif choice == "5":
            manager.quick_demo()
        elif choice == "6":
            manager.show_summary()
        else:
            print("无效选择，运行快速演示")
            manager.quick_demo()

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")


if __name__ == "__main__":
    main()