"""
认知图论实验平台主入口。

提供实验管理类，支持运行不同认知模型（随机网络、Q-learning、预设算法、自然涌现）
并进行对比分析。
"""

import sys
import os
from experiments.emergence_study_fixed import EmergenceStudyFixed
from core.semantic_network import SemanticConceptNetwork
from experiments.population_study import run_semantic_enhanced_experiment, demo_semantic_network
from utils.visualization import *

# 导入基准模型
from models.random_network import RandomNetworkModel
from models.qlearning_enhanced import EnhancedQLearningCognitiveGraph

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class CognitiveGraphExperimentManager:
    """
    认知图实验管理器。

    负责运行不同类型的认知模型实验，包括：
        - 随机网络模型（无智能基准）
        - 增强Q-learning模型（传统AI方法）
        - 预设算法模型（模拟传统认知计算范式）
        - 自然涌现模型（基于能量最小化的新范式）

    并提供对比分析功能。
    """

    def __init__(self):
        """初始化实验管理器，创建空的结果字典。"""
        self.experiment_results = {}

    def run_random_network_model(self, num_nodes=51, max_iterations=10000):
        """
        运行随机网络基准模型。

        Parameters
        ----------
        num_nodes : int, optional
            网络节点数，默认51。
        max_iterations : int, optional
            最大迭代次数，默认10000。

        Returns
        -------
        dict
            实验结果字典，包含 'improvement' 等键。
        """
        print("\n" + "=" * 50)
        print("随机网络基准模型（无智能机制）")
        print("=" * 50)

        base_params = {
            'forgetting_rate': 0.002,
            'base_learning_rate': 0.85,
            'hard_traversal_bias': 0.0,
            'soft_traversal_bias': 0.0,
            'compression_bias': 0.0,
            'migration_bias': 0.0,
            'learning_rate_variation': 0.1
        }

        random_model = RandomNetworkModel(base_params)
        result = random_model.run_experiment(num_nodes=num_nodes, max_iterations=max_iterations)

        random_model.visualize_graph("随机网络模型")

        self.experiment_results['random_network'] = result
        return result

    def run_qlearning_model(self, num_nodes=51, max_iterations=10000):
        """
        运行增强的Q-learning基准模型。

        Parameters
        ----------
        num_nodes : int, optional
            网络节点数，默认51。
        max_iterations : int, optional
            最大迭代次数，默认10000。

        Returns
        -------
        dict
            实验结果字典，包含 'improvement', 'q_table_stats' 等键。
        """
        print("\n" + "=" * 50)
        print("增强Q-learning基准模型（传统强化学习+改进）")
        print("=" * 50)

        base_params = {
            'forgetting_rate': 0.002,
            'base_learning_rate': 0.85,
            'hard_traversal_bias': 0.0,
            'soft_traversal_bias': 0.0,
            'compression_bias': 0.0,
            'migration_bias': 0.0,
            'learning_rate_variation': 0.1
        }

        qlearning_model = EnhancedQLearningCognitiveGraph(base_params)
        result = qlearning_model.run_experiment(num_nodes=num_nodes, max_iterations=max_iterations)

        qlearning_model.visualize_graph("增强Q-learning模型")

        if num_nodes >= 5 and 'path_examples' in result and result['path_examples']:
            print("\n最优路径示例:")
            for i, example in enumerate(result['path_examples'][:2]):
                print(f"  示例{i + 1}: {example['start']} -> {example['end']}")
                print(f"    路径: {example['path']}")
                print(f"    累计Q值: {example['q_value']:.3f}")

        if 'q_table_stats' in result:
            q_stats = result['q_table_stats']
            print(f"\nQ-table统计:")
            print(f"  大小: {q_stats['size']}")
            print(f"  稀疏度: {q_stats['sparsity']:.3f}")
            print(f"  非零条目: {q_stats['non_zero_entries']}")
            print(f"  平均值: {q_stats['mean']:.3f}")
            print(f"  正值比例: {q_stats['positive_ratio']:.3f}")

        self.experiment_results['qlearning'] = result
        return result

    def run_preset_algorithm_model(self, num_concepts=None):
        """
        运行传统机制设计模型（预设算法）。

        Parameters
        ----------
        num_concepts : int, optional
            概念数量，若为None则使用默认规模。

        Returns
        -------
        list
            每个个体的实验结果列表。
        """
        print("\n" + "=" * 50)
        print("传统机制设计模型（预设算法）")
        if num_concepts:
            print(f"使用概念数: {num_concepts}")
        print("=" * 50)

        results = run_semantic_enhanced_experiment(
            num_individuals=2,
            max_iterations=10000,
            num_concepts=num_concepts
        )

        if results and len(results) > 0:
            first_individual = results[0]
            if 'graph' in first_individual:
                first_individual['graph'].visualize_graph("传统机制模型")

        self.experiment_results['traditional'] = results
        return results

    def run_natural_emergence_model(self, num_individuals=2, max_iterations=10000, num_concepts=None):
        """
        运行纯粹能量模型，观察自然涌现现象。

        Parameters
        ----------
        num_individuals : int, optional
            个体数量，默认2。
        max_iterations : int, optional
            最大迭代次数，默认10000。
        num_concepts : int, optional
            概念数量，若为None则使用默认规模。

        Returns
        -------
        list
            每个个体的实验结果列表。
        """
        print("\n" + "=" * 50)
        print("纯粹能量模型 - 自然涌现观察")
        if num_concepts:
            print(f"使用概念数: {num_concepts}")
        print("=" * 50)

        study = EmergenceStudyFixed()
        results = study.run_pure_emergence_experiment(
            num_individuals=num_individuals,
            max_iterations=max_iterations,
            num_concepts=num_concepts
        )

        study.visualize_emergence_results()

        self.experiment_results['pure_emergence'] = results
        return results

    def run_benchmark_comparison(self, num_nodes=51, max_iterations=10000, num_concepts=None):
        """
        运行所有基准模型对比实验。

        Parameters
        ----------
        num_nodes : int, optional
            网络节点数，用于随机和Q-learning模型，默认51。
        max_iterations : int, optional
            最大迭代次数，默认10000。
        num_concepts : int, optional
            语义模型使用的概念数，若为None则使用默认规模。

        Returns
        -------
        dict
            包含各模型结果的字典，键为模型名称。
        """
        print("\n" + "=" * 60)
        print("基准模型对比实验")
        if num_concepts:
            print(f"语义模型使用概念数: {num_concepts}")
        print("=" * 60)

        results = {}

        print("\n1. 运行随机网络模型...")
        random_result = self.run_random_network_model(num_nodes, max_iterations)
        results['random'] = random_result

        print("\n2. 运行增强Q-learning模型...")
        qlearning_result = self.run_qlearning_model(num_nodes, max_iterations)
        results['qlearning'] = qlearning_result

        print("\n3. 运行传统机制模型...")
        from experiments.population_study import run_semantic_enhanced_experiment
        traditional_results = run_semantic_enhanced_experiment(
            num_individuals=1,
            max_iterations=max_iterations,
            num_concepts=num_concepts
        )
        if traditional_results:
            results['traditional'] = traditional_results[0]

        print("\n4. 运行自然涌现模型...")
        study = EmergenceStudyFixed()
        emergence_results = study.run_pure_emergence_experiment(
            num_individuals=1,
            max_iterations=max_iterations,
            num_concepts=num_concepts
        )
        if emergence_results:
            results['emergence'] = emergence_results[0]

        self._compare_benchmark_results(results)
        self.experiment_results['benchmark_comparison'] = results
        return results

    def _compare_benchmark_results(self, results):
        """
        内部方法：打印基准模型性能对比表，并绘制对比图。

        Parameters
        ----------
        results : dict
            各模型结果字典。
        """
        print("\n" + "=" * 60)
        print("基准模型性能对比")
        print("=" * 60)

        print(f"{'模型类型':<20} {'能耗降低(%)':<15} {'计算迭代':<12} {'节点数':<10}")
        print("-" * 60)

        for model_type, result in results.items():
            if isinstance(result, dict) and 'improvement' in result:
                improvement = result['improvement']
                iterations = result.get('iterations', 'N/A')
                nodes = result.get('num_nodes', 'N/A')
                print(f"{model_type:<20} {improvement:<15.1f} {iterations:<12} {nodes:<10}")
            elif isinstance(result, list) and len(result) > 0 and 'improvement' in result[0]:
                improvement = result[0]['improvement']
                print(f"{model_type:<20} {improvement:<15.1f} {'N/A':<12} {'N/A':<10}")

        print("-" * 60)
        self._plot_benchmark_comparison(results)

    def _plot_benchmark_comparison(self, results):
        """
        内部方法：绘制基准模型性能对比柱状图。

        Parameters
        ----------
        results : dict
            各模型结果字典。
        """
        try:
            import matplotlib.pyplot as plt

            model_names = []
            improvements = []
            colors = []

            color_map = {
                'random': 'gray',
                'qlearning': 'blue',
                'traditional': 'green',
                'emergence': 'red'
            }

            for model_type, result in results.items():
                if model_type in color_map:
                    if isinstance(result, dict) and 'improvement' in result:
                        model_names.append(model_type)
                        improvements.append(result['improvement'])
                        colors.append(color_map[model_type])
                    elif isinstance(result, list) and len(result) > 0 and 'improvement' in result[0]:
                        model_names.append(model_type)
                        improvements.append(result[0]['improvement'])
                        colors.append(color_map[model_type])

            if not model_names:
                return

            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, improvements, color=colors, alpha=0.7)

            for bar, improvement in zip(bars, improvements):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{improvement:.1f}%', ha='center', va='bottom')

            plt.xlabel('模型类型')
            plt.ylabel('能耗降低 (%)')
            plt.title('基准模型性能对比')
            plt.ylim(0, max(improvements) * 1.2 if improvements else 50)
            plt.grid(True, alpha=0.3)

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, label=name, alpha=0.7)
                               for name, color in color_map.items() if name in model_names]
            plt.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            plt.show()

            os.makedirs('results/comparison', exist_ok=True)
            plt.savefig('results/comparison/benchmark_comparison.png', dpi=300, bbox_inches='tight')
            print(f"对比图表已保存到: results/comparison/benchmark_comparison.png")

        except Exception as e:
            print(f"绘制对比图表时出错: {e}")

    def run_complete_study(self, num_concepts=None):
        """
        运行完整研究，包括基准对比和主要模型对比。

        Parameters
        ----------
        num_concepts : int, optional
            概念数量。

        Returns
        -------
        tuple
            传统模型和涌现模型的结果列表。
        """
        print("\n" + "=" * 50)
        print("完整认知图研究")
        if num_concepts:
            print(f"使用概念数: {num_concepts}")
        print("=" * 50)

        print("阶段1: 基准模型对比")
        benchmark_results = self.run_benchmark_comparison(num_nodes=51, max_iterations=10000, num_concepts=num_concepts)

        print("\n阶段2: 主要模型对比")
        traditional_results = self.run_preset_algorithm_model(num_concepts=num_concepts)
        emergence_results = self.run_natural_emergence_model(num_concepts=num_concepts)

        self._compare_results(traditional_results, emergence_results)
        return traditional_results, emergence_results

    def _compare_results(self, traditional, emergence):
        """
        内部方法：简单对比传统模型与涌现模型的结果。

        Parameters
        ----------
        traditional : list
            传统模型结果列表。
        emergence : list
            涌现模型结果列表。
        """
        print("\n" + "=" * 50)
        print("模型对比结果")
        print("=" * 50)

        if traditional and emergence:
            trad_improve = sum(r['improvement'] for r in traditional) / len(traditional)
            emerge_improve = sum(r.get('energy_improvement', 0) for r in emergence) / len(emergence)

            print(f"传统模型平均改善: {trad_improve:.1f}%")
            print(f"涌现模型平均改善: {emerge_improve:.1f}%")

    def quick_demo(self, num_concepts=None):
        """
        快速演示模式：运行语义网络演示和一个小型涌现实验。

        Parameters
        ----------
        num_concepts : int, optional
            概念数量。

        Returns
        -------
        list
            涌现模型实验结果。
        """
        print("\n" + "=" * 50)
        print("快速演示模式")
        if num_concepts:
            print(f"使用概念数: {num_concepts}")
        print("=" * 50)

        demo_semantic_network(num_concepts=num_concepts)

        return self.run_natural_emergence_model(
            num_individuals=1,
            max_iterations=2000,
            num_concepts=num_concepts
        )

    def show_summary(self):
        """
        显示已运行实验的总结信息。
        """
        print("\n" + "=" * 50)
        print("实验总结")
        print("=" * 50)

        for exp_type, results in self.experiment_results.items():
            if results:
                if isinstance(results, dict):
                    print(f"{exp_type}: 1个实验")
                    if 'improvement' in results:
                        print(f"  能耗改善: {results['improvement']:.1f}%")
                elif isinstance(results, list):
                    print(f"{exp_type}: {len(results)}个实验")
                else:
                    print(f"{exp_type}: 结果数据")


def check_dependencies():
    """
    检查必要的依赖库是否已安装。

    Returns
    -------
    bool
        所有依赖库都存在返回True，否则返回False。
    """
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


def get_concept_count_input():
    """
    交互式获取用户选择的概念数量。

    Returns
    -------
    int
        用户选择的概念数量。
    """
    print("\n选择概念数量:")
    print("1. 51个概念（默认）")
    print("2. 71个概念")
    print("3. 120个概念（完整）")
    print("4. 自定义概念数")

    choice = input("请选择 (1-4，默认1): ").strip() or "1"

    if choice == "1":
        return 51
    elif choice == "2":
        return 71
    elif choice == "3":
        return 120
    elif choice == "4":
        custom_input = input("请输入自定义概念数 (1-120，默认51): ").strip()
        if custom_input == "":
            return 51
        try:
            num_concepts = int(custom_input)
            if num_concepts < 1:
                print("概念数必须大于0，使用默认值51")
                return 51
            elif num_concepts > 120:
                print("概念数最大为120，使用120")
                return 120
            return num_concepts
        except ValueError:
            print("输入无效，使用默认值51")
            return 51
    else:
        print("输入无效，使用默认值51")
        return 51


def main():
    """
    主函数：提供交互式菜单，根据用户选择运行相应实验。
    """
    if not check_dependencies():
        return

    print("\n=== 认知图模型实验平台 ===")
    print("选择运行模式:")
    print("1. 随机网络基准模型（无智能）")
    print("2. 增强Q-learning基准模型（传统AI+改进）")
    print("3. 传统机制设计模型（预设算法）")
    print("4. 纯粹能量涌现模型（自然涌现）")
    print("5. 基准模型对比实验")
    print("6. 完整对比研究")
    print("7. 语义网络演示")
    print("8. 快速演示")
    print("9. 实验总结")

    manager = CognitiveGraphExperimentManager()

    try:
        choice = input("\n请选择模式 (1-9): ").strip()

        if choice == "1":
            num_nodes = input("网络节点数 (默认51): ").strip() or "51"
            max_iterations = input("迭代次数 (默认3000): ").strip() or "3000"
            manager.run_random_network_model(
                num_nodes=int(num_nodes),
                max_iterations=int(max_iterations)
            )
        elif choice == "2":
            num_nodes = input("网络节点数 (默认51): ").strip() or "51"
            max_iterations = input("迭代次数 (默认5000): ").strip() or "5000"
            manager.run_qlearning_model(
                num_nodes=int(num_nodes),
                max_iterations=int(max_iterations)
            )
        elif choice == "3":
            num_concepts = get_concept_count_input()
            manager.run_preset_algorithm_model(num_concepts=num_concepts)
        elif choice == "4":
            num_individuals = input("个体数量 (默认2): ").strip() or "2"
            max_iterations = input("迭代次数 (默认8000): ").strip() or "8000"
            num_concepts = get_concept_count_input()
            manager.run_natural_emergence_model(
                num_individuals=int(num_individuals),
                max_iterations=int(max_iterations),
                num_concepts=num_concepts
            )
        elif choice == "5":
            num_nodes = input("网络节点数 (默认51): ").strip() or "51"
            max_iterations = input("迭代次数 (默认3000): ").strip() or "3000"
            print("\n语义模型概念数量设置（随机和Q-learning模型不受影响）:")
            num_concepts = get_concept_count_input()
            manager.run_benchmark_comparison(
                num_nodes=int(num_nodes),
                max_iterations=int(max_iterations),
                num_concepts=num_concepts
            )
        elif choice == "6":
            num_concepts = get_concept_count_input()
            manager.run_complete_study(num_concepts=num_concepts)
        elif choice == "7":
            num_concepts = get_concept_count_input()
            demo_semantic_network(num_concepts=num_concepts)
        elif choice == "8":
            num_concepts = get_concept_count_input()
            manager.quick_demo(num_concepts=num_concepts)
        elif choice == "9":
            manager.show_summary()
        else:
            print("无效选择，运行基准模型对比")
            manager.run_benchmark_comparison()

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")


if __name__ == "__main__":
    main()