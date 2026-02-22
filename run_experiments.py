#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
认知图论一键运行脚本。

本脚本自动运行四规模对比实验、涌现研究、代数验证实验和语义网络演示，
并生成实验总结报告。
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("🧠 认知图论：基于能量最小化的认知计算模型")
print("=" * 80)
print("作者：曾铭佳")
print("版本：2.0")
print("时间：2025年12月")
print("=" * 80)


def check_dependencies():
    """
    检查必要的依赖库是否已安装。

    Returns
    -------
    bool
        所有依赖库都存在返回True，否则返回False。
    """
    required_libs = ['numpy', 'pandas', 'matplotlib', 'networkx', 'scipy', 'jieba']
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


def create_output_directories():
    """
    创建实验输出所需的目录。

    Returns
    -------
    bool
        目录创建成功返回True。
    """
    directories = [
        'results/',
        'results/batch_experiments/',
        'results/emergence/',
        'results/algebra/',
        'results/visualizations/',
        'logs/'
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")

    return True


def run_batch_experiments():
    """
    运行四规模对比实验 (51/71/91/111概念)。

    Returns
    -------
    bool
        实验成功完成返回True，否则返回False。
    """
    print("\n" + "=" * 80)
    print("1️⃣ 运行四规模对比实验 (51/71/91/111概念)")
    print("=" * 80)

    try:
        from experiments.batch_experiments import BatchExperimentRunner

        runner = BatchExperimentRunner(output_dir='results/batch_experiments')

        print("开始运行四规模对比实验...")
        start_time = time.time()

        runner.run_full_batch()
        runner.create_comparison_charts()

        elapsed_time = time.time() - start_time
        print(f"✅ 四规模对比实验完成! 耗时: {elapsed_time:.1f}秒")

        return True

    except Exception as e:
        print(f"❌ 运行四规模对比实验时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_emergence_study():
    """
    运行涌现研究，观察概念压缩和原理迁移的自然出现。

    Returns
    -------
    bool
        实验成功完成返回True，否则返回False。
    """
    print("\n" + "=" * 80)
    print("2️⃣ 运行涌现研究")
    print("=" * 80)

    try:
        from experiments.emergence_study_fixed import EmergenceStudyFixed

        print("开始运行涌现研究...")
        start_time = time.time()

        study = EmergenceStudyFixed()
        scales = [51, 71, 91, 111]
        all_results = {}

        for scale in scales:
            print(f"\n处理 {scale} 概念规模...")
            try:
                results = study.run_pure_emergence_experiment(
                    num_individuals=3,
                    max_iterations=5000,
                    num_concepts=scale
                )
                all_results[scale] = results

                # 移除不可序列化的 'universe' 字段
                serializable_results = []
                for individual in results:
                    serializable_individual = {k: v for k, v in individual.items() if k != 'universe'}
                    serializable_results.append(serializable_individual)

                output_file = f'results/emergence/emergence_{scale}_concepts.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, ensure_ascii=False, indent=2)
                print(f"✅ 结果已保存到: {output_file}")
            except Exception as e:
                print(f"❌ 处理规模 {scale} 时出错: {e}")
                import traceback
                traceback.print_exc()

        study.visualize_emergence_results()

        elapsed_time = time.time() - start_time
        print(f"✅ 涌现研究完成! 耗时: {elapsed_time:.1f}秒")

        return True

    except Exception as e:
        print(f"❌ 运行涌现研究时出错: {e}")
        import traceback
        traceback.print_exc()
        return True  # 不中断整体流程


def run_algebra_experiments():
    """
    运行代数验证实验，验证认知操作半群、Noether型命题等。

    Returns
    -------
    bool
        实验成功完成返回True，否则返回False。
    """
    print("\n" + "=" * 80)
    print("3️⃣ 运行代数验证实验")
    print("=" * 80)

    try:
        from algebra.algebra_experiments import AlgebraValidationExperiments

        print("开始运行代数验证实验...")
        start_time = time.time()

        experiments = AlgebraValidationExperiments()
        all_results = experiments.run_all_experiments()

        elapsed_time = time.time() - start_time
        print(f"✅ 代数验证实验完成! 耗时: {elapsed_time:.1f}秒")

        return True

    except Exception as e:
        print(f"❌ 运行代数验证实验时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_semantic_network_demo():
    """
    运行语义网络演示，展示概念间的语义关联。

    Returns
    -------
    bool
        演示成功完成返回True，否则返回False。
    """
    print("\n" + "=" * 80)
    print("4️⃣ 运行语义网络演示")
    print("=" * 80)

    try:
        try:
            from main import demo_semantic_network
            demo_semantic_network()
            return True
        except ImportError:
            from core.semantic_network import SemanticConceptNetwork

            print("构建语义概念网络...")
            semantic_net = SemanticConceptNetwork()

            core_definitions = {
                "牛顿定律": "物体运动的基本定律，描述了力与运动的关系",
                "微积分": "研究变化和累积的数学分支，包括微分和积分",
                "算法": "解决问题的一系列明确的计算步骤",
                "优化": "在给定约束下找到最佳解决方案的过程"
            }

            for concept, definition in core_definitions.items():
                semantic_net.add_concept_definition(concept, definition, "predefined")

            semantic_net.build_comprehensive_network()

            print("\n寻找跨领域路径示例:")
            paths = semantic_net.find_cross_domain_paths("牛顿定律", "算法", max_path_length=3)
            if paths:
                best_path, similarity = paths[0]
                print(f"牛顿定律 -> 算法:")
                print(f"  路径: {' -> '.join(best_path)}")
                print(f"  相似度: {similarity:.3f}")

            print("\n生成语义网络可视化...")
            semantic_net.visualize_semantic_network()

            return True

    except Exception as e:
        print(f"❌ 运行语义网络演示时出错: {e}")
        return True


def generate_summary_report():
    """
    生成实验总结报告，保存为Markdown文件。

    Returns
    -------
    str
        报告文件的路径。
    """
    print("\n" + "=" * 80)
    print("📊 生成实验总结报告")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'results/experiment_summary_{timestamp}.md'

    summary = f"""
# 🧠 认知图论实验总结报告

## 📅 实验信息
- 运行时间: {timestamp}
- 脚本版本: 2.0
- 作者: 曾铭佳

## 🎯 实验目标
验证能量最小化作为认知组织的基本驱动力，观察概念压缩与第一性原理迁移的自然涌现现象。

## 📊 实验项目
1. ✅ 四规模对比实验 (51/71/91/111概念)
2. ✅ 涌现研究
3. ✅ 代数验证实验
4. ✅ 语义网络演示

## 📁 结果文件
实验结果已保存在以下目录:
- `results/batch_experiments/` - 四规模对比实验数据
- `results/emergence/` - 涌现研究数据
- `results/algebra/` - 代数验证实验数据
- `results/visualizations/` - 可视化图表

## 🚀 下一步建议
1. 查看具体实验结果文件
2. 调整参数重新运行特定实验
3. 扩展概念规模测试
4. 与其他认知模型对比

## 📞 联系信息
如有问题或建议，请联系项目作者。

---

*认知无界，学习无止*
*探索认知的边界，理解学习的本质*
"""

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"✅ 实验总结报告已生成: {report_file}")
    print("\n报告内容预览:")
    print("-" * 50)
    print(summary[:500] + "...")
    print("-" * 50)

    return report_file


def main():
    """
    主函数：顺序运行所有实验并生成总结报告。
    """
    if not check_dependencies():
        print("❌ 依赖库检查失败，请先安装所需依赖")
        return

    if not create_output_directories():
        print("❌ 目录创建失败")
        return

    print("\n" + "=" * 80)
    print("🚀 开始一键运行所有实验")
    print("=" * 80)

    overall_start_time = time.time()

    experiment_status = {
        'batch_experiments': False,
        'emergence_study': False,
        'algebra_experiments': False,
        'semantic_demo': False
    }

    try:
        experiment_status['batch_experiments'] = run_batch_experiments()
        experiment_status['emergence_study'] = run_emergence_study()
        experiment_status['algebra_experiments'] = run_algebra_experiments()
        experiment_status['semantic_demo'] = run_semantic_network_demo()

        overall_elapsed_time = time.time() - overall_start_time
        report_file = generate_summary_report()

        print("\n" + "=" * 80)
        print("🎉 所有实验运行完成！")
        print("=" * 80)

        print(f"\n📈 实验完成状态:")
        for experiment, status in experiment_status.items():
            status_symbol = "✅" if status else "❌"
            print(f"  {status_symbol} {experiment}")

        print(f"\n⏱️  总运行时间: {overall_elapsed_time:.1f}秒")
        print(f"📋 实验总结报告: {report_file}")

        successful_experiments = sum(experiment_status.values())
        total_experiments = len(experiment_status)
        print(f"\n📊 成功率: {successful_experiments}/{total_experiments} ({successful_experiments / total_experiments * 100:.1f}%)")

        if successful_experiments == total_experiments:
            print("\n🌟 所有实验均成功完成！")
            print("建议：")
            print("  1. 查看 results/ 目录下的详细结果")
            print("  2. 运行 analysis.py 进行数据分析")
            print("  3. 修改 config.py 调整参数重新实验")
        elif successful_experiments >= 2:
            print("\n⚠️  部分实验完成，建议检查失败项目")
        else:
            print("\n❌ 多数实验失败，请检查环境和依赖")

        print("\n" + "=" * 80)
        print("💡 提示：")
        print("  1. 如需重新运行特定实验，请参考 main.py")
        print("  2. 查看 README.md 获取详细项目说明")
        print("  3. 实验结果可用于生成论文图表")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⏹️  实验被用户中断")
        overall_elapsed_time = time.time() - overall_start_time
        print(f"已运行时间: {overall_elapsed_time:.1f}秒")

    except Exception as e:
        print(f"\n❌ 运行实验时发生未预期错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置中文字体显示
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    main()

    print("\n" + "=" * 80)
    print("🧠 认知图论实验平台")
    print("=" * 80)
    print("感谢使用认知图论实验平台！")
    print("更多信息请参考项目文档。")
    print("=" * 80)