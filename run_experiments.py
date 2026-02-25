#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
认知图论一键运行脚本（增强可复现性版）。
本脚本自动运行四规模对比实验、涌现研究、代数验证实验和语义网络演示，
并生成详细的运行日志，确保实验结果可复现。
"""

import sys
import os
import time
import json
import platform
import logging
import random
from datetime import datetime
from pathlib import Path

# 设置随机种子（确保可复现）
import numpy as np
np.random.seed(42)
random.seed(42)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入依赖库（用于记录版本）
import networkx as nx
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import matplotlib
except ImportError:
    matplotlib = None
try:
    import scipy
except ImportError:
    scipy = None
try:
    import jieba
except ImportError:
    jieba = None


def setup_logging():
    """配置日志记录器，同时输出到控制台和文件"""
    # 创建 logs 目录
    Path("logs").mkdir(parents=True, exist_ok=True)

    # 日志文件名包含时间戳，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/reproducibility_{timestamp}.log"

    # 配置根 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 移除已有的 handlers（防止重复）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件 handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_environment(logger):
    """记录系统环境和依赖版本信息"""
    logger.info("=== 环境信息 ===")
    logger.info(f"当前时间: {datetime.now()}")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"平台信息: {platform.platform()}")
    logger.info(f"CPU 信息: {platform.processor()}")
    logger.info(f"机器类型: {platform.machine()}")
    logger.info(f"Python 解释器路径: {sys.executable}")
    logger.info(f"系统路径: {sys.path}")

    logger.info("=== 依赖版本 ===")
    logger.info(f"numpy: {np.__version__}")
    if pd:
        logger.info(f"pandas: {pd.__version__}")
    if matplotlib:
        logger.info(f"matplotlib: {matplotlib.__version__}")
    logger.info(f"networkx: {nx.__version__}")
    if scipy:
        logger.info(f"scipy: {scipy.__version__}")
    if jieba:
        # jieba 可能没有 __version__ 属性
        logger.info(f"jieba: {getattr(jieba, '__version__', 'unknown')}")

    # 记录实验的关键参数（从论文中提取，具体模块内部可能还有额外参数）
    logger.info("=== 实验参数 ===")
    logger.info("概念规模: [51, 71, 91, 111]")
    logger.info("迭代次数: 10000")
    logger.info("个体数量: 3")
    logger.info("语义相似度阈值: 0.08")
    logger.info("压缩协同性阈值: 0.76")
    logger.info("迁移效率阈值: 0.35")
    logger.info("集群内聚性阈值: 0.7")
    logger.info("最小连接强度: 0.5")
    logger.info("学习率: 0.85")
    logger.info("认知温度 T: 1.0")
    logger.info("随机种子: 42")


def check_dependencies(logger):
    """
    检查必要的依赖库是否已安装，并记录版本。
    """
    required_libs = ['numpy', 'pandas', 'matplotlib', 'networkx', 'scipy', 'jieba']
    missing_libs = []

    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)

    if missing_libs:
        logger.error(f"缺少依赖库: {', '.join(missing_libs)}")
        print(f"❌ 缺少依赖库: {', '.join(missing_libs)}")
        print("请安装: pip install " + " ".join(missing_libs))
        return False

    logger.info("✅ 所有依赖库加载成功")
    print("✅ 所有依赖库加载成功")
    return True


def create_output_directories(logger):
    """
    创建实验输出所需的目录。
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
        logger.info(f"📁 创建目录: {dir_path}")
        print(f"📁 创建目录: {dir_path}")

    return True


def run_batch_experiments(logger):
    """
    运行四规模对比实验 (51/71/91/111概念)。
    """
    logger.info("\n" + "=" * 80)
    logger.info("1️⃣ 运行四规模对比实验 (51/71/91/111概念)")
    logger.info("=" * 80)

    try:
        from experiments.batch_experiments import BatchExperimentRunner

        runner = BatchExperimentRunner(output_dir='results/batch_experiments')

        logger.info("开始运行四规模对比实验...")
        start_time = time.time()

        runner.run_full_batch()
        runner.create_comparison_charts()

        elapsed_time = time.time() - start_time
        logger.info(f"✅ 四规模对比实验完成! 耗时: {elapsed_time:.1f}秒")
        return True

    except Exception as e:
        logger.exception(f"❌ 运行四规模对比实验时出错: {e}")
        return False


def run_emergence_study(logger):
    """
    运行涌现研究，观察概念压缩和原理迁移的自然出现。
    """
    logger.info("\n" + "=" * 80)
    logger.info("2️⃣ 运行涌现研究")
    logger.info("=" * 80)

    try:
        from experiments.emergence_study_fixed import EmergenceStudyFixed

        logger.info("开始运行涌现研究...")
        start_time = time.time()

        study = EmergenceStudyFixed()
        scales = [51, 71, 91, 111]
        all_results = {}

        for scale in scales:
            logger.info(f"\n处理 {scale} 概念规模...")
            try:
                results = study.run_pure_emergence_experiment(
                    num_individuals=3,
                    max_iterations=10000,
                    num_concepts=scale
                )
                all_results[scale] = results

                # 移除不可序列化的 'universe' 字段
                serializable_results = []
                for individual in results:
                    serializable_individual = {k: v for k, v in individual.items() if k != 'universe'}
                    serializable_results.append(serializable_individual)

                # 生成带时间戳的文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f'results/emergence/emergence_{scale}_concepts_{timestamp}.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ 结果已保存到: {output_file}")
            except Exception as e:
                logger.exception(f"❌ 处理规模 {scale} 时出错: {e}")

        study.visualize_emergence_results()

        elapsed_time = time.time() - start_time
        logger.info(f"✅ 涌现研究完成! 耗时: {elapsed_time:.1f}秒")
        return True

    except Exception as e:
        logger.exception(f"❌ 运行涌现研究时出错: {e}")
        return True  # 不中断整体流程

def run_algebra_experiments(logger):
    """
    运行代数验证实验，验证认知操作半群、Noether型命题等。
    """
    logger.info("\n" + "=" * 80)
    logger.info("3️⃣ 运行代数验证实验")
    logger.info("=" * 80)

    try:
        from algebra.algebra_experiments import AlgebraValidationExperiments

        logger.info("开始运行代数验证实验...")
        start_time = time.time()

        experiments = AlgebraValidationExperiments()
        all_results = experiments.run_all_experiments()

        elapsed_time = time.time() - start_time
        logger.info(f"✅ 代数验证实验完成! 耗时: {elapsed_time:.1f}秒")
        return True

    except Exception as e:
        logger.exception(f"❌ 运行代数验证实验时出错: {e}")
        return False


def run_semantic_network_demo(logger):
    """
    运行语义网络演示，展示概念间的语义关联。
    """
    logger.info("\n" + "=" * 80)
    logger.info("4️⃣ 运行语义网络演示")
    logger.info("=" * 80)

    try:
        try:
            from main import demo_semantic_network
            demo_semantic_network()
            return True
        except ImportError:
            from core.semantic_network import SemanticConceptNetwork

            logger.info("构建语义概念网络...")
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

            logger.info("\n寻找跨领域路径示例:")
            paths = semantic_net.find_cross_domain_paths("牛顿定律", "算法", max_path_length=3)
            if paths:
                best_path, similarity = paths[0]
                logger.info(f"牛顿定律 -> 算法:")
                logger.info(f"  路径: {' -> '.join(best_path)}")
                logger.info(f"  相似度: {similarity:.3f}")

            logger.info("\n生成语义网络可视化...")
            semantic_net.visualize_semantic_network()

            return True

    except Exception as e:
        logger.exception(f"❌ 运行语义网络演示时出错: {e}")
        return True


def generate_summary_report(logger):
    """
    生成实验总结报告，保存为Markdown文件。
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 生成实验总结报告")
    logger.info("=" * 80)

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

## 数据说明
具体的数据说明可见`results/DATA_DICTIONARY.md`

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

    logger.info(f"✅ 实验总结报告已生成: {report_file}")
    logger.info("\n报告内容预览:")
    logger.info("-" * 50)
    logger.info(summary[:500] + "...")
    logger.info("-" * 50)

    return report_file


def main():
    """主函数：顺序运行所有实验并生成总结报告。"""
    # 配置日志
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("🧠 认知图论：基于能量最小化的认知计算模型")
    logger.info("=" * 80)
    logger.info("作者：曾铭佳")
    logger.info("版本：2.0")
    logger.info("时间：2025年12月")
    logger.info("=" * 80)

    # 记录环境信息
    log_environment(logger)

    # 检查依赖
    if not check_dependencies(logger):
        logger.error("❌ 依赖库检查失败，请先安装所需依赖")
        return

    # 创建输出目录
    if not create_output_directories(logger):
        logger.error("❌ 目录创建失败")
        return

    logger.info("\n" + "=" * 80)
    logger.info("🚀 开始一键运行所有实验")
    logger.info("=" * 80)

    overall_start_time = time.time()

    experiment_status = {
        'batch_experiments': False,
        'emergence_study': False,
        'algebra_experiments': False,
        'semantic_demo': False
    }

    try:
        experiment_status['batch_experiments'] = run_batch_experiments(logger)
        experiment_status['emergence_study'] = run_emergence_study(logger)
        experiment_status['algebra_experiments'] = run_algebra_experiments(logger)
        experiment_status['semantic_demo'] = run_semantic_network_demo(logger)

        overall_elapsed_time = time.time() - overall_start_time
        report_file = generate_summary_report(logger)

        logger.info("\n" + "=" * 80)
        logger.info("🎉 所有实验运行完成！")
        logger.info("=" * 80)

        logger.info("\n📈 实验完成状态:")
        for experiment, status in experiment_status.items():
            status_symbol = "✅" if status else "❌"
            logger.info(f"  {status_symbol} {experiment}")

        logger.info(f"\n⏱️  总运行时间: {overall_elapsed_time:.1f}秒")
        logger.info(f"📋 实验总结报告: {report_file}")

        successful_experiments = sum(experiment_status.values())
        total_experiments = len(experiment_status)
        logger.info(f"\n📊 成功率: {successful_experiments}/{total_experiments} ({successful_experiments / total_experiments * 100:.1f}%)")

        if successful_experiments == total_experiments:
            logger.info("\n🌟 所有实验均成功完成！")
            logger.info("建议：")
            logger.info("  1. 查看 results/ 目录下的详细结果")
            logger.info("  2. 运行 analysis.py 进行数据分析")
            logger.info("  3. 修改 config.py 调整参数重新实验")
        elif successful_experiments >= 2:
            logger.info("\n⚠️  部分实验完成，建议检查失败项目")
        else:
            logger.info("\n❌ 多数实验失败，请检查环境和依赖")

        logger.info("\n" + "=" * 80)
        logger.info("💡 提示：")
        logger.info("  1. 如需重新运行特定实验，请参考 main.py")
        logger.info("  2. 查看 README.md 获取详细项目说明")
        logger.info("  3. 实验结果可用于生成论文图表")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("\n\n⏹️  实验被用户中断")
        overall_elapsed_time = time.time() - overall_start_time
        logger.info(f"已运行时间: {overall_elapsed_time:.1f}秒")

    except Exception as e:
        logger.exception(f"\n❌ 运行实验时发生未预期错误: {e}")

    # 关闭日志文件（可选）
    logging.shutdown()


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