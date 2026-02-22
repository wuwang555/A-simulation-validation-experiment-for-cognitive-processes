"""
认知图论结果分析模块。

提供对实验结果的统计分析和网络指标计算功能。
"""

import numpy as np


def analyze_population_results(results):
    """
    分析群体实验结果，计算并打印统计信息。

    Parameters
    ----------
    results : list of dict
        每个个体的实验结果列表，每个字典应包含以下键：
            - 'improvement' : float  # 能耗改善百分比
            - 'compression_centers' : list  # 概念压缩中心列表
            - 'migration_bridges' : list  # 迁移桥梁列表

    Returns
    -------
    dict
        包含以下统计指标的字典：
            - 'mean_improvement' : float   # 平均能耗改善
            - 'std_improvement' : float    # 能耗改善标准差
            - 'mean_compressions' : float  # 平均概念压缩数
            - 'mean_migrations' : float    # 平均迁移数
    """
    print(f"\n=== 群体统计结果 ===")

    improvements = [r['improvement'] for r in results]
    compressions = [r['compression_centers'] for r in results]
    migrations = [r['migration_bridges'] for r in results]

    print(f"能耗降低统计:")
    print(f"  平均: {np.mean(improvements):.1f}%")
    print(f"  标准差: {np.std(improvements):.1f}%")
    print(f"  范围: {min(improvements):.1f}% - {max(improvements):.1f}%")

    print(f"概念压缩统计:")
    print(f"  平均: {np.mean(compressions):.1f}个")
    print(f"  范围: {min(compressions)} - {max(compressions)}个")

    print(f"迁移桥梁统计:")
    print(f"  平均: {np.mean(migrations):.1f}个")
    print(f"  范围: {min(migrations)} - {max(migrations)}个")

    print(f"\n=== 个体差异分析 ===")

    best_individual = max(results, key=lambda x: x['improvement'])
    worst_individual = min(results, key=lambda x: x['improvement'])

    print(f"最优个体: {best_individual['individual_id']} (能耗降低: {best_individual['improvement']:.1f}%)")
    print(f"最差个体: {worst_individual['individual_id']} (能耗降低: {worst_individual['improvement']:.1f}%)")

    return {
        'mean_improvement': np.mean(improvements),
        'std_improvement': np.std(improvements),
        'mean_compressions': np.mean(compressions),
        'mean_migrations': np.mean(migrations)
    }


def get_network_stats(G, iteration_count, concept_centers):
    """
    获取认知网络的统计信息。

    Parameters
    ----------
    G : networkx.Graph
        认知网络图对象。
    iteration_count : int
        当前迭代次数。
    concept_centers : dict
        概念压缩中心字典，键为节点名，值为压缩信息。

    Returns
    -------
    dict
        包含以下键的字典：
            - 'nodes' : int                     # 节点数
            - 'edges' : int                      # 边数
            - 'iterations' : int                  # 迭代次数
            - 'avg_energy' : float                 # 平均能耗
            - 'compression_centers' : int          # 压缩中心数量
            - 'migration_bridges' : int            # 迁移桥梁总数
    """
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'iterations': iteration_count,
        'avg_energy': calculate_network_energy(G),
        'compression_centers': len(concept_centers),
        'migration_bridges': 0
    }

    for node in G.nodes():
        if 'migration_bridges' in G.nodes[node]:
            stats['migration_bridges'] += len(G.nodes[node]['migration_bridges'])

    return stats


def calculate_network_energy(G):
    """
    计算网络平均能耗。

    Parameters
    ----------
    G : networkx.Graph
        认知网络图对象，边应包含 'weight' 属性表示能耗。

    Returns
    -------
    float
        所有边的平均能耗；若无边则返回 0。
    """
    if G.number_of_edges() == 0:
        return 0
    energies = [G[u][v]['weight'] for u, v in G.edges()]
    return np.mean(energies)


if __name__ == "__main__":
    # 简单测试：创建模拟数据并调用分析函数
    mock_results = [
        {'individual_id': 'ind1', 'improvement': 23.5, 'compression_centers': [1, 2], 'migration_bridges': [1]},
        {'individual_id': 'ind2', 'improvement': 18.2, 'compression_centers': [3], 'migration_bridges': []},
    ]
    analyze_population_results(mock_results)