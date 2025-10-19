import numpy as np

def analyze_population_results(results):
    """分析群体结果"""
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



def get_network_stats(self):
    """获取网络统计信息"""
    stats = {
        'nodes': self.G.number_of_nodes(),
        'edges': self.G.number_of_edges(),
        'iterations': self.iteration_count,
        'avg_energy': self.get_average_energy(),
        'compression_centers': len(self.concept_centers),
        'migration_bridges': 0
    }

    for node in self.G.nodes():
        if 'migration_bridges' in self.G.nodes[node]:
            stats['migration_bridges'] += len(self.G.nodes[node]['migration_bridges'])

    return stats


def calculate_network_energy(G):
    """计算网络平均能耗"""
    if G.number_of_edges() == 0:
        return 0
    energies = [G[u][v]['weight'] for u, v in G.edges()]
    return np.mean(energies)
