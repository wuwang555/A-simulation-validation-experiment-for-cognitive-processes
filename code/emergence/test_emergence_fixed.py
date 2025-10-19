# test_emergence_fixed.py
from emergence.universe_enhanced import CognitiveUniverseEnhanced
from emergence.detector_fixed import EmergenceDetectorFixed
import matplotlib.pyplot as plt


def test_enhanced_emergence():
    """测试修复后的涌现检测"""
    print("=== 测试修复的涌现检测 ===")

    # 创建增强宇宙
    universe = CognitiveUniverseEnhanced()
    universe.initialize_semantic_network()

    print(f"初始网络: {universe.G.number_of_nodes()}节点, {universe.G.number_of_edges()}边")
    print(f"初始能量: {universe.calculate_network_energy():.3f}")

    # 运行带检测的演化
    observations = universe.evolve_with_emergence_detection(
        iterations=2000,
        detection_interval=100
    )

    # 报告结果
    print("\n" + "=" * 60)
    print("涌现检测结果报告")
    print("=" * 60)

    compressions = observations['natural_compressions']
    migrations = observations['natural_migrations']

    print(f"概念压缩事件: {len(compressions)}")
    for i, comp in enumerate(compressions[:5]):  # 显示前5个
        print(f"  {i + 1}. 中心: {comp['center']}")
        print(f"     相关节点: {comp['related_nodes'][:3]}...")  # 显示前3个
        print(f"     强度: {comp['emergence_strength']:.3f}")
        print(f"     检测迭代: {comp['detection_iteration']}")

    print(f"\n原理迁移事件: {len(migrations)}")
    for i, mig in enumerate(migrations[:5]):
        print(f"  {i + 1}. 原理: {mig['principle_node']}")
        print(f"     路径: {mig['from_node']} -> {mig['to_node']}")
        print(f"     效率: {mig['efficiency_gain']:.3f}")
        print(f"     检测迭代: {mig['detection_iteration']}")

    # 可视化
    visualize_emergence_results(universe, observations)


def visualize_emergence_results(universe, observations):
    """可视化涌现结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 能量收敛曲线
    ax1.plot(universe.energy_history, 'b-', alpha=0.7)
    ax1.set_title('能量收敛过程')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('平均认知能耗')
    ax1.grid(True, alpha=0.3)

    # 标记涌现事件
    compressions = observations['natural_compressions']
    migrations = observations['natural_migrations']

    for comp in compressions:
        iter_num = comp['detection_iteration']
        if iter_num < len(universe.energy_history):
            ax1.axvline(x=iter_num, color='red', alpha=0.5, linestyle='--',
                        label='压缩' if comp == compressions[0] else "")

    for mig in migrations:
        iter_num = mig['detection_iteration']
        if iter_num < len(universe.energy_history):
            ax1.axvline(x=iter_num, color='green', alpha=0.5, linestyle=':',
                        label='迁移' if mig == migrations[0] else "")

    if compressions or migrations:
        ax1.legend()

    # 2. 涌现事件时间线
    events = []
    for comp in compressions:
        events.append((comp['detection_iteration'], '压缩', comp['center']))
    for mig in migrations:
        events.append((mig['detection_iteration'], '迁移', mig['principle_node']))

    if events:
        events.sort()
        iterations, types, nodes = zip(*events)
        colors = ['red' if t == '压缩' else 'green' for t in types]
        ax2.scatter(iterations, range(len(iterations)), c=colors, alpha=0.6)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('涌现事件序号')
        ax2.set_title('涌现事件时间线')
        ax2.grid(True, alpha=0.3)

    # 3. 涌现强度分布
    if compressions:
        strengths = [comp['emergence_strength'] for comp in compressions]
        ax3.hist(strengths, bins=10, alpha=0.7, color='red')
        ax3.set_xlabel('涌现强度')
        ax3.set_ylabel('频率')
        ax3.set_title('概念压缩强度分布')

    # 4. 迁移效率分布
    if migrations:
        efficiencies = [mig['efficiency_gain'] for mig in migrations]
        ax4.hist(efficiencies, bins=10, alpha=0.7, color='green')
        ax4.set_xlabel('迁移效率')
        ax4.set_ylabel('频率')
        ax4.set_title('原理迁移效率分布')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_enhanced_emergence()