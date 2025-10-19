import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from core.cognitive_states import CognitiveState


def visualize_energy_convergence(energy_history, concept_centers):
    """可视化能耗收敛过程"""
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history, 'b-', alpha=0.7, linewidth=1)
    plt.xlabel('迭代次数')
    plt.ylabel('平均认知能耗')
    plt.title('认知能耗收敛过程')
    plt.grid(True, alpha=0.3)

    colors = ['red', 'green', 'orange', 'purple']
    for i, (center, info) in enumerate(concept_centers.items()):
        iteration = info['iteration']
        if iteration < len(energy_history):
            color = colors[i % len(colors)]
            plt.axvline(x=iteration, color=color, alpha=0.5, linestyle='--',
                        label=f'压缩: {center}' if i < 4 else "")

    if len(concept_centers) > 0:
        plt.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


def visualize_cognitive_states(cognitive_energy_history, energy_history):
    """可视化认知状态变化"""
    if not cognitive_energy_history:
        return

    # 添加迭代次数到认知能量历史中（如果没有的话）
    for i, entry in enumerate(cognitive_energy_history):
        if 'iteration' not in entry:
            entry['iteration'] = i

    iterations = [e['iteration'] for e in cognitive_energy_history]
    energies = [e['energy'] for e in cognitive_energy_history]

    # 确保网络能量历史长度匹配
    if len(energy_history) > len(iterations):
        network_energies = energy_history[:len(iterations)]
    else:
        network_energies = energy_history + [energy_history[-1]] * (
                    len(iterations) - len(energy_history)) if energy_history else []

    states = [e['state'] for e in cognitive_energy_history]

    state_colors = {
        CognitiveState.FOCUSED: 'green',
        CognitiveState.EXPLORATORY: 'blue',
        CognitiveState.FATIGUED: 'red',
        CognitiveState.INSPIRED: 'purple'
    }

    colors = [state_colors[state] for state in states]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.scatter(iterations, energies, c=colors, alpha=0.6)
    plt.plot(iterations, energies, 'gray', alpha=0.3)
    plt.ylabel('主观认知能耗')
    plt.title('主观认知状态与能耗变化')

    for state, color in state_colors.items():
        plt.plot([], [], 'o', color=color, label=state.value)
    plt.legend()

    plt.subplot(2, 1, 2)
    if network_energies:
        plt.plot(iterations, network_energies, 'b-', alpha=0.7)
    plt.xlabel('迭代次数')
    plt.ylabel('网络平均能耗')
    plt.title('认知网络能耗演化')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    state_counts = {}
    for state in states:
        state_counts[state] = state_counts.get(state, 0) + 1

    print("\n=== 认知状态统计 ===")
    for state, count in state_counts.items():
        percentage = (count / len(states)) * 100
        print(f"{state.value}: {count}次 ({percentage:.1f}%)")


def visualize_graph(G, concept_centers, title="认知图", figsize=(12, 8)):
    """可视化认知图"""
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in concept_centers:
            node_colors.append('red')
            node_sizes.append(2000)
        elif any('migration_bridges' in G.nodes[n] for n in G.nodes()):
            node_colors.append('orange')
            node_sizes.append(1500)
        else:
            node_colors.append('lightblue')
            node_sizes.append(800)

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        energy = G[u][v]['weight']
        edge_widths.append(max(0.5, 3 - energy * 1.5))

        if energy < 0.3:
            edge_colors.append('green')
        elif energy < 0.7:
            edge_colors.append('blue')
        else:
            edge_colors.append('gray')

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           alpha=0.6, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, font_size=8,
                            font_family='SimHei')

    plt.title(title, fontsize=16, fontfamily='SimHei')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 计算统计信息
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    avg_energy = np.mean([G[u][v]['weight'] for u, v in G.edges()]) if edges > 0 else 0
    migration_bridges = 0
    for node in G.nodes():
        if 'migration_bridges' in G.nodes[node]:
            migration_bridges += len(G.nodes[node]['migration_bridges'])

    print(f"网络统计:")
    print(f"  节点: {nodes}, 边: {edges}")
    print(f"  平均能耗: {avg_energy:.3f}")
    print(f"  概念压缩中心: {len(concept_centers)}")
    print(f"  迁移桥梁: {migration_bridges}")


def visualize_semantic_network(semantic_network, concept_definitions=None, highlight_concepts=None):
    """可视化语义网络"""
    try:
        G = nx.Graph()

        for concept in semantic_network:
            domain = get_domain(concept, concept_definitions) if concept_definitions else "other"
            G.add_node(concept, domain=domain)

        for concept1, neighbors in semantic_network.items():
            for concept2, similarity in neighbors.items():
                if similarity > 0.1:
                    G.add_edge(concept1, concept2, weight=similarity)

        domain_colors = {
            "physics": "lightblue",
            "math": "lightgreen",
            "cs": "lightcoral",
            "principles": "gold",
            "other": "lightgray"
        }

        node_colors = [domain_colors[G.nodes[node].get('domain', 'other')] for node in G.nodes()]

        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=3, iterations=50)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)

        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color='gray')

        nx.draw_networkx_labels(G, pos, font_size=8, font_family='SimHei')

        plt.title("语义概念网络", fontsize=16, fontfamily='SimHei')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("需要安装networkx和matplotlib来可视化网络")


def get_domain(concept, concept_definitions=None):
    """获取概念所属领域"""
    domains = {
        "physics": ["牛顿定律", "力学", "运动学", "能量守恒", "动量", "万有引力", "摩擦力", "静电力"],
        "math": ["微积分", "几何学", "拓扑学", "线性代数", "概率论", "统计学", "代数", "离散数学"],
        "cs": ["算法", "数据结构", "机器学习", "神经网络", "计算机视觉", "自然语言处理", "数据库", "操作系统"],
        "principles": ["优化", "变换", "迭代", "抽象", "模式识别", "对称", "递归", "归纳"]
    }

    for domain, concepts in domains.items():
        if concept in concepts:
            return domain
    return "other"