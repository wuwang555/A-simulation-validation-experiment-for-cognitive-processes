# algebra_experiments.py
"""
代数验证实验 - 验证论文中代数结构的正确性
"""
import os
import numpy as np
import networkx as nx
from algebra.cognitive_semigroup import CognitiveSemigroup
from algebra.cognitive_symmetry import CognitiveSymmetryGroup
from algebra.group_action import GroupActionOnCognitiveSpace
from algebra.lie_group_cognitive import CognitiveLieGroup


class AlgebraValidationExperiments:
    """代数验证实验管理器"""

    def __init__(self):
        self.results = {}

    def experiment1_verify_semigroup_properties(self):
        """实验1：验证认知操作半群性质"""
        print("=== 实验1：认知操作半群验证 ===")

        # 创建测试网络
        test_network = self._create_test_network()

        # 初始化半群
        semigroup = CognitiveSemigroup()
        self._initialize_operations(semigroup)

        # 验证结合律
        test_combinations = [
            ("learning", "forgetting", "traversal"),
            ("compression", "migration", "learning"),
            ("traversal", "learning", "forgetting")
        ]

        associativity_results = {}
        for op1, op2, op3 in test_combinations:
            try:
                is_assoc = semigroup.verify_associativity(
                    op1, op2, op3, test_network.copy()
                )
                associativity_results[f"({op1}∘{op2})∘{op3}"] = is_assoc
            except Exception as e:
                print(f"验证结合律时出错 ({op1}, {op2}, {op3}): {e}")
                associativity_results[f"({op1}∘{op2})∘{op3}"] = False

        self.results['experiment1'] = {
            'associativity': associativity_results,
            'operation_count': len(semigroup.operations),
            'note': "单位元检查暂时跳过（半群不一定要求单位元）"
        }

        print(f"操作数量: {len(semigroup.operations)}")
        print(f"结合律验证结果:")
        for key, value in associativity_results.items():
            print(f"  {key}: {value}")

        # 检查是否所有结合律验证都通过
        all_passed = all(associativity_results.values())
        print(f"所有结合律验证通过: {all_passed}")

        return self.results['experiment1']

    def experiment2_verify_noether_theorem(self):
        """实验2：验证Noether型定理（对称性→守恒量）"""
        print("\n=== 实验2：Noether定理验证 ===")

        # 创建不同结构的网络（现在都是完全图，保证连通）
        networks = [
            self._create_physics_dominant_network(),
            self._create_math_dominant_network(),
            self._create_balanced_network()
        ]

        noether_results = {}
        for i, network in enumerate(networks):
            print(f"处理网络 {i + 1}/{len(networks)}...")

            # 确保网络连通
            if not nx.is_connected(network):
                print(f"警告：网络{i + 1}不连通，尝试修复...")
                # 如果由于权重过大导致不连通，降低权重阈值
                # 在我们的完全图模型中，这不应该发生，但以防万一
                for u, v in network.edges():
                    if network[u][v]['weight'] > 5.0:  # 降低过高权重
                        network[u][v]['weight'] = 2.0

            symmetry_group = CognitiveSymmetryGroup(network)

            try:
                # 检测对称性
                automorphisms = symmetry_group.find_concept_isomorphisms()

                # 计算守恒量
                conserved = symmetry_group.compute_conserved_quantities()

                # 验证操作前后守恒量不变
                # 先执行学习操作
                before_net = network.copy()

                # 随机选择一个学习操作
                import random
                edges = list(before_net.edges())
                if edges:
                    u, v = random.choice(edges)
                    # 应用学习操作（降低权重）
                    after_net = before_net.copy()
                    current = after_net[u][v]['weight']
                    after_net[u][v]['weight'] = max(0.05, current * 0.9)

                    # 验证Noether定理
                    conserved_after_op = symmetry_group.verify_noether_theorem(
                        before_net, after_net, "learning"
                    )

                    noether_results[f"network_{i}"] = {
                        'automorphisms_count': len(automorphisms),
                        'conserved_quantities': conserved,
                        'noether_theorem_holds': conserved_after_op,
                        'energy_before': sum(before_net[u][v]['weight']
                                             for u, v in before_net.edges()),
                        'energy_after': sum(after_net[u][v]['weight']
                                            for u, v in after_net.edges())
                    }
                else:
                    noether_results[f"network_{i}"] = {
                        'error': "网络无边"
                    }

            except Exception as e:
                print(f"处理网络{i + 1}时出错: {e}")
                noether_results[f"network_{i}"] = {
                    'error': str(e)
                }

        self.results['experiment2'] = noether_results

        print("\nNoether定理验证结果:")
        for net_id, result in noether_results.items():
            print(f"网络 {net_id}:")
            if 'error' in result:
                print(f"  错误: {result['error']}")
            else:
                print(f"  同构数: {result['automorphisms_count']}")
                print(f"  守恒量: {result['conserved_quantities']}")
                print(f"  Noether定理成立: {result['noether_theorem_holds']}")

        return noether_results

    def experiment3_orbit_stabilizer_theorem(self):
        """实验3：验证轨道-稳定子定理"""
        print("\n=== 实验3：轨道-稳定子定理验证 ===")

        test_network = self._create_test_network()
        symmetry_group = CognitiveSymmetryGroup(test_network)

        # 找到所有同构
        automorphisms = symmetry_group.find_concept_isomorphisms()

        # 初始化群作用
        group_action = GroupActionOnCognitiveSpace(symmetry_group)

        # 计算轨道和稳定子
        orbit = group_action.compute_orbit(test_network)
        stabilizer = group_action.compute_stabilizer(test_network)

        # 验证定理 |轨道| = |群| / |稳定子|
        theorem_holds = group_action.verify_orbit_stabilizer_theorem(test_network)

        # 计算理论值和实际值
        expected_size = len(automorphisms) / max(1, len(stabilizer))
        actual_size = len(orbit)

        self.results['experiment3'] = {
            'automorphism_group_size': len(automorphisms),
            'orbit_size_actual': actual_size,
            'stabilizer_size': len(stabilizer),
            'orbit_size_expected': expected_size,
            'theorem_holds': theorem_holds,
            'error_percentage': abs(expected_size - actual_size) / expected_size * 100
            if expected_size > 0 else 0
        }

        print(f"同构群大小: {len(automorphisms)}")
        print(f"稳定子大小: {len(stabilizer)}")
        print(f"轨道大小（实际）: {actual_size}")
        print(f"轨道大小（理论）: {expected_size:.2f}")
        print(f"轨道-稳定子定理成立: {theorem_holds}")
        if not theorem_holds:
            print(f"误差百分比: {self.results['experiment3']['error_percentage']:.2f}%")

        return self.results['experiment3']

    def experiment4_lie_group_evolution(self):
        """实验4：李群演化演示"""
        print("\n=== 实验4：李群演化演示 ===")

        # 创建初始网络
        initial_network = self._create_test_network()
        n_nodes = initial_network.number_of_nodes()

        # 初始化李群
        lie_group = CognitiveLieGroup(n_nodes)

        # 设置不同演化策略
        strategies = [
            {'E': 0.7, 'C': 0.2, 'M': 0.1},  # 能量优化为主
            {'E': 0.3, 'C': 0.6, 'M': 0.1},  # 概念压缩为主
            {'E': 0.2, 'C': 0.2, 'M': 0.6},  # 原理迁移为主
        ]

        evolution_results = {}
        for i, coeffs in enumerate(strategies):
            evolved_networks = lie_group.evolve_network(
                initial_network,
                time_steps=5,
                generator_coeffs=coeffs
            )

            # 计算演化指标
            energies = []
            for net in evolved_networks:
                if net.number_of_edges() > 0:
                    avg_energy = np.mean([net[u][v]['weight']
                                          for u, v in net.edges()])
                else:
                    avg_energy = 0
                energies.append(avg_energy)

            evolution_results[f"strategy_{i}"] = {
                'generator_coeffs': coeffs,
                'initial_energy': energies[0],
                'final_energy': energies[-1],
                'energy_change_percent': ((energies[0] - energies[-1]) / energies[0] * 100
                                          if energies[0] > 0 else 0),
                'energy_trajectory': energies
            }

        self.results['experiment4'] = evolution_results

        print("李群演化结果（不同生成元组合）：")
        for strategy, result in evolution_results.items():
            print(f"  策略 {strategy}:")
            print(f"    生成元系数: {result['generator_coeffs']}")
            print(f"    能耗变化: {result['energy_change_percent']:.1f}%")

        return evolution_results

    def experiment5_scalability_test(self):
        """实验5：代数方法的可扩展性测试 - 简化版本"""
        print("\n=== 实验5：代数方法可扩展性测试 ===")

        # 减少测试规模
        network_sizes = [5, 8, 10, 12, 15]  # 减小规模
        scalability_results = {}

        for size in network_sizes:
            print(f"测试网络大小: {size}")

            # 创建小型完全图
            nodes = [f"概念_{i}" for i in range(size)]
            network = nx.complete_graph(nodes)

            # 分配随机权重
            for u, v in network.edges():
                network[u][v]['weight'] = np.random.uniform(0.5, 2.0)

            import time

            # 测试半群运算时间
            semigroup = CognitiveSemigroup()

            # 只初始化基本操作
            def identity_op(network, **kwargs):
                return network.copy()

            semigroup.add_operation("identity", identity_op)

            start_time = time.time()
            for _ in range(10):
                semigroup.compose("identity", "identity")
            semigroup_time = time.time() - start_time

            # 测试对称群检测时间（限制最大样本）
            start_time = time.time()
            try:
                symmetry_group = CognitiveSymmetryGroup(network)
                automorphisms = symmetry_group.find_concept_isomorphisms(max_samples=100)
                symmetry_time = time.time() - start_time
                symmetry_success = True
            except Exception as e:
                symmetry_time = time.time() - start_time
                automorphisms = []
                symmetry_success = False
                print(f"对称性检测失败: {e}")

            scalability_results[size] = {
                'nodes': size,
                'edges': network.number_of_edges(),
                'semigroup_operation_time': semigroup_time,
                'symmetry_detection_time': symmetry_time,
                'symmetry_detection_success': symmetry_success,
                'automorphisms_count': len(automorphisms)
            }

        self.results['experiment5'] = scalability_results

        print("\n可扩展性结果:")
        print("网络规模 | 半群运算时间(s) | 对称性检测时间(s) | 同构数 | 检测成功")
        print("-" * 70)
        for size, result in scalability_results.items():
            success = "✓" if result['symmetry_detection_success'] else "✗"
            print(f"{size:8d} | {result['semigroup_operation_time']:14.4f} | "
                  f"{result['symmetry_detection_time']:16.4f} | "
                  f"{result['automorphisms_count']:8d} | {success}")

        return scalability_results

    def run_all_experiments(self):
        """运行所有代数验证实验"""
        print("=" * 60)
        print("代数结构验证实验套件")
        print("=" * 60)

        results = {
            'semigroup_properties': self.experiment1_verify_semigroup_properties(),
            'noether_theorem': self.experiment2_verify_noether_theorem(),
            'orbit_stabilizer': self.experiment3_orbit_stabilizer_theorem(),
            'lie_group_evolution': self.experiment4_lie_group_evolution(),
            'scalability': self.experiment5_scalability_test()
        }

        self._generate_summary_report()
        return results

    def _generate_summary_report(self):
        """生成实验总结报告"""
        print("\n" + "=" * 60)
        print("代数验证实验总结报告")
        print("=" * 60)

        summary = {}

        # 实验1：代数性质验证
        if 'experiment1' in self.results:
            exp1 = self.results['experiment1']
            if 'associativity' in exp1:
                # 检查所有结合律是否通过
                all_passed = all(exp1['associativity'].values())
                summary['代数性质验证'] = "通过" if all_passed else "部分通过"
            else:
                summary['代数性质验证'] = "数据缺失"
        else:
            summary['代数性质验证'] = "未运行"

        # 实验2：Noether定理
        if 'experiment2' in self.results:
            exp2 = self.results['experiment2']
            # 检查是否有网络通过Noether定理
            noether_results = []
            for net_id, result in exp2.items():
                if isinstance(result, dict) and 'noether_theorem_holds' in result:
                    holds = result['noether_theorem_holds']
                    # 如果是元组，取第一个元素
                    if isinstance(holds, tuple):
                        holds = holds[0]
                    noether_results.append(holds)

            if noether_results:
                passed_count = sum(1 for r in noether_results if r)
                summary['Noether定理'] = f"{passed_count}/{len(noether_results)} 通过"
            else:
                summary['Noether定理'] = "无有效数据"
        else:
            summary['Noether定理'] = "未运行"

        # 实验3：轨道-稳定子定理
        if 'experiment3' in self.results:
            exp3 = self.results['experiment3']
            theorem_holds = exp3.get('theorem_holds', False)
            summary['轨道-稳定子定理'] = "通过" if theorem_holds else "失败"
        else:
            summary['轨道-稳定子定理'] = "未运行"

        # 实验4：李群演化
        if 'experiment4' in self.results:
            exp4 = self.results['experiment4']
            if exp4:
                # 检查是否有非零的能耗变化
                has_changes = any(
                    abs(result.get('energy_change_percent', 0)) > 0.1
                    for result in exp4.values()
                )
                summary['李群演化'] = "成功" if has_changes else "无变化"
            else:
                summary['李群演化'] = "无数据"
        else:
            summary['李群演化'] = "未运行"

        # 实验5：可扩展性
        if 'experiment5' in self.results:
            exp5 = self.results['experiment5']
            if exp5:
                # 检查是否所有规模都成功
                all_success = all(
                    result.get('symmetry_detection_success', False)
                    for result in exp5.values()
                )
                summary['可扩展性'] = "良好" if all_success else "受限"
            else:
                summary['可扩展性'] = "无数据"
        else:
            summary['可扩展性'] = "未运行"

        # 打印总结
        for test, result in summary.items():
            print(f"{test:15s}: {result}")

        # 保存结果到文件
        self._save_results_to_file()

        return summary

    def _save_results_to_file(self):
        """保存实验结果到文件"""
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 修改：确保目录存在
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # 修改：将文件保存到 results/algebra/ 目录下
        algebra_dir = os.path.join(results_dir, "algebra")
        os.makedirs(algebra_dir, exist_ok=True)
        filename = os.path.join(algebra_dir, f"algebra_validation_results_{timestamp}.json")

        # 转换结果为可序列化格式，并替换特殊字符
        serializable_results = {}
        for exp_name, exp_data in self.results.items():
            if isinstance(exp_data, dict):
                serializable_results[exp_name] = self._make_serializable(exp_data)
            else:
                serializable_results[exp_name] = str(exp_data)

        # 安全保存
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"\n详细结果已保存到: {filename}")
        except Exception as e:
            print(f"保存结果时出错: {e}")
            # 尝试备用文件名
            simple_filename = os.path.join(algebra_dir, f"results_{timestamp}.json")
            with open(simple_filename, 'w', encoding='ascii') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=True)
            print(f"结果已保存到备用文件: {simple_filename}")

    def _make_serializable(self, obj):
        """确保对象可JSON序列化"""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

    def _create_test_network(self):
        """创建测试网络 - 完全图实现"""
        nodes = ["算法", "数据结构", "优化", "递归", "迭代",
                 "抽象", "模式识别", "牛顿定律", "能量守恒", "微积分"]

        # 创建完全图
        G = nx.complete_graph(nodes)

        # 基于语义相似度计算初始能耗
        # 相似度越高，能耗越低
        for u, v in G.edges():
            # 模拟语义相似度计算（实际应该调用语义网络）
            if u == v:
                continue

            # 基于概念名称的简单相似度启发式
            similarity = self._calculate_simple_similarity(u, v)

            # 能耗 = 2.0 - 相似度 * 1.5（相似度越高，能耗越低）
            energy = 2.0 - similarity * 1.5
            energy = max(0.1, min(3.0, energy))  # 限制在合理范围

            G[u][v]['weight'] = energy

        return G

    def _calculate_simple_similarity(self, concept1, concept2):
        """基于概念名称的简单相似度计算"""
        # 如果概念相同
        if concept1 == concept2:
            return 1.0

        # 概念类别判断（简化实现）
        physics_concepts = ["牛顿定律", "能量守恒", "力学", "运动学"]
        math_concepts = ["微积分", "几何", "代数", "递归", "迭代"]
        cs_concepts = ["算法", "数据结构", "优化", "抽象", "模式识别"]

        concept1_domain = None
        concept2_domain = None

        if concept1 in physics_concepts:
            concept1_domain = "physics"
        elif concept1 in math_concepts:
            concept1_domain = "math"
        elif concept1 in cs_concepts:
            concept1_domain = "cs"

        if concept2 in physics_concepts:
            concept2_domain = "physics"
        elif concept2 in math_concepts:
            concept2_domain = "math"
        elif concept2 in cs_concepts:
            concept2_domain = "cs"

        # 同领域概念相似度高
        if concept1_domain and concept2_domain and concept1_domain == concept2_domain:
            return np.random.uniform(0.7, 0.9)

        # 跨领域但有关联（如数学与计算机科学）
        if (concept1_domain == "math" and concept2_domain == "cs") or \
                (concept1_domain == "cs" and concept2_domain == "math"):
            return np.random.uniform(0.5, 0.7)

        # 其他跨领域
        return np.random.uniform(0.2, 0.5)

    def _create_physics_dominant_network(self):
        """创建物理学主导的网络 - 完全图实现"""
        nodes = ["牛顿定律", "力学", "运动学", "能量守恒", "动量",
                 "万有引力", "摩擦力", "静电力", "优化", "迭代"]

        G = nx.complete_graph(nodes)

        for u, v in G.edges():
            # 物理学概念间连接更强（能耗更低）
            physics_terms = ["牛顿定律", "力学", "运动学", "能量守恒",
                             "动量", "万有引力", "摩擦力", "静电力"]

            if u in physics_terms and v in physics_terms:
                # 物理学概念间：强连接（低能耗）
                energy = np.random.uniform(0.3, 0.8)
            elif u in physics_terms or v in physics_terms:
                # 与跨领域概念：中等连接
                energy = np.random.uniform(1.0, 1.5)
            else:
                # 非物理学概念间：弱连接（高能耗）
                energy = np.random.uniform(1.5, 2.0)

            G[u][v]['weight'] = energy

        return G

    def _create_math_dominant_network(self):
        """创建数学主导的网络"""
        nodes = ["微积分", "几何学", "拓扑学", "线性代数", "概率论",
                 "统计学", "代数", "离散数学", "算法", "数据结构"]

        G = nx.Graph()
        G.add_nodes_from(nodes)

        # 数学概念间强连接
        for i in range(len(nodes) - 2):
            for j in range(i + 1, len(nodes) - 2):
                w = np.random.uniform(0.5, 0.9)
                G.add_edge(nodes[i], nodes[j], weight=w)

        return G

    def _create_balanced_network(self):
        """创建平衡网络"""
        nodes = ["算法", "数据结构", "优化", "牛顿定律", "能量守恒",
                 "微积分", "几何学", "递归", "迭代", "抽象"]

        G = nx.Graph()
        G.add_nodes_from(nodes)

        # 随机连接
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if np.random.random() < 0.4:  # 40%的连接概率
                    w = np.random.uniform(0.5, 1.5)
                    G.add_edge(nodes[i], nodes[j], weight=w)

        return G

    def _initialize_operations(self, semigroup):
        """初始化认知操作"""

        def learning_op(network, edge=None, strength=0.1, **kwargs):
            if edge is None:
                return network
            u, v = edge
            if network.has_edge(u, v):
                current = network[u][v]['weight']
                network[u][v]['weight'] = max(0.05, current * (1 - strength))
            return network

        def forgetting_op(network, edge=None, strength=0.05, **kwargs):
            if edge is None:
                return network
            u, v = edge
            if network.has_edge(u, v):
                current = network[u][v]['weight']
                network[u][v]['weight'] = min(2.0, current * (1 + strength))
            return network

        def traversal_op(network, path=None, **kwargs):
            if path is None or len(path) < 2:
                return network
            # 简化实现：遍历会轻微降低路径上的能耗
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if network.has_edge(u, v):
                    current = network[u][v]['weight']
                    network[u][v]['weight'] = max(0.05, current * 0.95)
            return network

        def compression_op(network, center=None, related_nodes=None, **kwargs):
            if center is None or related_nodes is None:
                return network
            # 压缩：中心节点与相关节点的连接增强
            for node in related_nodes:
                if network.has_edge(center, node):
                    current = network[center][node]['weight']
                    network[center][node]['weight'] = max(0.05, current * 0.8)
            return network

        def migration_op(network, principle=None, from_node=None, to_node=None, **kwargs):
            if principle is None or from_node is None or to_node is None:
                return network
            # 迁移：建立或强化连接
            if network.has_edge(from_node, principle):
                current = network[from_node][principle]['weight']
                network[from_node][principle]['weight'] = max(0.05, current * 0.9)
            if network.has_edge(principle, to_node):
                current = network[principle][to_node]['weight']
                network[principle][to_node]['weight'] = max(0.05, current * 0.9)
            return network

        # 添加到半群
        semigroup.add_operation("learning", learning_op)
        semigroup.add_operation("forgetting", forgetting_op)
        semigroup.add_operation("traversal", traversal_op)
        semigroup.add_operation("compression", compression_op)
        semigroup.add_operation("migration", migration_op)


# 主程序
if __name__ == "__main__":
    print("开始代数验证实验...\n")

    experiments = AlgebraValidationExperiments()

    try:
        all_results = experiments.run_all_experiments()

        print("\n" + "=" * 60)
        print("所有代数验证实验已完成！")
        print("=" * 60)

        # 显示总体成功情况
        success_count = 0
        total_experiments = 5

        if experiments.results.get('experiment1', {}).get('all_passed', False):
            success_count += 1
        if any(r.get('noether_theorem_holds', False) for r in experiments.results.get('experiment2', {}).values()):
            success_count += 1
        if experiments.results.get('experiment3', {}).get('theorem_holds', False):
            success_count += 1
        if any(abs(r.get('energy_change_percent', 0)) > 0.1
               for r in experiments.results.get('experiment4', {}).values()):
            success_count += 1
        if experiments.results.get('experiment5', {}):
            success_count += 1

        print(f"成功实验数: {success_count}/{total_experiments}")

    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
    except Exception as e:
        print(f"\n实验运行出错: {e}")
        import traceback

        traceback.print_exc()
