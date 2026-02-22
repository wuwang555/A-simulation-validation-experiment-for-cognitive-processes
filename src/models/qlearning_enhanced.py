"""
增强的Q-learning模型 - 解决动态环境问题
采用更合理的状态表示和奖励机制，使智能体能够学习低能耗路径。
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, Any, List, Tuple
from core.cognitive_graph import BaseCognitiveGraph


class EnhancedQLearningCognitiveGraph(BaseCognitiveGraph):
    """增强的Q-learning认知图模型。

    该类将认知图建模为强化学习环境，智能体通过遍历网络学习最优路径，
    目标是最大化累积奖励（即最小化能耗）。
    """

    def __init__(self, individual_params: Dict[str, Any], network_seed: int = 42):
        """初始化增强Q-learning认知图。

        Args:
            individual_params (Dict[str, Any]): 个体参数。
            network_seed (int): 随机种子。
        """
        super().__init__(individual_params, network_seed)

        # Q-learning参数 - 调整以更好地学习
        self.learning_rate = 0.15  # 适中的学习率
        self.discount_factor = 0.85  # 降低折扣因子，更注重近期奖励
        self.exploration_rate = 0.25  # 较高的初始探索率
        self.min_exploration_rate = 0.05
        self.exploration_decay = 0.9995  # 非常缓慢的衰减

        # Q-table (状态数 × 动作数)
        # 状态：当前节点 + 认知状态（简化编码）
        self.q_table = None

        # 状态编码
        self.state_encoder = {}
        self.state_decoder = {}

        # 当前状态
        self.current_q_state = None
        self.current_cognitive_state = None

        # 性能跟踪
        self.episode_rewards = []
        self.best_paths = {}

    def initialize_network(self, num_nodes=51, connection_prob=0.2):
        """初始化网络。

        创建随机图并为边分配随机权重。

        Args:
            num_nodes (int): 节点数量。
            connection_prob (float): 边连接概率。
        """
        # 创建随机图
        self.G = nx.erdos_renyi_graph(num_nodes, connection_prob, seed=self.network_seed)

        # 为边分配随机权重
        for u, v in self.G.edges():
            weight = np.random.uniform(0.5, 1.5)  # 降低初始权重范围
            self.G[u][v]['weight'] = weight
            self.G[u][v]['original_weight'] = weight
            self.G[u][v]['traversal_count'] = 0
            self.last_activation_time[(u, v)] = 0

        # 为节点命名（使用简单编号，避免下划线问题）
        node_names = [f"概念{i}" for i in range(num_nodes)]  # 移除下划线
        mapping = {i: node_names[i] for i in range(num_nodes)}
        self.G = nx.relabel_nodes(self.G, mapping)

        # 初始化状态编码
        self._initialize_state_space()

        print(f"增强Q-learning网络初始化: {num_nodes}节点, {self.G.number_of_edges()}条边")
        print(f"状态空间大小: {len(self.state_encoder)}")

    def _initialize_state_space(self):
        """初始化状态空间。

        状态 = 节点 × 认知状态（探索/专注/疲劳），每个状态对应一个唯一的整数ID。
        """
        # 状态 = 节点 × 认知状态（简化）
        # 使用特殊分隔符 '|' 来避免与节点名称冲突
        nodes = list(self.G.nodes())
        cognitive_states = ['探索', '专注', '疲劳']  # 简化的认知状态

        state_id = 0
        for node in nodes:
            for state in cognitive_states:
                state_key = f"{node}|{state}"  # 使用 | 作为分隔符
                self.state_encoder[state_key] = state_id
                self.state_decoder[state_id] = (node, state)
                state_id += 1

        # 初始化Q-table
        n_states = len(self.state_encoder)
        n_actions = len(nodes)  # 动作是选择下一个节点
        self.q_table = np.zeros((n_states, n_actions))

        # 为有效的状态-动作对初始化较小的正值
        for state_key, state_id in self.state_encoder.items():
            # 使用正确的分隔符分割
            current_node, _ = state_key.split('|')
            for action_idx, action_node in enumerate(nodes):
                if self.G.has_edge(current_node, action_node):
                    # 有效连接初始化小的正值
                    self.q_table[state_id, action_idx] = 0.1

    def encode_state(self, node, cognitive_state):
        """编码状态为整数ID。

        Args:
            node (str): 当前节点。
            cognitive_state (str): 认知状态。

        Returns:
            int: 状态ID。
        """
        state_key = f"{node}|{cognitive_state}"  # 使用 | 作为分隔符
        return self.state_encoder.get(state_key, 0)

    def decode_state(self, state_id):
        """解码状态ID为节点和认知状态。

        Args:
            state_id (int): 状态ID。

        Returns:
            tuple: (node, cognitive_state)
        """
        return self.state_decoder.get(state_id, ("未知", "未知"))

    def get_cognitive_state_simplified(self):
        """简化的认知状态判断。

        基于当前迭代次数判断认知状态，用于状态空间。

        Returns:
            str: 认知状态，'探索'、'专注'或'疲劳'。
        """
        # 基于当前网络能耗和迭代次数判断认知状态
        if self.iteration_count < 1000:
            return "探索"
        elif self.iteration_count % 100 < 30:
            return "专注"
        else:
            return "疲劳"

    def get_possible_actions(self, current_node):
        """获取当前节点可能的动作（邻居节点）。

        Args:
            current_node (str): 当前节点。

        Returns:
            list: 每个元素为 (action_node, action_idx) 的列表。
        """
        nodes = list(self.G.nodes())

        # 如果当前节点有邻居，优先选择邻居
        neighbors = list(self.G.neighbors(current_node))
        if neighbors:
            return [(node, idx) for idx, node in enumerate(nodes) if node in neighbors]
        else:
            # 如果没有邻居，可以选择任意节点（但概率很低）
            return [(node, idx) for idx, node in enumerate(nodes) if node != current_node]

    def choose_action(self, state_id):
        """使用ε-greedy策略选择动作。

        Args:
            state_id (int): 当前状态ID。

        Returns:
            tuple: (action_node, action_idx)，若无可能动作则返回(None, None)。
        """
        current_node, cognitive_state = self.decode_state(state_id)

        possible_actions = self.get_possible_actions(current_node)
        if not possible_actions:
            return None, None

        # 探索：随机选择动作
        if random.random() < self.exploration_rate:
            action_node, action_idx = random.choice(possible_actions)
            return action_node, action_idx

        # 利用：选择Q值最高的动作
        else:
            # 只考虑可能的动作
            best_value = -float('inf')
            best_action = None
            best_idx = None

            for action_node, action_idx in possible_actions:
                if self.q_table[state_id, action_idx] > best_value:
                    best_value = self.q_table[state_id, action_idx]
                    best_action = action_node
                    best_idx = action_idx

            return best_action, best_idx

    def calculate_reward(self, current_node, action_node, cognitive_state):
        """计算奖励。

        奖励函数设计为鼓励低能耗路径，同时考虑认知状态和最近激活情况。
        对应于论文中能量函数 E_ij(t) 的负相关：低能耗对应高奖励。

        Args:
            current_node (str): 当前节点。
            action_node (str): 选择的动作节点。
            cognitive_state (str): 认知状态。

        Returns:
            float: 奖励值。
        """
        if not self.G.has_edge(current_node, action_node):
            return -2.0  # 无效连接的惩罚

        # 基础奖励：负的边权重（能耗越低奖励越高）
        weight = self.G[current_node][action_node]['weight']
        base_reward = 1.5 - weight  # 调整奖励范围到正值

        # 认知状态奖励调整
        state_bonus = 0
        if cognitive_state == "专注":
            state_bonus = 0.3
        elif cognitive_state == "探索":
            state_bonus = 0.1
        else:  # 疲劳
            state_bonus = -0.2

        # 学习进展奖励：如果这条边最近被学习过
        time_since_activation = self.iteration_count - self.last_activation_time.get((current_node, action_node), 0)
        recency_bonus = 0.2 if time_since_activation < 100 else 0

        total_reward = base_reward + state_bonus + recency_bonus

        return max(-1.0, total_reward)  # 确保奖励不低于-1.0

    def update_q_value(self, state_id, action_idx, reward, next_state_id):
        """使用贝尔曼方程更新Q值。

        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            state_id (int): 当前状态ID。
            action_idx (int): 动作索引。
            reward (float): 获得的奖励。
            next_state_id (int): 下一个状态ID。
        """
        current_q = self.q_table[state_id, action_idx]

        # 下一个状态的最大Q值
        next_max_q = np.max(self.q_table[next_state_id, :]) if np.any(self.q_table[next_state_id, :] != 0) else 0

        # 贝尔曼方程
        target_q = reward + self.discount_factor * next_max_q
        new_q = current_q + self.learning_rate * (target_q - current_q)

        # 限制Q值范围，避免过大或过小
        self.q_table[state_id, action_idx] = np.clip(new_q, -5.0, 5.0)

    def apply_learning_from_experience(self, current_node, action_node, reward):
        """从经验中学习，调整边权重（模拟Hebbian学习）。

        正奖励降低边权重（能耗），负奖励增加边权重。

        Args:
            current_node (str): 当前节点。
            action_node (str): 选择的动作节点。
            reward (float): 获得的奖励。
        """
        if not self.G.has_edge(current_node, action_node):
            return

        current_weight = self.G[current_node][action_node]['weight']

        # 基于奖励的学习：正奖励降低权重，负奖励增加权重
        learning_rate = 0.1
        weight_change = -learning_rate * reward  # 负相关：奖励越高，权重降低越多

        new_weight = max(0.1, min(2.0, current_weight + weight_change))
        self.G[current_node][action_node]['weight'] = new_weight

        # 更新激活时间
        self.last_activation_time[(current_node, action_node)] = self.iteration_count

    def qlearning_step(self):
        """执行一步Q-learning，包括动作选择、奖励计算和Q值更新。

        Returns:
            float: 获得的奖励。
        """
        if self.current_q_state is None:
            # 初始化状态
            random_node = random.choice(list(self.G.nodes()))
            cognitive_state = self.get_cognitive_state_simplified()
            self.current_q_state = self.encode_state(random_node, cognitive_state)
            self.current_cognitive_state = cognitive_state

        # 选择动作
        action_node, action_idx = self.choose_action(self.current_q_state)
        if action_node is None:
            return 0

        # 获取当前节点
        current_node, _ = self.decode_state(self.current_q_state)

        # 计算奖励
        reward = self.calculate_reward(current_node, action_node, self.current_cognitive_state)

        # 记录遍历
        self.traversal_history.append({
            'path': [current_node, action_node],
            'iteration': self.iteration_count,
            'reward': reward,
            'cognitive_state': self.current_cognitive_state
        })

        # 确定下一个状态
        next_cognitive_state = self.get_cognitive_state_simplified()
        next_state_id = self.encode_state(action_node, next_cognitive_state)

        # 更新Q值
        self.update_q_value(self.current_q_state, action_idx, reward, next_state_id)

        # 从经验中学习（调整边权重）
        self.apply_learning_from_experience(current_node, action_node, reward)

        # 更新当前状态
        self.current_q_state = next_state_id
        self.current_cognitive_state = next_cognitive_state

        return reward

    def apply_intelligent_forgetting(self):
        """智能遗忘机制，基于Q值和激活时间调整边权重。

        长时间未激活的边会逐渐恢复到原始权重，模拟遗忘过程。
        """
        current_time = self.iteration_count

        for u, v in self.G.edges():
            time_since_activation = current_time - self.last_activation_time.get((u, v), 0)

            if time_since_activation > 200:  # 长时间未激活
                # 编码状态（使用简化认知状态）
                cognitive_state = self.get_cognitive_state_simplified()
                state_id = self.encode_state(u, cognitive_state)

                # 找到对应的动作索引
                nodes = list(self.G.nodes())
                if v in nodes:
                    action_idx = nodes.index(v)

                    # 基于Q值的遗忘：Q值越低，遗忘越快
                    q_value = self.q_table[state_id, action_idx]
                    forget_factor = 0.1 * (2.0 - np.tanh(q_value))  # Q值越低，遗忘因子越大

                    current_weight = self.G[u][v]['weight']
                    original_weight = self.G[u][v].get('original_weight', 1.5)

                    # 向原始权重回归
                    new_weight = current_weight + (original_weight - current_weight) * forget_factor
                    new_weight = max(0.1, min(2.0, new_weight))

                    self.G[u][v]['weight'] = new_weight

    def find_best_path(self, start_node, end_node, max_length=5):
        """基于学习到的Q值寻找最优路径。

        Args:
            start_node (str): 起始节点。
            end_node (str): 目标节点。
            max_length (int): 最大路径长度。

        Returns:
            tuple: (路径列表, 总Q值)，若无路径则返回 (None, 0)。
        """
        if start_node not in self.G or end_node not in self.G:
            return None, 0

        nodes = list(self.G.nodes())

        # 初始化
        path = [start_node]
        current_node = start_node
        total_q_value = 0

        for step in range(max_length - 1):
            if current_node == end_node:
                break

            # 获取当前状态
            cognitive_state = self.get_cognitive_state_simplified()
            state_id = self.encode_state(current_node, cognitive_state)

            # 选择Q值最高的下一个节点（排除已访问节点）
            best_q = -float('inf')
            best_next = None

            for neighbor in self.G.neighbors(current_node):
                if neighbor in path:  # 避免循环
                    continue

                if neighbor in nodes:
                    action_idx = nodes.index(neighbor)
                    q_value = self.q_table[state_id, action_idx]

                    if q_value > best_q:
                        best_q = q_value
                        best_next = neighbor

            if best_next is None:
                break

            path.append(best_next)
            total_q_value += best_q
            current_node = best_next

        return path, total_q_value

    def enhanced_training(self, max_iterations=5000):
        """增强的Q-learning训练主循环。

        Args:
            max_iterations (int): 最大迭代次数。

        Returns:
            float: 能耗改善百分比。
        """
        print(f"开始增强Q-learning训练: {max_iterations}次迭代")

        initial_energy = self.calculate_network_energy()
        self.energy_history.append(initial_energy)

        total_reward = 0
        exploration_history = []

        for iteration in range(max_iterations):
            self.iteration_count += 1

            # 执行Q-learning步骤
            reward = self.qlearning_step()
            total_reward += reward

            # 定期应用智能遗忘
            if iteration % 100 == 0:
                self.apply_intelligent_forgetting()

            # 衰减探索率
            if iteration % 50 == 0:
                self.exploration_rate = max(self.min_exploration_rate,
                                          self.exploration_rate * self.exploration_decay)
                exploration_history.append(self.exploration_rate)

            # 记录能量
            if iteration % 10 == 0:
                current_energy = self.calculate_network_energy()
                self.energy_history.append(current_energy)

            # 定期报告
            if iteration % 500 == 0:
                current_energy = self.calculate_network_energy()
                improvement = ((initial_energy - current_energy) / initial_energy * 100) if initial_energy > 0 else 0
                avg_reward = total_reward / (iteration + 1) if iteration > 0 else 0

                # 显示探索进展
                explored_ratio = np.mean(self.q_table != 0)

                print(f"迭代 {iteration}: 能耗={current_energy:.3f} (改善:{improvement:.1f}%)")
                print(f"  探索率={self.exploration_rate:.3f}, 平均奖励={avg_reward:.3f}, Q表探索率={explored_ratio:.3f}")

                # 显示一个示例路径
                if iteration > 1000 and len(self.G.nodes()) >= 5:
                    nodes = list(self.G.nodes())
                    start = nodes[0]
                    end = nodes[4]
                    path, q_value = self.find_best_path(start, end)
                    if path and len(path) > 1:
                        print(f"  示例路径 {start}->{end}: {'->'.join(path)} (Q值:{q_value:.3f})")

        final_energy = self.calculate_network_energy()
        total_improvement = ((initial_energy - final_energy) / initial_energy * 100) if initial_energy > 0 else 0

        print(f"\n增强Q-learning训练完成!")
        print(f"初始能耗: {initial_energy:.3f}, 最终能耗: {final_energy:.3f}")
        print(f"总改善: {total_improvement:.1f}%")
        print(f"平均奖励: {total_reward / max_iterations:.3f}")
        print(f"最终探索率: {self.exploration_rate:.4f}")

        # 分析Q-table
        q_stats = self._analyze_q_table()
        print(f"Q-table统计: {q_stats}")

        return total_improvement

    def _analyze_q_table(self):
        """分析Q-table，返回统计信息。

        Returns:
            dict: Q-table统计信息。
        """
        if self.q_table is None:
            return {}

        flat_q = self.q_table.flatten()
        non_zero_q = flat_q[flat_q != 0]

        stats = {
            'size': self.q_table.shape,
            'total_entries': self.q_table.size,
            'non_zero_entries': len(non_zero_q),
            'sparsity': 1 - (len(non_zero_q) / self.q_table.size),
            'mean': np.mean(non_zero_q) if len(non_zero_q) > 0 else 0,
            'std': np.std(non_zero_q) if len(non_zero_q) > 0 else 0,
            'min': np.min(non_zero_q) if len(non_zero_q) > 0 else 0,
            'max': np.max(non_zero_q) if len(non_zero_q) > 0 else 0,
            'positive_ratio': np.sum(non_zero_q > 0) / len(non_zero_q) if len(non_zero_q) > 0 else 0
        }

        return stats

    def run_experiment(self, num_nodes=51, max_iterations=5000):
        """运行完整实验，包括网络初始化、训练和结果收集。

        Args:
            num_nodes (int): 节点数量。
            max_iterations (int): 最大迭代次数。

        Returns:
            dict: 实验结果字典。
        """
        self.initialize_network(num_nodes)
        improvement = self.enhanced_training(max_iterations)

        # 收集统计数据
        q_stats = self._analyze_q_table()

        # 测试最优路径查找
        path_examples = []
        if num_nodes >= 10:
            nodes = list(self.G.nodes())
            for i in range(3):
                start = nodes[i]
                end = nodes[i + 3]
                path, q_value = self.find_best_path(start, end)
                if path:
                    path_examples.append({
                        'start': start,
                        'end': end,
                        'path': '->'.join(path),
                        'q_value': q_value
                    })

        return {
            'model_type': 'enhanced_qlearning',
            'num_nodes': num_nodes,
            'iterations': max_iterations,
            'initial_energy': self.energy_history[0] if self.energy_history else 0,
            'final_energy': self.calculate_network_energy(),
            'improvement': improvement,
            'q_table_stats': q_stats,
            'exploration_rate_final': self.exploration_rate,
            'path_examples': path_examples,
            'network_stats': self.get_network_stats()
        }


if __name__ == "__main__":
    # 简单测试：创建一个小型网络并运行短时间训练
    print("测试 EnhancedQLearningCognitiveGraph...")
    params = {}  # 空参数，使用默认值
    model = EnhancedQLearningCognitiveGraph(params)
    result = model.run_experiment(num_nodes=51, max_iterations=8000)
    print("测试完成。改善率:", result['improvement'])