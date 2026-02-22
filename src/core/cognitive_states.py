"""
认知状态模块
-------------
定义 CognitiveState 枚举及 CognitiveStateManager 类，管理认知状态的转移和主观能耗更新。
"""

import random
from enum import Enum
from collections import defaultdict
from typing import List, Dict, Any

# 从配置文件导入状态转移矩阵和能耗范围
from config import STATE_TRANSITION_MATRIX, STATE_ENERGY_RANGES


class CognitiveState(Enum):
    """认知状态枚举，对应四种典型认知状态。"""
    FOCUSED = "专注状态"
    EXPLORATORY = "探索状态"
    FATIGUED = "疲劳状态"
    INSPIRED = "灵感状态"


class CognitiveStateManager:
    """认知状态管理器，维护当前状态、主观能耗及历史记录。

    状态转移由马尔可夫链控制（转移矩阵从配置加载），每次状态变化后主观能耗在对应范围内随机更新。
    """

    def __init__(self):
        self.current_state = CognitiveState.FOCUSED
        self.subjective_energy = 1.5
        self.cognitive_energy_history: List[Dict[str, Any]] = []

    def update_cognitive_state(self) -> None:
        """根据状态转移矩阵更新认知状态。

        转移概率从配置中获取，随机决定下一状态。若状态改变，则调用 _update_subjective_energy。
        无论状态是否改变，都会记录当前状态和能耗至历史列表。
        """
        current_state_name = self.current_state.name
        transition_probs = STATE_TRANSITION_MATRIX[current_state_name]

        rand_val = random.random()
        cumulative_prob = 0

        for new_state_name, prob in transition_probs.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                new_state = CognitiveState[new_state_name]
                if new_state != self.current_state:
                    self.current_state = new_state
                    self._update_subjective_energy()
                break

        # 记录迭代次数（以历史长度作为迭代编号）
        iteration_count = len(self.cognitive_energy_history)
        self.cognitive_energy_history.append({
            'iteration': iteration_count,
            'state': self.current_state,
            'energy': self.subjective_energy
        })

    def _update_subjective_energy(self) -> None:
        """根据当前状态更新主观认知能耗。

        新能耗从状态对应的能耗范围（配置定义）中均匀采样，并限制在 [0.1, 3.0] 之间。
        """
        state_name = self.current_state.name
        energy_range = STATE_ENERGY_RANGES[state_name]
        self.subjective_energy = random.uniform(energy_range[0], energy_range[1])
        self.subjective_energy = max(0.1, min(3.0, self.subjective_energy))


if __name__ == "__main__":
    mgr = CognitiveStateManager()
    for _ in range(10):
        mgr.update_cognitive_state()
        print(f"状态: {mgr.current_state.value}, 能耗: {mgr.subjective_energy:.2f}")