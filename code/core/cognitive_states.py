from enum import Enum
from typing import Dict, Any
import random

class CognitiveState(Enum):
    """认知状态枚举"""
    FOCUSED = "专注状态"
    EXPLORATORY = "探索状态"
    FATIGUED = "疲劳状态"
    INSPIRED = "灵感状态"


class CognitiveStateManager:
    """认知状态管理器"""

    def __init__(self):
        self.current_state = CognitiveState.FOCUSED
        self.subjective_energy = 1.5
        self.cognitive_energy_history = []

        # 状态转移矩阵
        self.state_transition_matrix = {
            CognitiveState.FOCUSED: {
                CognitiveState.EXPLORATORY: 0.3,
                CognitiveState.FATIGUED: 0.1,
                CognitiveState.INSPIRED: 0.2,
                CognitiveState.FOCUSED: 0.4
            },
            CognitiveState.EXPLORATORY: {
                CognitiveState.FOCUSED: 0.3,
                CognitiveState.FATIGUED: 0.2,
                CognitiveState.INSPIRED: 0.2,
                CognitiveState.EXPLORATORY: 0.3
            },
            CognitiveState.FATIGUED: {
                CognitiveState.FOCUSED: 0.3,
                CognitiveState.EXPLORATORY: 0.4,
                CognitiveState.INSPIRED: 0.1,
                CognitiveState.FATIGUED: 0.2
            },
            CognitiveState.INSPIRED: {
                CognitiveState.FOCUSED: 0.4,
                CognitiveState.EXPLORATORY: 0.3,
                CognitiveState.FATIGUED: 0.1,
                CognitiveState.INSPIRED: 0.2
            }
        }

        # 状态能耗范围
        self.state_energy_ranges = {
            CognitiveState.FOCUSED: (1.5, 2.5),
            CognitiveState.EXPLORATORY: (1.0, 1.8),
            CognitiveState.FATIGUED: (0.8, 1.2),
            CognitiveState.INSPIRED: (2.0, 3.0)
        }

    def update_cognitive_state(self):
        """更新认知状态"""
        current_state = self.current_state
        transition_probs = self.state_transition_matrix[current_state]

        rand_val = random.random()
        cumulative_prob = 0

        for new_state, prob in transition_probs.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                if new_state != current_state:
                    self.current_state = new_state
                    self._update_subjective_energy()
                break

        # 记录迭代次数
        iteration_count = len(self.cognitive_energy_history)
        self.cognitive_energy_history.append({
            'iteration': iteration_count,  # 添加迭代次数
            'state': self.current_state,
            'energy': self.subjective_energy
        })


    def _update_subjective_energy(self):
        """根据当前状态更新主观认知能耗"""
        energy_range = self.state_energy_ranges[self.current_state]
        self.subjective_energy = random.uniform(energy_range[0], energy_range[1])
        self.subjective_energy = max(0.1, min(3.0, self.subjective_energy))
