from enum import Enum
import random
from config import STATE_TRANSITION_MATRIX, STATE_ENERGY_RANGES

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

    def update_cognitive_state(self):
        """更新认知状态"""
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

        # 记录迭代次数
        iteration_count = len(self.cognitive_energy_history)
        self.cognitive_energy_history.append({
            'iteration': iteration_count,
            'state': self.current_state,
            'energy': self.subjective_energy
        })

    def _update_subjective_energy(self):
        """根据当前状态更新主观认知能耗"""
        state_name = self.current_state.name
        energy_range = STATE_ENERGY_RANGES[state_name]
        self.subjective_energy = random.uniform(energy_range[0], energy_range[1])
        self.subjective_energy = max(0.1, min(3.0, self.subjective_energy))