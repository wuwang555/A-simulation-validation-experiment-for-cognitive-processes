import numpy as np
from typing import Dict, Any
import random

from config import BASE_PARAMETERS, VARIATION_RANGES


class IndividualVariation:
    """个体差异模拟器"""

    def __init__(self, base_parameters: Dict[str, Any] = None,
                 variation_ranges: Dict[str, Any] = None):
        self.base_parameters = base_parameters or BASE_PARAMETERS
        self.variation_ranges = variation_ranges or VARIATION_RANGES
        self.individual_parameters = {}

    def generate_individual(self, individual_id: str):
        """为单个个体生成参数"""
        params = {}
        for param, base_value in self.base_parameters.items():
            if param in self.variation_ranges:
                variation = self.variation_ranges[param]
                if isinstance(variation, (int, float)):
                    min_val = base_value * (1 - variation)
                    max_val = base_value * (1 + variation)
                    params[param] = np.random.uniform(min_val, max_val)
                elif isinstance(variation, tuple) and len(variation) == 2:
                    params[param] = np.random.uniform(variation[0], variation[1])
                else:
                    params[param] = base_value
            else:
                params[param] = base_value

        self.individual_parameters[individual_id] = params
        return params


def create_enhanced_individual_params(base_params):
    """创建增强的个体参数（包含认知状态相关参数）"""
    enhanced_params = base_params.copy()

    enhanced_params.update({
        'energy_variation': random.uniform(0.05, 0.15),
        'focus_bias': random.uniform(-0.1, 0.1),
        'exploration_bias': random.uniform(-0.1, 0.1),
        'fatigue_resistance': random.uniform(0.1, 0.3),
        'inspiration_frequency': random.uniform(0.05, 0.2)
    })

    return enhanced_params
