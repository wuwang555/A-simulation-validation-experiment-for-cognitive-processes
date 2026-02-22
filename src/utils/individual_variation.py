"""
个体差异模拟模块。

提供生成具有不同认知参数的个体，以模拟认知风格的个体差异。
"""

import numpy as np
from typing import Dict, Any
import random
from config import BASE_PARAMETERS, VARIATION_RANGES


class IndividualVariation:
    """
    个体差异模拟器。

    基于基础参数和变异范围，为每个个体生成个性化参数集，
    用于模拟认知风格的多样性（如学习率、遗忘率、探索倾向等）。

    Attributes
    ----------
    base_parameters : dict
        基础参数值。
    variation_ranges : dict
        各参数的变异范围。
    individual_parameters : dict
        存储已生成个体的参数，键为个体ID，值为参数字典。
    """

    def __init__(self, base_parameters: Dict[str, Any] = None,
                 variation_ranges: Dict[str, Any] = None):
        """
        初始化个体差异模拟器。

        Parameters
        ----------
        base_parameters : dict, optional
            基础参数，默认使用 config.BASE_PARAMETERS。
        variation_ranges : dict, optional
            变异范围，默认使用 config.VARIATION_RANGES。
        """
        self.base_parameters = base_parameters or BASE_PARAMETERS
        self.variation_ranges = variation_ranges or VARIATION_RANGES
        self.individual_parameters = {}

    def generate_individual(self, individual_id: str):
        """
        为单个个体生成个性化参数。

        Parameters
        ----------
        individual_id : str
            个体唯一标识符。

        Returns
        -------
        dict
            包含该个体所有参数的字典。
        """
        params = {}
        for param, base_value in self.base_parameters.items():
            if param in self.variation_ranges:
                variation = self.variation_ranges[param]
                if isinstance(variation, (int, float)):
                    # 相对比例变异
                    min_val = base_value * (1 - variation)
                    max_val = base_value * (1 + variation)
                    params[param] = np.random.uniform(min_val, max_val)
                elif isinstance(variation, tuple) and len(variation) == 2:
                    # 绝对范围变异
                    params[param] = np.random.uniform(variation[0], variation[1])
                else:
                    params[param] = base_value
            else:
                params[param] = base_value

        self.individual_parameters[individual_id] = params
        return params


def create_enhanced_individual_params(base_params):
    """
    创建增强的个体参数，添加认知状态相关的额外参数。

    Parameters
    ----------
    base_params : dict
        基础参数字典（由 IndividualVariation 生成）。

    Returns
    -------
    dict
        扩展后的参数字典，包含认知状态相关参数。
    """
    enhanced_params = base_params.copy()

    enhanced_params.update({
        'energy_variation': random.uniform(0.05, 0.15),
        'focus_bias': random.uniform(-0.1, 0.1),
        'exploration_bias': random.uniform(-0.1, 0.1),
        'fatigue_resistance': random.uniform(0.1, 0.3),
        'inspiration_frequency': random.uniform(0.05, 0.2)
    })

    return enhanced_params


if __name__ == "__main__":
    # 简单测试：生成一个个体并查看参数
    var = IndividualVariation()
    params = var.generate_individual("test_001")
    enhanced = create_enhanced_individual_params(params)
    print("基础参数:", params)
    print("增强参数:", enhanced)