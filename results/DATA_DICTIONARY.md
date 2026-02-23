# 数据字典 (Data Dictionary)

本数据字典详细说明认知图论实验所生成的所有数据文件的字段含义、数据类型、单位及取值来源。文件按类别分组，每个文件中的字段均基于论文《认知的几何、代数与动力学：基于能耗最小化原理的统一图论模型》中的定义及实验代码的实际输出进行解释。

---

## 1. 代数验证实验

### 1.1 `algebra_validation_results_*.json`

**描述**：该文件记录了第5节“代数验证实验”中五组实验的完整结果，包括半群结合律验证、Noether型命题验证、轨道-稳定子定理验证、李群演化演示和代数方法可扩展性测试。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `experiment1` | 实验1：认知操作半群验证结果 | object | - | 包含结合律验证的三个案例 |
| `experiment1.associativity` | 结合律验证的具体案例 | object | - | 键为操作组合，值为布尔值（True表示满足结合律） |
| `experiment1.operation_count` | 参与验证的基本操作数量 | integer | - | 实验中定义了13种基本操作 |
| `experiment1.note` | 备注说明 | string | - | 说明单位元检验暂时跳过（半群不要求单位元） |
| `experiment2` | 实验2：Noether型命题验证结果 | object | - | 包含三个不同结构网络的验证数据 |
| `experiment2.network_*` | 每个网络的验证结果 | object | - | 网络索引0,1,2分别对应5、10、15节点网络 |
| `network_*.automorphisms_count` | 网络的自同构数量 | integer | - | 实验中均为0，反映认知网络的低对称性 |
| `network_*.conserved_quantities` | 守恒量（演化前的值） | object | - | 包含total_energy, structural_entropy, fractal_dimension |
| `network_*.noether_theorem_holds` | Noether命题是否成立 | [bool, object] | - | 第一个值为整体布尔结论；第二个对象给出每个守恒量在演化前后的相对变化及守恒性判断 |
| `network_*.energy_before` | 演化前的全局认知能量 | float | - | 无单位，相对值 |
| `network_*.energy_after` | 演化后的全局认知能量 | float | - | 同上 |
| `experiment3` | 实验3：轨道-稳定子定理验证 | object | - | 包含对称群大小、稳定子大小、轨道大小及定理成立判断 |
| `experiment3.automorphism_group_size` | 自同构群的大小 | integer | - | 实验中为0 |
| `experiment3.orbit_size_actual` | 实际轨道大小 | integer | - | 实验中为0 |
| `experiment3.stabilizer_size` | 稳定子大小 | integer | - | 实验中为0 |
| `experiment3.orbit_size_expected` | 根据定理预期的轨道大小 | float | - | 0.0 |
| `experiment3.theorem_holds` | 定理是否成立 | string | - | "True" |
| `experiment3.error_percentage` | 误差百分比 | float | - | 0 |
| `experiment4` | 实验4：李群演化策略演示 | object | - | 包含三种不同生成元组合的演化结果 |
| `experiment4.strategy_*` | 每种策略的详细结果 | object | - | 策略0:能量优化主导；策略1:概念压缩主导；策略2:原理迁移主导 |
| `strategy_*.generator_coeffs` | 李代数生成元的系数 | object | - | 键E(能量优化)、C(概念压缩)、M(原理迁移)，值对应系数 |
| `strategy_*.initial_energy` | 初始认知能耗 | float | - | 相对值 |
| `strategy_*.final_energy` | 最终认知能耗 | float | - | 相对值 |
| `strategy_*.energy_change_percent` | 能耗变化百分比 | float | % | 负值表示降低 |
| `strategy_*.energy_trajectory` | 演化过程中各时间步的能耗 | list of float | - | 6个时间点的能耗值 |
| `experiment5` | 实验5：代数方法可扩展性测试 | object | - | 键为网络规模（节点数），值为对应结果 |
| `experiment5.<n>` | 节点数为n的网络的测试结果 | object | - | n = 5,8,10,12,15 |
| `<n>.nodes` | 节点数 | integer | - | |
| `<n>.edges` | 完全图的边数 | integer | - | |
| `<n>.semigroup_operation_time` | 半群运算耗时 | float | 秒 | |
| `<n>.symmetry_detection_time` | 对称性检测耗时 | float | 秒 | |
| `<n>.symmetry_detection_success` | 检测是否成功 | boolean | - | |
| `<n>.automorphisms_count` | 检测到的自同构数量 | integer | - | |

### 1.2 `lie_evolution_strategy_0_*.csv`

**描述**：李群演化演示中策略0（能量优化主导）的能耗随时间步的变化轨迹。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `time_step` | 演化时间步 | integer | 步 | 从0到5共6步 |
| `avg_energy` | 当前时间步的平均认知能耗 | float | - | 相对值，对应李群演化方程的解 |

---

## 2. 四规模对比实验

### 2.1 `config_*.json`

**描述**：实验配置文件，记录了四规模对比实验的参数设置。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `iterations` | 实验迭代次数 | integer | - | 固定为10,000 |
| `repetitions` | 重复次数 | integer | - | 固定为1（单次运行） |
| `models` | 对比的四种模型名称列表 | list of string | - | `["random", "qlearning", "traditional", "emergence"]` |
| `scales` | 四个概念规模列表 | list of integer | - | `[51, 71, 91, 111]` |
| `timestamp` | 实验时间戳 | string | - | 格式：YYYYMMDD_HHMMSS |

### 2.2 `detailed_results_*.csv`

**描述**：四规模对比实验的详细结果，每一行对应一个模型在一个规模下的运行记录。文件采用UTF-8编码，第一行是标题行。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `model` | 模型名称 | string | - | 取值：`random`, `qlearning`, `traditional`, `emergence` |
| `scale` | 概念网络规模 | integer | - | 51, 71, 91, 111 |
| `elapsed_time` | 实验运行时间 | float | 秒 | |
| `iterations` | 实际迭代次数 | integer | - | 固定10,000 |
| `improvement` | 能耗改善率 | float | % | 负值表示能耗增加；计算公式：(初始能耗-最终能耗)/初始能耗×100 |
| `num_nodes` | 概念节点数 | float | - | 与scale相同 |
| `num_edges` | 初始边数 | float | - | 对于random和qlearning模型为0（未使用图结构） |
| `avg_energy` | 最终平均能耗 | float | - | 相对值 |
| `q_table_sparsity` | Q表稀疏度 | float | - | 仅qlearning模型有，非零条目占比 |
| `q_table_non_zero` | Q表非零条目数 | float | - | 仅qlearning模型有 |
| `compression_centers` | 概念压缩中心数 | float | - | 仅traditional和emergence模型有，表示检测到的压缩中心数量 |
| `migration_bridges` | 原理迁移桥梁数 | float | - | 仅traditional和emergence模型有，表示检测到的迁移路径数量 |
| `compression_frequency` | 压缩频率（平均每概念压缩次数） | float | - | 仅emergence模型有，由代码计算得出 |
| `migration_frequency` | 迁移频率（平均每概念迁移次数） | float | - | 仅emergence模型有，由代码计算得出 |

### 2.3 `summary_*.json`

**描述**：将`detailed_results_*.csv`按规模汇总为JSON格式，便于程序读取。每个规模下包含四种模型的详细结果，字段与CSV对应。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `"<scale>"` | 按规模分组的对象 | object | - | 键为字符串形式的规模（如"51"） |
| `<scale>.<model>` | 按模型分组的对象 | object | - | 模型名为`random`, `qlearning`, `traditional`, `emergence` |
| `<model>.model` | 模型名称 | string | - | |
| `<model>.scale` | 规模 | float | - | |
| `<model>.elapsed_time` | 运行时间 | float | 秒 | |
| `<model>.iterations` | 迭代次数 | float | - | |
| `<model>.improvement` | 能耗改善率 | float | % | |
| `<model>.num_nodes` | 节点数 | float | - | |
| `<model>.num_edges` | 边数 | float | - | |
| `<model>.avg_energy` | 最终平均能耗 | float | - | |
| `<model>.q_table_sparsity` | Q表稀疏度 | float | - | 仅qlearning存在 |
| `<model>.q_table_non_zero` | Q表非零条目数 | float | - | 仅qlearning存在 |
| `<model>.compression_centers` | 压缩中心数 | float | - | 仅traditional/emergence存在 |
| `<model>.migration_bridges` | 迁移桥梁数 | float | - | 仅traditional/emergence存在 |
| `<model>.compression_frequency` | 压缩频率 | float | - | 仅emergence存在 |
| `<model>.migration_frequency` | 迁移频率 | float | - | 仅emergence存在 |

---

## 3. 自然涌现模型个体数据（111概念）

### 3.1 `emergence_111_concepts.json`

**描述**：自然涌现模型在111概念网络上的三个独立个体（涌现个体_1, 涌现个体_2, 涌现个体_3）的详细运行数据，包括个体参数、观测到的压缩和迁移事件、能耗变化等。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `[0]` | 第一个个体对象 | object | - | 数组包含三个个体 |
| `individual_id` | 个体标识符 | string | - | 如“涌现个体_1” |
| `parameters` | 个体初始参数 | object | - | 每个参数的取值范围见论文第7.1节及代码随机初始化 |
| `parameters.forgetting_rate` | 遗忘率 | float | - | 控制遗忘操作的强度 |
| `parameters.base_learning_rate` | 基础学习率 | float | - | 控制学习操作的强度 |
| `parameters.hard_traversal_bias` | 硬遍历偏好 | float | - | 偏向确定性遍历的概率 |
| `parameters.soft_traversal_bias` | 软遍历偏好 | float | - | 偏向随机探索的概率 |
| `parameters.compression_bias` | 概念压缩偏好 | float | - | 触发压缩的倾向性 |
| `parameters.migration_bias` | 原理迁移偏好 | float | - | 触发迁移的倾向性 |
| `parameters.learning_rate_variation` | 学习率变异系数 | float | - | 控制学习率随时间的变化幅度 |
| `parameters.energy_variation` | 能耗变异系数 | float | - | 控制能耗计算的随机扰动 |
| `parameters.focus_bias` | 专注状态偏好 | float | - | 进入专注状态的概率偏移 |
| `parameters.exploration_bias` | 探索状态偏好 | float | - | 进入探索状态的概率偏移 |
| `parameters.fatigue_resistance` | 疲劳抵抗系数 | float | - | 影响进入疲劳状态的阈值 |
| `parameters.inspiration_frequency` | 灵感状态频率 | float | - | 进入灵感状态的基础概率 |
| `observations` | 观测到的涌现现象 | object | - | 包含压缩、迁移等列表 |
| `observations.natural_compressions` | 自然概念压缩事件列表 | list of object | - | 每个压缩事件一个对象 |
| `compression.center` | 压缩中心节点 | string | - | 概念名称 |
| `compression.related_nodes` | 相关节点列表 | list of string | - | 被压缩到中心的节点 |
| `compression.energy_synergy` | 能量协同性 | float | - | 压缩后内部连接的平均能耗降低比例 |
| `compression.cohesion` | 集群内聚性 | float | - | 压缩集群的紧密程度，范围为0~1，实验中均为1.0 |
| `compression.emergence_strength` | 涌现强度 | float | - | 综合能量协同性、内聚性等计算的指标，>0.76触发 |
| `compression.cluster_size` | 集群大小 | integer | - | 相关节点个数 |
| `compression.avg_connection_strength` | 平均连接强度 | float | - | 压缩后内部连接的平均权重 |
| `compression.detection_iteration` | 检测到压缩的迭代次数 | integer | - | |
| `observations.natural_migrations` | 第一性原理迁移事件列表 | list of object | - | 每个迁移事件一个对象 |
| `migration.type` | 迁移类型 | string | - | 固定为"first_principles_migration" |
| `migration.principle_node` | 核心原理节点 | string | - | 作为桥梁的节点 |
| `migration.from_node` | 起始节点 | string | - | |
| `migration.to_node` | 目标节点 | string | - | |
| `migration.efficiency_gain` | 效率增益 | float | - | 迁移路径能耗相比于直接连接的降低比例，阈值0.35 |
| `migration.path` | 迁移路径节点列表 | list of string | - | 包含中介节点的完整路径 |
| `migration.emergence_iteration` | 迁移发生迭代 | integer | - | |
| `migration.domain_span` | 领域跨度 | integer | - | 跨越的学科领域数量 |
| `observations.energy_convergence_phases` | 能量收敛阶段 | list | - | 实验中未记录，留空 |
| `final_energy` | 最终平均能耗 | float | - | 相对值 |
| `initial_energy` | 初始平均能耗 | float | - | 相对值 |
| `energy_improvement` | 能耗改善百分比 | float | % | |
| `compression_count` | 压缩事件总数 | integer | - | |
| `migration_count` | 迁移事件总数 | integer | - | |
| `computation_time` | 计算时间 | float | 秒 | |

### 3.2 `emergence_results_*.xlsx`

**描述**：Excel文件，包含两个工作表：“概念压缩”和“第一性原理迁移”，分别记录所有自然涌现个体（三个个体）在实验过程中检测到的概念压缩和原理迁移事件。此文件汇总了各个个体的事件，便于分析。

#### 工作表：概念压缩

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `个体ID` | 个体标识符 | string | - | 如“涌现个体_1” |
| `中心节点` | 压缩中心节点 | string | - | 概念名称 |
| `相关节点数` | 相关节点个数 | integer | - | |
| `相关节点` | 相关节点列表（逗号分隔） | string | - | |
| `能量协同性` | 能量协同性 | float | - | 同JSON定义 |
| `集群内聚性` | 集群内聚性 | float | - | 实验中均为1.0 |
| `涌现强度` | 涌现强度 | float | - | |
| `检测迭代` | 检测迭代次数 | integer | - | |
| `当前网络能耗` | 检测时刻的网络平均能耗 | float | - | 相对值 |
| `时间戳` | 实验时间戳 | string | - | 格式：YYYY-MM-DD HH:MM:SS |

#### 工作表：第一性原理迁移

注：该表在本次实验中为空（无迁移事件记录于该工作表，可能因迁移次数少且已记录于JSON），但保留结构。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| 个体ID | 个体标识符 | string | - | |
| 原理节点 | 核心原理节点 | string | - | |
| 起始节点 | 起始节点 | string | - | |
| 目标节点 | 目标节点 | string | - | |
| 效率增益 | 效率增益 | float | - | |
| 迁移路径 | 完整路径节点列表 | string | - | 以逗号分隔 |
| 领域跨度 | 领域跨度 | integer | - | |
| 检测迭代 | 检测迭代次数 | integer | - | |
| 当前网络能耗 | 检测时刻的网络平均能耗 | float | - | |
| 时间戳 | 实验时间戳 | string | - | |

### 3.3 `energy_history_涌现个体_*.csv`

**描述**：每个自然涌现个体的认知能耗随迭代次数变化的详细历史记录。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `iteration` | 迭代次数 | integer | - | 从0到10000 |
| `energy` | 当前迭代的网络平均能耗 | float | - | 相对值 |

---

## 4. 通用说明

- **能耗值**：所有能耗均为相对值，无量纲，用于比较不同网络状态下的认知负荷。初始能耗在1.5左右，经过优化可降至0.2左右。
- **百分比**：如改善率、效率增益等，均为百分比值（%），计算公式已在备注中说明。
- **时间戳**：格式为YYYY-MM-DD HH:MM:SS 或 YYYYMMDD_HHMMSS，记录实验运行时间。
- **缺失值**：某些字段在特定模型或条件下不存在（如qlearning模型无压缩中心字段），在CSV中表示为空单元格，在JSON中不出现该键。

---

*本数据字典最后更新于2026年2月24日，对应实验运行时间戳20260224。*