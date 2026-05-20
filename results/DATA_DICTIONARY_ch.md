# 数据字典 (Data Dictionary)

本数据字典详细说明认知图论实验*所生成*的所有数据文件的字段含义、数据类型、单位及取值来源。文件按类别分组，每个文件中的字段均基于论文《认知的几何、代数与动力学：基于能耗最小化原理的统一图论模型》中的定义及实验代码的实际输出进行解释。

---

## 1. 代数验证实验(results/algebra)

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

## 2. 四规模对比实验(results/batch_experiments)

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

## 3. 自然涌现模型个体数据（111概念）(results/emergence)

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

注：该表在本次实验中为空（无迁移事件记录于该工作表，因迁移次数少且已记录于JSON），但保留结构。

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
## 4. 分析实验模块 (results/analysis)

### 4.1 压缩协同性阈值扫描 (results/analysis/add_scan)

#### 4.1.1 `threshold_scan_results.json`

**描述**：该文件记录在固定步长（5）下，不同压缩协同性阈值（即涌现强度 `emergence_strength` 的最低门槛）所对应的概念压缩事件总次数，用于分析阈值选择对涌现检测敏感性的影响。阈值以字符串形式的浮点数表示，覆盖从 0.0 至 1.0 的范围（步长 0.05）。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `<threshold>` | 压缩协同性阈值 | float (键为字符串) | - | 例如 `"0.0"`, `"0.05"`, …, `"1.0"`。只有满足 `emergence_strength >= threshold` 的压缩事件才被计入 |
| 值 | 该阈值下检测到的概念压缩总次数 | integer | - | 跨所有个体和规模的合计 |

#### 4.1.2 `threshold_scan_plot.png`

**描述**：将 `threshold_scan_results.json` 可视化的静态图片，横轴为压缩协同性阈值，纵轴为概念压缩次数，便于直观确定合适的操作阈值。

---

### 4.2 Zipf 律检验与客观度量 (results/analysis/objective_metrics)

#### 4.2.1 `compressions_scale_ind0_*.csv`

**描述**：各规模下某个指定个体（`ind0`）的概念压缩事件明细，内容与第 3.2 节 `emergence_results_*.xlsx` 的“概念压缩”工作表高度一致，但仅保留数值型列用于后续分析。表头为 `detection_iteration,center,related_nodes,cluster_size,energy_synergy,cohesion,emergence_strength`。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `detection_iteration` | 检测到压缩的迭代次数 | integer | - | 等同第 3.2 节的“检测迭代” |
| `center` | 压缩中心节点名称 | string | - | 概念名 |
| `related_nodes` | 相关节点列表（逗号分隔） | string | - | 被压缩至中心的概念 |
| `cluster_size` | 集群大小 | integer | - | 相关节点个数 |
| `energy_synergy` | 能量协同性 | float | - | 定义同第 3.1 节 `compression.energy_synergy` |
| `cohesion` | 集群内聚性 | float | - | 定义同第 3.1 节 `compression.cohesion` |
| `emergence_strength` | 涌现强度 | float | - | 定义同第 3.1 节 `compression.emergence_strength` |

#### 4.2.2 `energy_history_scale_ind0_*.csv`

**描述**：与第 3.3 节 `energy_history_涌现个体_*.csv` 完全一致，记录个体在全部迭代中的网络平均能耗。字段定义请直接参见第 3.3 节。

#### 4.2.3 `migrations_scale_ind0_*.csv`

**描述**：与第 3.2 节 `emergence_results_*.xlsx` 的“第一性原理迁移”工作表内容一致，记录迁移事件明细。字段定义请参见第 3.2 节相应工作表。

#### 4.2.4 `individual_summary_*.csv` 与 `individual_summary_*.json`

**描述**：对每个规模下每个个体的运行结果进行汇总，包含能耗改善、事件计数，以及对能耗衰减曲线进行指数拟合的评估指标。CSV 与 JSON 内容相同，JSON 格式便于程序读取。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `scale` | 概念网络规模 | integer | - | 取值：51, 71, 91, 111 |
| `individual` | 个体编号 | integer | - | 从 1 开始 |
| `initial_energy` | 初始平均能耗 | float | - | 相对值 |
| `final_energy` | 最终平均能耗 | float | - | 相对值 |
| `energy_improvement` | 能耗改善百分比 | float | % | 计算公式：`(initial - final)/initial × 100` |
| `total_compressions` | 压缩事件总数 | integer | - | |
| `total_migrations` | 迁移事件总数 | integer | - | |
| `elapsed_time` | 运行耗时 | float | 秒 | |
| `fit_model` | 能耗衰减拟合模型标识 | string | - | 实验中为 `"exp"`（指数衰减模型） |
| `fit_r2` | 拟合决定系数 R² | float | - | 越接近 1 拟合越好 |
| `fit_rmse` | 拟合均方根误差 | float | - | 与能耗同量纲的相对值 |
| `fit_fluctuation` | 拟合残差的波动幅度 | float | - | 反映能耗围绕指数衰减的随机涨落强度 |
| `fit_params` | 拟合参数列表的字符串表示 | string | - | 例如指数模型 `a·exp(-b·t) + c` 的参数 `[a, b, c]`，需解析后使用 |

> **注**：拟合基于 `energy_history_*.csv` 的迭代-能耗数据，用于评价认知能耗收敛趋势的指数衰减特性。

#### 4.2.5 `zipf_center_result_*.json`

**描述**：汇总所有个体、所有规模下的压缩事件，统计各概念作为压缩中心出现的频次，并进行 Zipf 律检验（对频率-位序取对数后的线性回归）。记录回归参数及前十大高频中心。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `slope` | 线性回归斜率（Zipf 指数） | float | - | log(位序) ~ log(频次) 的斜率，通常为负值 |
| `intercept` | 回归截距 | float | - | |
| `r2` | 回归决定系数 R² | float | - | 衡量频次分布符合 Zipf 律的程度 |
| `p_value` | 斜率显著性的 p 值 | float | - | 越小表示幂律关系越显著 |
| `total_events` | 参与统计的压缩事件总数 | integer | - | |
| `unique_centers` | 唯一压缩中心的数量 | integer | - | |
| `top_centers` | 频次最高的前十个中心及其出现次数 | object | - | 键为概念名，值为出现次数 |

#### 4.2.6 `zipf_node_total_result_*.json`

**描述**：与 `zipf_center_result_*.json` 类似，但统计对象扩展为所有压缩事件中涉及的每个节点（包括中心节点及相关节点）的总出现次数，以在更大范围内验证认知涌现的幂律分布特征。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `slope` | Zipf 指数 | float | - | |
| `intercept` | 截距 | float | - | |
| `r2` | 决定系数 | float | - | |
| `p_value` | p 值 | float | - | |
| `total_occurrences` | 所有节点在事件中的总出现次数 | integer | - | 每个节点每次被计入一次 |
| `unique_nodes` | 唯一节点个数 | integer | - | |
| `top_nodes` | 频次最高的前十个节点及其出现次数 | object | - | 键为概念名，值为总出现次数 |

---

### 4.3 压缩势统计分析 (results/analysis/potential_analysis)

#### 4.3.1 `emergence_<scale>_ind0_*.json`

**描述**：记录涌现个体在特定规模下的完整观测数据，其整体结构与第 3.1 节 `emergence_111_concepts.json` 一致，但在每个压缩事件对象中新增了 `compression_potential` 字段。`compression_potential` 定义为：压缩集群内部所有边的平均权重除以集群外部所有边的平均权重，用于衡量该次压缩的“势能”——数值越高，说明集群内部连接相对外部越紧密，压缩效益越大。

新增字段（其余字段参见第 3.1 节）：

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `compression_potential` | 压缩势 | float | - | 内部平均边权 / 外部平均边权；大于 1 表示内部连接强度高于外部 |

> 注：文件名中的 `<scale>` 为 51、71、91、111 等概念规模。

#### 4.3.2 `potential_summary_*.json`

**描述**：对各规模下所有个体的所有压缩事件的 `compression_potential` 值进行汇总统计，包含计数、均值、标准差及四分位数，用于分析压缩势在不同规模下的分布特性。

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `<scale>` | 概念规模（字符串键） | object | - | 例如 `"51"`, `"71"`, `"91"`, `"111"` |
| `count` | 该规模下压缩事件总数 | integer | - | |
| `mean` | 压缩势均值 | float | - | |
| `std` | 压缩势标准差 | float | - | |
| `min` | 最小值 | float | - | |
| `max` | 最大值 | float | - | |
| `q25` | 第一四分位数（25%） | float | - | |
| `q50` | 中位数（50%） | float | - | |
| `q75` | 第三四分位数（75%） | float | - | |

#### 4.3.3 `potential_dist_scale.png`

**描述**：展示各规模下压缩势的频次分布直方图，横轴为压缩势区间，纵轴为频次，便于直观比较不同规模的压缩势分布形态。

---
## 5. 可视化结果图 (results/Fresults/animations)

**描述**：该目录存放由自然涌现个体运行数据生成的认知图演化动画，包含 2D 动态 GIF 和 3D 交互式 HTML 两种形式。所有动画均基于同一位个体（`ind0`，对应“涌现个体_1”）在完整 10,000 次迭代中的网络快照生成，按不同概念规模分别输出。

### 5.1 文件命名规范

| 文件模式 | 格式 | 说明 |
|----------|------|------|
| `cognitive_anim_dynamic_{scale}_ind0_*.gif` | GIF | 规模为 `{scale}` 个概念的 2D 认知图演化动画。 `{scale}` 取值为 51, 71, 91, 111。 |
| `cognitive_3d_{scale}_ind0_*.html` | HTML | 规模为 `{scale}` 个概念的 3D 交互式认知图演化动画。可在浏览器中旋转/缩放观察。`{scale}` 取值同上。 |

时间戳部分（`*`）为生成文件时的实验批次标识，格式为 `YYYYMMDD_HHMMSS`。

### 5.2 视觉元素编码说明

以下编码规则同时适用于 2D 与 3D 动画。各视觉元素直接映射自底层图数据（概念节点、边权重、遍历操作和瞬时能耗），其定义与第 3 节中自然涌现模型的输出一致。

| 元素 | 视觉表现 | 数据映射与含义 |
|------|----------|----------------|
| **节点** | 浅蓝色圆点，黑色边框 | 代表概念节点（如“算法”、“神经网络”、“能量守恒”等）。节点的存在与标识来自概念网络中的节点列表。 |
| **节点标签** | 黑色文字，显示于节点旁 | 概念的名称，与`emergence_111_concepts.json`中的概念标识一致。 |
| **普通边** | 颜色渐变与粗细可变 | 表示概念之间的认知关联。**颜色**：从蓝绿色到红紫色渐变——越偏蓝绿表示当前边的能耗（连接权重）越低（学习充分、认知阻力小），越偏红紫表示能耗越高（生疏、阻力大）。**粗细**：线越粗表示边的当前能耗越高（即连接强度弱、遍历代价大），线越细表示能耗越低（连接流畅）。每帧的边颜色和粗细根据该迭代下的平均能耗动态更新。 |
| **硬遍历路径** | 蓝色实线箭头 | 系统执行“硬遍历”——沿着已有低能耗路径进行高效、确定的认知操作，巩固已有知识。箭头指示遍历方向。 |
| **软遍历尝试** | 橙色虚线箭头 | 软遍历过程中尝试过的多个可能方向，体现探索性。虚线表示这些是备选路径，未被最终执行。 |
| **软遍历实际路径** | 红色实线箭头 | 软遍历最终选择的路径，用于探索新关联或跨域连接。红色实线表示实际发生的遍历。 |
| **高亮节点** | 黄色圆点，黑色边框 | 当前遍历路径上经过的节点，表示当前正在被激活/访问的概念。同一时刻可能有多条路径的节点被高亮。 |
| **标题栏** | 左上角（2D）或界面HUD（3D）文字 | 显示当前迭代次数 `iteration` 和网络全局平均能耗 `energy`（相对值），与 `energy_history_涌现个体_*.csv` 中的对应行数据一致。 |

#### 补充说明

- 边的颜色和粗细随迭代次数动态变化，直接反映底层图数据中边权重的更新（第 3 节中个体的学习和遗忘效果）。
- 硬遍历通常沿一条路径前进（蓝色实线），软遍历则先尝试多个方向（橙色虚线），再确定实际路径（红色实线），两者反映了第 3.1 节参数中 `hard_traversal_bias` 和 `soft_traversal_bias` 控制的认知策略。
- 在部分动画中，节点位置会随网络结构变化而缓慢移动（动态布局），以直观体现认知空间的重构。

---

## 6. 群体认知状态与能耗历史 (results/population)

### 6.1 `energy_history_Individual_*.csv` 与 `energy_history_个体_*.csv`

**描述**：记录群体实验中每个独立个体（或与第 3 节一致的涌现个体）在全部迭代过程中的逐次认知能耗及认知状态。文件名中的 `Individual_*` 为通用标识，`个体_*` 为中文标识，均指代同一批个体。该文件与第 3.3 节的 `energy_history_涌现个体_*.csv` 功能相同，但每行的 `energy` 列扩展为完整的结构化记录，包含该迭代的认知状态枚举。

**文件格式**：CSV 文件，两列，逗号分隔。首行为标题 `iteration,energy`。从第二行起每行代表一次迭代，`energy` 列包含一个 Python 字典的字符串表示，内含 `iteration`、`state` 和 `energy` 三个键。

**样例行**：
```
0,"{'iteration': 0, 'state': <CognitiveState.FOCUSED: '专注状态'>, 'energy': 1.5}"
```

| 字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|------|------|----------|------|------------|
| `iteration` | 迭代次数 | integer | - | 从 0 到 10,000，与第 3.3 节的定义一致 |
| `energy` | 当前迭代的认知状态与能耗记录 | string（JSON-like字典） | - | 需解析为字典后使用，见下方子字段说明 |

#### `energy` 字段解析后的子字段

| 子字段 | 含义 | 数据类型 | 单位 | 来源/备注 |
|--------|------|----------|------|------------|
| `iteration` | 迭代次数（冗余校验） | integer | - | 应与所在行的 `iteration` 列一致 |
| `state` | 当前认知状态 | string (枚举) | - | 取值来自 `CognitiveState` 枚举：`FOCUSED`（专注状态）、`EXPLORATION`（探索状态）、`FATIGUED`（疲劳状态）、`INSPIRATION`（灵感状态） |
| `energy` | 当前迭代的网络平均能耗 | float | - | 相对值，含义与第 3.3 节相同 |

> **注意**：若直接读取 CSV，`energy` 列是字符串，需进行解析（如使用 `eval()` 或 `json.loads()` 替换单引号后转换）。`state` 的表示形式为 `<CognitiveState.FOCUSED: '专注状态'>`，在分析时建议提取枚举名称或中文描述部分。
---


## 7. 通用说明

- **能耗值**：所有能耗均为相对值，无量纲，用于比较不同网络状态下的认知负荷。初始能耗在1.5左右，经过优化可降至0.2左右。
- **百分比**：如改善率、效率增益等，均为百分比值（%），计算公式已在备注中说明。
- **时间戳**：格式为YYYY-MM-DD HH:MM:SS 或 YYYYMMDD_HHMMSS，记录实验运行时间。
- **缺失值**：某些字段在特定模型或条件下不存在（如qlearning模型无压缩中心字段），在CSV中表示为空单元格，在JSON中不出现该键。

---

## 8. 补充说明

- 在 `results/` 目录下存在按规模命名的独立 Excel 文件：`51_concepts.xlsx`、`71_concepts.xlsx`、`91_concepts.xlsx`、`111_concepts.xlsx`。这些文件的内容与第 3.2 节 `emergence_results_*.xlsx` 完全一致，仅是按概念规模从主文件中抽取出的独立副本，便于按规模单独分发或查阅。其工作表结构与字段定义请直接参见第 3.2 节，此处不再重复列出。
- 所有 `.xlsx` 文件中的时间戳、能耗值、概念名称等均与对应的 JSON 及 CSV 记录保持同步，可交叉验证。

---

*本数据字典最后更新于2026年5月20日。*