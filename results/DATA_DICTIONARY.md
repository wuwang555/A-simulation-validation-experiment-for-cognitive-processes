# Data Dictionary

This data dictionary details the field meanings, data types, units, and value sources for all data files *generated* by the cognitive graph theory experiments. Files are grouped by category. The fields in each file are explained based on the definitions in the paper *"Geometry, Algebra and Dynamics of Cognition: A Unified Graph-Theoretic Model Based on the Principle of Energy Minimization"* and the actual output of the experimental code.

---

## 1. Algebraic Validation Experiments (results/algebra)

### 1.1 `algebra_validation_results_*.json`

**Description**: This file records the complete results of the five groups of experiments in Section 5 "Algebraic Validation Experiments," including semigroup associativity verification, Noether-type proposition verification, orbit-stabilizer theorem verification, Lie group evolution demonstration, and algebraic method scalability test.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `experiment1` | Experiment 1: Cognitive operation semigroup validation results | object | - | Contains three cases of associativity verification |
| `experiment1.associativity` | Specific cases of associativity verification | object | - | Keys are operation combinations, values are booleans (True indicates associativity holds) |
| `experiment1.operation_count` | Number of basic operations involved in verification | integer | - | 13 basic operations defined in the experiment |
| `experiment1.note` | Remarks | string | - | Indicates that the identity element check is temporarily skipped (semigroups do not require identity elements) |
| `experiment2` | Experiment 2: Noether-type proposition validation results | object | - | Contains validation data for three networks with different structures |
| `experiment2.network_*` | Validation results for each network | object | - | Network indices 0, 1, 2 correspond to 5-, 10-, and 15-node networks |
| `network_*.automorphisms_count` | Number of automorphisms of the network | integer | - | All are 0 in the experiment, reflecting the low symmetry of cognitive networks |
| `network_*.conserved_quantities` | Conserved quantities (values before evolution) | object | - | Includes total_energy, structural_entropy, fractal_dimension |
| `network_*.noether_theorem_holds` | Whether the Noether theorem holds | [bool, object] | - | The first value is the overall boolean conclusion; the second object gives the relative change of each conserved quantity before and after evolution and the conservation judgment |
| `network_*.energy_before` | Global cognitive energy before evolution | float | - | Dimensionless relative value |
| `network_*.energy_after` | Global cognitive energy after evolution | float | - | Same as above |
| `experiment3` | Experiment 3: Orbit-Stabilizer theorem verification | object | - | Contains automorphism group size, stabilizer size, orbit size, and whether the theorem holds |
| `experiment3.automorphism_group_size` | Size of the automorphism group | integer | - | 0 in the experiment |
| `experiment3.orbit_size_actual` | Actual orbit size | integer | - | 0 in the experiment |
| `experiment3.stabilizer_size` | Stabilizer size | integer | - | 0 in the experiment |
| `experiment3.orbit_size_expected` | Expected orbit size according to the theorem | float | - | 0.0 |
| `experiment3.theorem_holds` | Whether the theorem holds | string | - | "True" |
| `experiment3.error_percentage` | Error percentage | float | - | 0 |
| `experiment4` | Experiment 4: Lie group evolution strategy demonstration | object | - | Contains evolution results for three different generator combinations |
| `experiment4.strategy_*` | Detailed results for each strategy | object | - | Strategy 0: Energy optimization dominant; Strategy 1: Concept compression dominant; Strategy 2: Principle migration dominant |
| `strategy_*.generator_coeffs` | Coefficients of the Lie algebra generators | object | - | Keys E (Energy optimization), C (Concept compression), M (Principle migration), values are the coefficients |
| `strategy_*.initial_energy` | Initial cognitive energy consumption | float | - | Relative value |
| `strategy_*.final_energy` | Final cognitive energy consumption | float | - | Relative value |
| `strategy_*.energy_change_percent` | Percentage change in energy consumption | float | % | Negative value indicates reduction |
| `strategy_*.energy_trajectory` | Energy consumption at each time step during evolution | list of float | - | Energy values at 6 time points |
| `experiment5` | Experiment 5: Algebraic method scalability test | object | - | Keys are network sizes (number of nodes), values are corresponding results |
| `experiment5.<n>` | Test results for network with n nodes | object | - | n = 5,8,10,12,15 |
| `<n>.nodes` | Number of nodes | integer | - | |
| `<n>.edges` | Number of edges in the complete graph | integer | - | |
| `<n>.semigroup_operation_time` | Time consumed by semigroup operations | float | seconds | |
| `<n>.symmetry_detection_time` | Time consumed by symmetry detection | float | seconds | |
| `<n>.symmetry_detection_success` | Whether detection succeeded | boolean | - | |
| `<n>.automorphisms_count` | Number of automorphisms detected | integer | - | |

### 1.2 `lie_evolution_strategy_0_*.csv`

**Description**: The trajectory of energy consumption over time steps for Strategy 0 (energy optimization dominant) in the Lie group evolution demonstration.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `time_step` | Evolution time step | integer | step | 6 steps from 0 to 5 |
| `avg_energy` | Average cognitive energy consumption at the current time step | float | - | Relative value, corresponding to the solution of the Lie group evolution equation |

---

## 2. Four-Scale Comparative Experiments (results/batch_experiments)

### 2.1 `config_*.json`

**Description**: Experiment configuration file, recording the parameter settings for the four-scale comparative experiments.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `iterations` | Number of experiment iterations | integer | - | Fixed at 10,000 |
| `repetitions` | Number of repetitions | integer | - | Fixed at 1 (single run) |
| `models` | List of the four model names compared | list of string | - | `["random", "qlearning", "traditional", "emergence"]` |
| `scales` | List of the four concept scales | list of integer | - | `[51, 71, 91, 111]` |
| `timestamp` | Experiment timestamp | string | - | Format: YYYYMMDD_HHMMSS |

### 2.2 `detailed_results_*.csv`

**Description**: Detailed results of the four-scale comparative experiments. Each row corresponds to the run record of one model at one scale. The file is UTF-8 encoded; the first row is the header row.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `model` | Model name | string | - | Values: `random`, `qlearning`, `traditional`, `emergence` |
| `scale` | Concept network scale | integer | - | 51, 71, 91, 111 |
| `elapsed_time` | Experiment running time | float | seconds | |
| `iterations` | Actual number of iterations | integer | - | Fixed at 10,000 |
| `improvement` | Energy improvement rate | float | % | Negative value indicates energy increase; Formula: (initial_energy - final_energy)/initial_energy × 100 |
| `num_nodes` | Number of concept nodes | float | - | Same as scale |
| `num_edges` | Initial number of edges | float | - | 0 for random and qlearning models (no graph structure used) |
| `avg_energy` | Final average energy | float | - | Relative value |
| `q_table_sparsity` | Q-table sparsity | float | - | Only for the qlearning model; proportion of non-zero entries |
| `q_table_non_zero` | Number of non-zero entries in Q-table | float | - | Only for the qlearning model |
| `compression_centers` | Number of concept compression centers | float | - | Only for traditional and emergence models; number of compression centers detected |
| `migration_bridges` | Number of principle migration bridges | float | - | Only for traditional and emergence models; number of migration paths detected |
| `compression_frequency` | Compression frequency (average compressions per concept) | float | - | Only for the emergence model; calculated by the code |
| `migration_frequency` | Migration frequency (average migrations per concept) | float | - | Only for the emergence model; calculated by the code |

### 2.3 `summary_*.json`

**Description**: Summarizes `detailed_results_*.csv` by scale into JSON format for easy programmatic reading. Each scale contains detailed results for the four models; fields correspond to the CSV.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `"<scale>"` | Object grouped by scale | object | - | Key is the scale as a string (e.g., "51") |
| `<scale>.<model>` | Object grouped by model | object | - | Model names: `random`, `qlearning`, `traditional`, `emergence` |
| `<model>.model` | Model name | string | - | |
| `<model>.scale` | Scale | float | - | |
| `<model>.elapsed_time` | Running time | float | seconds | |
| `<model>.iterations` | Number of iterations | float | - | |
| `<model>.improvement` | Energy improvement rate | float | % | |
| `<model>.num_nodes` | Number of nodes | float | - | |
| `<model>.num_edges` | Number of edges | float | - | |
| `<model>.avg_energy` | Final average energy | float | - | |
| `<model>.q_table_sparsity` | Q-table sparsity | float | - | Only exists for qlearning |
| `<model>.q_table_non_zero` | Number of non-zero entries in Q-table | float | - | Only exists for qlearning |
| `<model>.compression_centers` | Number of compression centers | float | - | Only exists for traditional/emergence |
| `<model>.migration_bridges` | Number of migration bridges | float | - | Only exists for traditional/emergence |
| `<model>.compression_frequency` | Compression frequency | float | - | Only exists for emergence |
| `<model>.migration_frequency` | Migration frequency | float | - | Only exists for emergence |

---

## 3. Natural Emergence Model Individual Data (111 concepts) (results/emergence)

### 3.1 `emergence_111_concepts.json`

**Description**: Detailed run data of three independent individuals (Emergence Individual_1, Emergence Individual_2, Emergence Individual_3) of the natural emergence model on a 111-concept network, including individual parameters, observed compression and migration events, and energy changes.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `[0]` | First individual object | object | - | Array contains three individuals |
| `individual_id` | Individual identifier | string | - | e.g., "Emergence Individual_1" |
| `parameters` | Initial parameters of the individual | object | - | Value ranges of each parameter are described in Section 7.1 of the paper and randomly initialized by the code |
| `parameters.forgetting_rate` | Forgetting rate | float | - | Controls the intensity of the forgetting operation |
| `parameters.base_learning_rate` | Base learning rate | float | - | Controls the intensity of the learning operation |
| `parameters.hard_traversal_bias` | Hard traversal bias | float | - | Probability biased towards deterministic traversal |
| `parameters.soft_traversal_bias` | Soft traversal bias | float | - | Probability biased towards random exploration |
| `parameters.compression_bias` | Concept compression bias | float | - | Tendency to trigger compression |
| `parameters.migration_bias` | Principle migration bias | float | - | Tendency to trigger migration |
| `parameters.learning_rate_variation` | Learning rate variation coefficient | float | - | Controls the amplitude of learning rate change over time |
| `parameters.energy_variation` | Energy variation coefficient | float | - | Controls random perturbation in energy calculation |
| `parameters.focus_bias` | Focus state bias | float | - | Probability offset for entering the focused state |
| `parameters.exploration_bias` | Exploration state bias | float | - | Probability offset for entering the exploration state |
| `parameters.fatigue_resistance` | Fatigue resistance coefficient | float | - | Affects the threshold for entering the fatigued state |
| `parameters.inspiration_frequency` | Inspiration state frequency | float | - | Base probability of entering the inspiration state |
| `observations` | Observed emergence phenomena | object | - | Contains lists of compressions, migrations, etc. |
| `observations.natural_compressions` | List of natural concept compression events | list of object | - | One object per compression event |
| `compression.center` | Compression center node | string | - | Concept name |
| `compression.related_nodes` | List of related nodes | list of string | - | Nodes compressed to the center |
| `compression.energy_synergy` | Energy synergy | float | - | Average energy reduction ratio of internal connections after compression |
| `compression.cohesion` | Cluster cohesion | float | - | Tightness of the compression cluster, range 0~1; all 1.0 in the experiment |
| `compression.emergence_strength` | Emergence strength | float | - | Metric calculated from energy synergy, cohesion, etc.; triggers when > 0.76 |
| `compression.cluster_size` | Cluster size | integer | - | Number of related nodes |
| `compression.avg_connection_strength` | Average connection strength | float | - | Average weight of internal connections after compression |
| `compression.detection_iteration` | Iteration number when compression was detected | integer | - | |
| `observations.natural_migrations` | List of first-principles migration events | list of object | - | One object per migration event |
| `migration.type` | Migration type | string | - | Fixed as "first_principles_migration" |
| `migration.principle_node` | Core principle node | string | - | Node serving as the bridge |
| `migration.from_node` | Starting node | string | - | |
| `migration.to_node` | Target node | string | - | |
| `migration.efficiency_gain` | Efficiency gain | float | - | Reduction ratio of the migration path energy compared to direct connection; threshold 0.35 |
| `migration.path` | List of nodes on the migration path | list of string | - | Complete path including intermediary nodes |
| `migration.emergence_iteration` | Iteration when migration occurred | integer | - | |
| `migration.domain_span` | Domain span | integer | - | Number of subject domains crossed |
| `observations.energy_convergence_phases` | Energy convergence phases | list | - | Not recorded in the experiment; left empty |
| `final_energy` | Final average energy | float | - | Relative value |
| `initial_energy` | Initial average energy | float | - | Relative value |
| `energy_improvement` | Energy improvement percentage | float | % | |
| `compression_count` | Total number of compression events | integer | - | |
| `migration_count` | Total number of migration events | integer | - | |
| `computation_time` | Computation time | float | seconds | |

### 3.2 `emergence_results_*.xlsx`

**Description**: Excel file containing two worksheets: "Concept Compression" and "First-Principles Migration", which respectively record the concept compression and principle migration events detected during the experiment for all natural emergence individuals (three individuals). This file aggregates events from each individual for easier analysis.

#### Worksheet: Concept Compression

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `Individual ID` | Individual identifier | string | - | e.g., "Emergence Individual_1" |
| `Center Node` | Compression center node | string | - | Concept name |
| `Number of Related Nodes` | Number of related nodes | integer | - | |
| `Related Nodes` | List of related nodes (comma-separated) | string | - | |
| `Energy Synergy` | Energy synergy | float | - | Same definition as in JSON |
| `Cluster Cohesion` | Cluster cohesion | float | - | All 1.0 in the experiment |
| `Emergence Strength` | Emergence strength | float | - | |
| `Detection Iteration` | Detection iteration number | integer | - | |
| `Current Network Energy` | Network average energy at detection time | float | - | Relative value |
| `Timestamp` | Experiment timestamp | string | - | Format: YYYY-MM-DD HH:MM:SS |

#### Worksheet: First-Principles Migration

Note: This worksheet is empty in this experiment (no migration events recorded in this worksheet because the number of migrations is small and they are already recorded in JSON), but the structure is retained.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| Individual ID | Individual identifier | string | - | |
| Principle Node | Core principle node | string | - | |
| From Node | Starting node | string | - | |
| To Node | Target node | string | - | |
| Efficiency Gain | Efficiency gain | float | - | |
| Migration Path | Complete path node list | string | - | Comma-separated |
| Domain Span | Domain span | integer | - | |
| Detection Iteration | Detection iteration number | integer | - | |
| Current Network Energy | Network average energy at detection time | float | - | |
| Timestamp | Experiment timestamp | string | - | |

### 3.3 `energy_history_emergence_individual_*.csv`

**Description**: Detailed history of cognitive energy consumption over iterations for each natural emergence individual.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `iteration` | Iteration number | integer | - | From 0 to 10000 |
| `energy` | Network average energy at the current iteration | float | - | Relative value |

---
## 4. Analysis Experiment Modules (results/analysis)

### 4.1 Compression Synergy Threshold Scan (results/analysis/add_scan)

#### 4.1.1 `threshold_scan_results.json`

**Description**: This file records the total number of concept compression events under different compression synergy thresholds (i.e., the minimum threshold for `emergence_strength`) at a fixed step size (0.05), used to analyze the impact of threshold selection on the sensitivity of emergence detection. Thresholds are represented as string-formatted floats, covering the range from 0.0 to 1.0 (step 0.05).

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `<threshold>` | Compression synergy threshold | float (key as string) | - | e.g., `"0.0"`, `"0.05"`, ..., `"1.0"`. Only compression events with `emergence_strength >= threshold` are counted |
| value | Total number of concept compressions detected under that threshold | integer | - | Summed across all individuals and scales |

#### 4.1.2 `threshold_scan_plot.png`

**Description**: A static image visualizing `threshold_scan_results.json`, with the compression synergy threshold on the x-axis and the number of concept compressions on the y-axis, facilitating intuitive determination of an appropriate operational threshold.

---

### 4.2 Zipf's Law Test and Objective Metrics (results/analysis/objective_metrics)

#### 4.2.1 `compressions_scale_ind0_*.csv`

**Description**: Details of concept compression events for a specific individual (`ind0`) at each scale. The content is highly consistent with the "Concept Compression" worksheet of `emergence_results_*.xlsx` in Section 3.2, but retains only numeric columns for subsequent analysis. The header is `detection_iteration,center,related_nodes,cluster_size,energy_synergy,cohesion,emergence_strength`.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `detection_iteration` | Iteration number when compression was detected | integer | - | Same as "Detection Iteration" in Section 3.2 |
| `center` | Compression center node name | string | - | Concept name |
| `related_nodes` | List of related nodes (comma-separated) | string | - | Concepts compressed to the center |
| `cluster_size` | Cluster size | integer | - | Number of related nodes |
| `energy_synergy` | Energy synergy | float | - | Definition same as `compression.energy_synergy` in Section 3.1 |
| `cohesion` | Cluster cohesion | float | - | Definition same as `compression.cohesion` in Section 3.1 |
| `emergence_strength` | Emergence strength | float | - | Definition same as `compression.emergence_strength` in Section 3.1 |

#### 4.2.2 `energy_history_scale_ind0_*.csv`

**Description**: Exactly the same as `energy_history_emergence_individual_*.csv` in Section 3.3, recording the network average energy of the individual across all iterations. Please refer directly to Section 3.3 for field definitions.

#### 4.2.3 `migrations_scale_ind0_*.csv`

**Description**: Content consistent with the "First-Principles Migration" worksheet of `emergence_results_*.xlsx` in Section 3.2, recording migration event details. Please refer to the corresponding worksheet in Section 3.2 for field definitions.

#### 4.2.4 `individual_summary_*.csv` and `individual_summary_*.json`

**Description**: Summarizes the run results for each individual at each scale, including energy improvement, event counts, and evaluation metrics for the exponential fit of the energy decay curve. The CSV and JSON contain the same content; the JSON format is easier for programmatic reading.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `scale` | Concept network scale | integer | - | Values: 51, 71, 91, 111 |
| `individual` | Individual number | integer | - | Starts from 1 |
| `initial_energy` | Initial average energy | float | - | Relative value |
| `final_energy` | Final average energy | float | - | Relative value |
| `energy_improvement` | Energy improvement percentage | float | % | Formula: `(initial - final)/initial × 100` |
| `total_compressions` | Total number of compression events | integer | - | |
| `total_migrations` | Total number of migration events | integer | - | |
| `elapsed_time` | Running time consumed | float | seconds | |
| `fit_model` | Identifier of the energy decay fitting model | string | - | In the experiment, it is `"exp"` (exponential decay model) |
| `fit_r2` | Coefficient of determination R² of the fit | float | - | Closer to 1 indicates better fit |
| `fit_rmse` | Root mean square error of the fit | float | - | Relative value, same dimension as energy |
| `fit_fluctuation` | Fluctuation amplitude of the fit residuals | float | - | Reflects the intensity of random fluctuations of energy around the exponential decay |
| `fit_params` | String representation of the fitting parameter list | string | - | e.g., for the exponential model `a·exp(-b·t) + c`, the parameters are `[a, b, c]`; needs to be parsed before use |

> **Note**: The fitting is based on the iteration-energy data from `energy_history_*.csv` and is used to evaluate the exponential decay characteristics of the cognitive energy convergence trend.

#### 4.2.5 `zipf_center_result_*.json`

**Description**: Aggregates compression events across all individuals and scales, counts the frequency of each concept appearing as a compression center, and performs a Zipf's law test (linear regression on log-transformed rank-frequency data). Records regression parameters and the top ten most frequent centers.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `slope` | Linear regression slope (Zipf exponent) | float | - | Slope of log(rank) ~ log(frequency), typically negative |
| `intercept` | Regression intercept | float | - | |
| `r2` | Coefficient of determination R² | float | - | Measures the degree to which the frequency distribution conforms to Zipf's law |
| `p_value` | p-value for slope significance | float | - | The smaller the value, the more significant the power-law relationship |
| `total_events` | Total number of compression events included in the statistics | integer | - | |
| `unique_centers` | Number of unique compression centers | integer | - | |
| `top_centers` | Top ten most frequent centers and their occurrence counts | object | - | Keys are concept names, values are occurrence counts |

#### 4.2.6 `zipf_node_total_result_*.json`

**Description**: Similar to `zipf_center_result_*.json`, but the statistical object is expanded to include the total occurrence count of every node involved in all compression events (including both center nodes and related nodes), to verify the power-law distribution characteristics of cognitive emergence on a larger scale.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `slope` | Zipf exponent | float | - | |
| `intercept` | Intercept | float | - | |
| `r2` | Coefficient of determination | float | - | |
| `p_value` | p-value | float | - | |
| `total_occurrences` | Total occurrences of all nodes in events | integer | - | Each node is counted once per event it appears in |
| `unique_nodes` | Number of unique nodes | integer | - | |
| `top_nodes` | Top ten most frequent nodes and their occurrence counts | object | - | Keys are concept names, values are total occurrence counts |

---

### 4.3 Compression Potential Statistical Analysis (results/analysis/potential_analysis)

#### 4.3.1 `emergence_<scale>_ind0_*.json`

**Description**: Records the complete observation data of an emergence individual at a specific scale. The overall structure is consistent with `emergence_111_concepts.json` in Section 3.1, but a new `compression_potential` field is added to each compression event object. `compression_potential` is defined as the average weight of all edges inside the compression cluster divided by the average weight of all edges outside the cluster, measuring the "potential" of this compression—higher values indicate that the internal connections of the cluster are stronger relative to the external ones, implying greater compression benefit.

New field (see Section 3.1 for the remaining fields):

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `compression_potential` | Compression potential | float | - | Internal average edge weight / external average edge weight; greater than 1 means internal connection strength is higher than external |

> Note: `<scale>` in the file name stands for concept scales such as 51, 71, 91, 111.

#### 4.3.2 `potential_summary_*.json`

**Description**: Provides summary statistics of the `compression_potential` values for all compression events across all individuals at each scale, including count, mean, standard deviation, and quartiles, for analyzing the distribution characteristics of compression potential at different scales.

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `<scale>` | Concept scale (string key) | object | - | e.g., `"51"`, `"71"`, `"91"`, `"111"` |
| `count` | Total number of compression events at this scale | integer | - | |
| `mean` | Mean compression potential | float | - | |
| `std` | Standard deviation of compression potential | float | - | |
| `min` | Minimum value | float | - | |
| `max` | Maximum value | float | - | |
| `q25` | First quartile (25%) | float | - | |
| `q50` | Median (50%) | float | - | |
| `q75` | Third quartile (75%) | float | - | |

#### 4.3.3 `potential_dist_scale.png`

**Description**: Shows frequency distribution histograms of compression potential at each scale, with the compression potential interval on the x-axis and frequency on the y-axis, facilitating intuitive comparison of compression potential distribution shapes across different scales.

---
## 5. Visualization Results (results/Fresults/animations)

**Description**: This directory stores cognitive graph evolution animations generated from the run data of a natural emergence individual, including 2D animated GIFs and 3D interactive HTML formats. All animations are generated based on network snapshots of the same individual (`ind0`, corresponding to "Emergence Individual_1") across the full 10,000 iterations, output separately for different concept scales.

### 5.1 File Naming Convention

| File Pattern | Format | Description |
|----------|------|------|
| `cognitive_anim_dynamic_{scale}_ind0_*.gif` | GIF | 2D cognitive graph evolution animation for the scale of `{scale}` concepts. `{scale}` values: 51, 71, 91, 111. |
| `cognitive_3d_{scale}_ind0_*.html` | HTML | 3D interactive cognitive graph evolution animation for the scale of `{scale}` concepts. Can be rotated/zoomed in the browser. `{scale}` values as above. |

The timestamp part (`*`) is the experiment batch identifier when the file was generated, in the format `YYYYMMDD_HHMMSS`.

### 5.2 Visual Element Encoding Description

The following encoding rules apply to both 2D and 3D animations. Each visual element maps directly from the underlying graph data (concept nodes, edge weights, traversal operations, and instantaneous energy), and their definitions are consistent with the output of the natural emergence model in Section 3.

| Element | Visual Representation | Data Mapping and Meaning |
|------|----------|----------------|
| **Node** | Light blue dot with black border | Represents a concept node (e.g., "Algorithm", "Neural Network", "Conservation of Energy"). The existence and identity of nodes come from the node list in the concept network. |
| **Node Label** | Black text displayed next to the node | The name of the concept, consistent with the concept identifiers in `emergence_111_concepts.json`. |
| **Ordinary Edge** | Color gradient and variable thickness | Represents cognitive associations between concepts. **Color**: Gradient from cyan/green to red/magenta—more cyan/green indicates lower current edge energy (weight) (well-learned, low cognitive resistance), more red/magenta indicates higher energy (unfamiliar, high resistance). **Thickness**: Thicker lines indicate higher current edge energy (i.e., weak connection strength, high traversal cost); thinner lines indicate lower energy (smooth connection). Edge colors and thickness dynamically update each frame based on the average energy at that iteration. |
| **Hard Traversal Path** | Blue solid arrow | The system performs a "hard traversal"—efficient, deterministic cognitive operations along existing low-energy paths to consolidate existing knowledge. The arrow indicates the traversal direction. |
| **Soft Traversal Attempt** | Orange dashed arrow | Multiple possible directions tried during soft traversal, reflecting exploratory behavior. Dashed lines indicate these are alternative paths, not ultimately executed. |
| **Soft Traversal Actual Path** | Red solid arrow | The path finally chosen by soft traversal, used to explore new associations or cross-domain connections. The red solid line represents the actual traversal that occurred. |
| **Highlighted Node** | Yellow dot with black border | Nodes passed on the current traversal path, indicating the concept currently being activated/visited. Nodes from multiple paths may be highlighted simultaneously. |
| **Title Bar** | Text at top-left (2D) or HUD in interface (3D) | Displays the current iteration number `iteration` and the network's global average energy `energy` (relative value), consistent with the corresponding row data in `energy_history_emergence_individual_*.csv`. |

#### Supplementary Notes

- Edge colors and thicknesses change dynamically with iterations, directly reflecting the updates to edge weights in the underlying graph data (the learning and forgetting effects of the individuals in Section 3).
- Hard traversal usually proceeds along a single path (blue solid line), while soft traversal first tries multiple directions (orange dashed lines) and then determines the actual path (red solid line). Both reflect the cognitive strategies controlled by `hard_traversal_bias` and `soft_traversal_bias` in Section 3.1 parameters.
- In some animations, node positions may slowly move as the network structure changes (dynamic layout) to visually represent the reconstruction of the cognitive space.

---

## 6. Population Cognitive State and Energy History (results/population)

### 6.1 `energy_history_Individual_*.csv` and `energy_history_individual_*.csv`

**Description**: Records the per-iteration cognitive energy and cognitive state for each independent individual in the population experiment (or the same emergence individuals as in Section 3). `Individual_*` in the file name is the generic identifier, and `individual_*` is the Chinese identifier; both refer to the same batch of individuals. This file has the same function as `energy_history_emergence_individual_*.csv` in Section 3.3, but the `energy` column in each row is expanded into a complete structured record containing the cognitive state enumeration for that iteration.

**File Format**: CSV file with two columns, comma-separated. The first row is the header `iteration,energy`. Starting from the second row, each row represents one iteration, and the `energy` column contains a string representation of a Python dictionary containing the three keys `iteration`, `state`, and `energy`.

**Sample row**:
```
0,"{'iteration': 0, 'state': <CognitiveState.FOCUSED: 'focused state'>, 'energy': 1.5}"
```

| Field | Meaning | Data Type | Unit | Source/Remarks |
|------|------|----------|------|------------|
| `iteration` | Iteration number | integer | - | From 0 to 10,000, consistent with the definition in Section 3.3 |
| `energy` | Cognitive state and energy record at the current iteration | string (JSON-like dict) | - | Needs to be parsed as a dictionary before use; see sub-field descriptions below |

#### Parsed Sub-fields of the `energy` Field

| Sub-field | Meaning | Data Type | Unit | Source/Remarks |
|--------|------|----------|------|------------|
| `iteration` | Iteration number (redundant check) | integer | - | Should be consistent with the `iteration` column of the row |
| `state` | Current cognitive state | string (enum) | - | Values from the `CognitiveState` enum: `FOCUSED` (focused state), `EXPLORATION` (exploration state), `FATIGUED` (fatigued state), `INSPIRATION` (inspiration state) |
| `energy` | Network average energy at the current iteration | float | - | Relative value, same meaning as in Section 3.3 |

> **Note**: When reading the CSV directly, the `energy` column is a string and needs to be parsed (e.g., using `eval()` or `json.loads()` after replacing single quotes). The representation of `state` is like `<CognitiveState.FOCUSED: 'focused state'>`; when analyzing, it is recommended to extract the enum name or the description part.
---

## 7. General Notes

- **Energy Values**: All energy values are relative and dimensionless, used for comparing cognitive loads under different network states. The initial energy is around 1.5, which can be reduced to around 0.2 after optimization.
- **Percentages**: Such as improvement rate, efficiency gain, etc., are all percentage values (%). The calculation formulas are described in the remarks.
- **Timestamps**: Format is YYYY-MM-DD HH:MM:SS or YYYYMMDD_HHMMSS, recording the experiment run time.
- **Missing Values**: Some fields do not exist under specific models or conditions (e.g., the qlearning model has no compression center field). They are represented as empty cells in CSV and the key is absent in JSON.

---

## 8. Supplementary Notes

- In the `results/` directory, there exist independent Excel files named by scale: `51_concepts.xlsx`, `71_concepts.xlsx`, `91_concepts.xlsx`, `111_concepts.xlsx`. The content of these files is completely identical to that of `emergence_results_*.xlsx` in Section 3.2; they are merely independent copies extracted from the main file by concept scale, making it convenient to distribute or consult separately by scale. For their worksheet structures and field definitions, please refer directly to Section 3.2; they are not repeated here.
- The timestamps, energy values, concept names, etc., in all `.xlsx` files are synchronized with the corresponding JSON and CSV records and can be cross-validated.

---

*This data dictionary was last updated on May 20, 2026.*