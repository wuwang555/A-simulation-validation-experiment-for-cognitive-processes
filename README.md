# Cognitive Process Simulation: Experimental Validation of Energy Minimization Theory

## Overview
This experiment is based on the theoretical framework of **Cognitive Graph Theory (CGT)**, using computational simulation to validate the mechanisms of cognitive structure formation and evolution driven by the principle of energy minimization. It aims to reveal the unified dynamics behind efficient human learning and innovative thinking, and to provide a theoretical foundation for next-generation artificial intelligence systems.

*(Note: This is a validation study; there may be minor omissions. It is not a complete, production-ready project. This project acknowledges the use of AI assistance in its development.)*

---

[**中文文档 (Chinese README)**](README_zh.md) | **English**

## 🔬 Research Significance: Bridging Intuition and Scientific Frontier

**This research represents an extraordinary case where independent theoretical exploration from first principles converges with cutting-edge scientific consensus.** Through introspective observation of personal learning processes, I independently formulated a computational theory of cognition that addresses core challenges currently debated in the field of embodied intelligence.

### 🧠 Deep Alignment with Embodied Intelligence Challenges
My "Cognitive Graph Theory" provides precisely the kind of theoretical framework needed for developing the "brain" component of truly intelligent systems:

1. **Addressing "Data Scarcity and High Energy Consumption"**: The natural emergence paradigm is **small-sample** and **principle-driven**, achieving significant energy reduction (88.0%→47.9% optimization) through intrinsic energy optimization rather than massive labeled data.
2. **Overcoming "Inadequate Cognitive and Adaptive Capacity"**: The framework provides rigorous mathematical descriptions of cognition through **geometry** (cognitive space), **algebra** (group operations), and **dynamics** (energy evolution), establishing foundations for deep reasoning models.
3. **Transcending "Statistical Correlation-Based AI"**: Starting from two basic postulates (cognitive spacetime and energy dynamics), the model enables complex cognitive structures to emerge naturally, addressing the fundamental limitations of current pattern-matching approaches.

### ✨ From Personal Observation to Universal Principle
What makes this work particularly compelling is the journey:
- **Source**: Not derived from literature review, but from **introspective observation** of personal thought processes.
- **Path**: Independent construction of a coherent, mathematical theoretical framework.
- **Convergence**: Discovery that independent exploration reaches **critical consensus** with leading global scholars through different paths.

This convergence suggests that I may have identified fundamental principles of intelligent systems. While others discuss "what kind of intelligence we need," this computational model explores "what fundamental principles such intelligence might follow."

## Core Objectives
- Validate whether energy optimization serves as the fundamental driver of cognitive organization.
- Observe the natural emergence of **Concept Compression** and **First-Principle Transfer** within cognitive structures.
- Explore the adaptive mechanisms of cognitive systems under different knowledge scales.
- Compare the performance differences between four cognitive paradigms.

## Experimental Design

### Four-Scale Comparative Experiment
| Scale | Concepts | Domains Covered | Purpose |
|-------|----------|-----------------|---------|
| 51 Concepts | 51 | Physics, Mathematics, Computer Science, AI, Cognitive Science (7 domains) | Baseline validation |
| 71 Concepts | 71 | Extended with Control Theory, Differential Geometry, Game Theory | Medium-scale adaptation |
| 91 Concepts | 91 | Further extended with Compiler Principles, HCI, Ethics | Complex-scale robustness |
| 111 Concepts | 111 | Rich cross-domain knowledge ecosystem | Critical-scale analysis |

### Four Contrasting Paradigms
1. **Random Network Baseline**: Reference without intelligent mechanisms.
2. **Simple Reinforcement Learning (Q-learning)**: Represents traditional AI methods.
3. **Preset Algorithm Model**: Simulates traditional cognitive computing paradigms (e.g., ACT-R).
4. **Pure Energy Emergence Model**: New paradigm proposed in this paper (based only on two postulates).

### Unified Parameter Settings
- Iterations: 10,000
- Cognitive Temperature \( T = 1.0 \)
- Learning Rate: 0.85
- Concept Compression Synergy Threshold: 0.76
- First-Principle Transfer Efficiency Threshold: 0.35
- Cluster Cohesion Threshold: 0.7

## 🔬 Key Findings

### 1. Significant Energy Optimization Performance
| Concept Scale | Emergence Model | Preset Algorithm Model | Relative Advantage |
|---------------|----------------|------------------------|--------------------|
| 51 Concepts | **88.0%** energy reduction | 23.1% | +64.9% |
| 71 Concepts | **72.3%** energy reduction | 15.5% | +56.8% |
| 91 Concepts | **61.4%** energy reduction | 12.2% | +49.2% |
| 111 Concepts | **47.9%** energy reduction | 7.6% | +40.3% |

### 2. Spontaneous Emergence of Concept Compression
- **Significant changes in compression density**: From 9.14 compressions per concept (51 concepts) to 0.39 compressions per concept (111 concepts).
- **Perfect cohesion**: All compressions achieved cohesion = 1.0.
- **Enhanced emergence strength**: Increased from 0.79 to 0.97 (111 concepts).

### 3. First-Principle Transfer Simulating Creative Thinking
- **Transfer efficiency**: Average 0.35-0.45, with maximum reaching 0.449.
- **Hub nodes**: "Pattern Recognition" appeared in 6 out of 7 transfers.
- **Strategy shift**: Transitioned from exploratory transfers to conservative, precise transfers.

### 4. Scale-Adaptive Mechanisms
- **Cognitive load management**: System adjusted compression and transfer strategies based on complexity.
- **Expert-novice difference simulation**: Changes in compression density perfectly simulated cognitive development from beginner to expert.
- **Critical phenomena**: Fundamental strategy shift at 111 concepts.

## 📁 Project Structure

```
├── README.md                          # Project documentation (this file)
├── README_zh.md                       # Chinese documentation
├── LICENSE                            # Open source license
├── sesfullu01.py                      # One-click run script (recommended entry)
├── code/                              # Core code directory
│   ├── main.py                        # Main program entry
│   ├── config.py                      # Configuration file (this project)
│   ├── algebra/                       # Algebraic verification module
│   ├── core/                          # Core modules
│   ├── emergence/                     # Emergence detection module
│   ├── experiments/                   # Experimental modules
│   ├── models/                        # Model definitions
│   ├── utils/                         # Utility functions
│   └── results/                       # Experimental results
├── article/                           # Paper-related files
└── .idea/                             # IDE configuration files
```

### About Semantic Similarity Calculation
This project uses a **simplified version** of semantic similarity calculation, primarily serving experimental validation purposes. In another independent project, we have built a **more complex and comprehensive Chinese semantic processing framework** including:

- **Detailed Chinese meta-concept mapping**: e.g., the character "上" possesses multiple attributes like "space," "time," and "hierarchy."
- **Rich concept definitions**: Each concept has detailed descriptions covering multiple dimensions.
- **Multi-level semantic relationships**: Including causal relationships, parallel relationships, progressive relationships, etc.
- **Concept abstraction hierarchy**: Dividing concepts into concrete, general, abstract, and meta-concepts.
- **Semantic enhancement configurations**: Chinese cognitive feature lexicons across dimensions like core thinking, learning processes, cognitive levels, knowledge types, and thinking qualities.

These efforts provide the foundation for more refined semantic analysis. To maintain this project's focus (validating the energy minimization principle), we use simplified semantic similarity calculations here.

## 🚀 Quick Start

### Environment Requirements
- Python 3.8+
- Main dependencies: numpy, pandas, matplotlib, networkx, scipy

### Installation
```bash
# Recommended to use conda to create a virtual environment
conda create -n cognitive_graph python=3.9
conda activate cognitive_graph

# Install dependencies
pip install numpy pandas matplotlib networkx scipy
```

### Running Experiments
#### Method 1: One-click run all experiments (recommended)
```bash
python run_experiments.py
```

#### Method 2: Run by module
```bash
# Run four-scale comparison experiments
python code/experiments/batch_experiments.py

# Run emergence study
python code/experiments/emergence_study_fixed.py

# Run algebraic verification experiments
python code/algebra/algebra_experiments.py
```

### Viewing Results
Experimental results are saved in:
- **Excel files**: Experimental results for each scale in `code/results/`
- **Chart files**: PNG charts in `code/results/batch_experiments/`
- **Paper charts**: PNG files in `article/` can be directly used in papers

## 📈 Results Reproduction

### 1. Four-Scale Energy Optimization Comparison
The experiment reproduces Table 7.2 data:
- Emergence model: 51 concepts 88.0% → 111 concepts 47.9%
- Preset algorithm model: 51 concepts 23.1% → 111 concepts 7.6%

### 2. Concept Compression Density Changes
The experiment reproduces Table 7.3 data:
- Total compressions: 466 times (51 concepts) → 43 times (111 concepts)
- Compressions per concept: 9.14 times/concept → 0.39 times/concept

### 3. Multi-level Comparison Experiment
The experiment reproduces Table 7.4 data, validating performance differences among four paradigms:
- Random network: Negative optimization (-12.9% to -59.4%)
- Q-learning: Severe performance degradation (13.5%→2.9%)
- Preset algorithm: Moderate degradation (23.1%→7.6%)
- Natural emergence: Optimal robustness (88.0%→47.9%)

## 📚 Paper Related

### Paper Content
- **Complete paper**: `article/CognitiveGraph.pdf`
- **LaTeX source**: `article/CognitiveGraph.tex`
- **References**: `article/references.bib`

### Key Chapters
1. **Section 3**: Axiomatic system of Cognitive Graph Theory (geometric description)
2. **Section 4**: Algebraic structure of Cognitive Graph Theory (group theory description)
3. **Section 5**: Emergence mechanisms driven by energy dynamics
4. **Section 6**: Algebraic verification experiments (5 validations)
5. **Section 7**: Four-scale experiment design and result analysis (core)
6. **Section 8**: Discussion and theoretical significance
7. **Section 9**: Conclusions and future directions

### Key Figures in the Paper
- **Figure 7.1**: Four-model performance comparison chart (`article/performance_comparison.png`)
- **Figure 7.2**: Scale effect curve chart (`article/scale_effect.png`)
- **Figure 8.1**: Theoretical integration schematic diagram (LaTeX generated)

## 🧪 Experimental Validation

### Completed Validations
1. ✅ **Algebraic validation** (Section 6): Cognitive operation semigroup, Noether-type propositions, orbit-stabilizer theorem, Lie group evolution framework
2. ✅ **Four-scale experiments** (Section 7): Systematic comparison of 51/71/91/111 concept networks
3. ✅ **Paradigm comparison**: Comprehensive comparison of random network, Q-learning, preset algorithm, and natural emergence
4. ✅ **Statistical validation**: Scale effects, compression density attenuation, transfer efficiency, and other metrics

### Reproducibility Guarantees
- **Fixed random seeds**: Ensures experimental results are reproducible
- **Unified parameter settings**: Four-scale experiments use the same thresholds
- **Complete data preservation**: All experimental results saved as Excel files

## 🔍 Deep Exploration

### Code Module Description
| Module | Function | Key Files |
|--------|----------|-----------|
| **Algebraic Structure** | Group theory description and validation of cognitive operations | `algebra/cognitive_semigroup.py` |
| **Emergence Detection** | Automatic detection of concept compression and principle transfer | `emergence/detector_fixed.py` |
| **State Management** | Simulation of cognitive states (focused, exploratory, inspired, fatigued) | `core/cognitive_states.py` |
| **Comparison Models** | Implementation of four cognitive paradigms | Files in `models/` directory |
| **Visualization** | Experimental result chart generation | `utils/visualization.py` |

### Key Parameter Adjustment
To adjust experimental parameters, modify:
1. **Scale parameters**: `CONCEPT_SETS` in `code/config.py`
2. **Threshold settings**: `thresholds` in `code/emergence/detector_fixed.py`
3. **Energy function**: `compute_edge_energy` in `code/core/cognitive_graph.py`

## 📈 Extensions and Improvements

### Current Limitations
1. **Network scale limitation**: Currently maximum 111 concepts, still distant from real cognition
2. **Insufficient semantic depth**: Concept nodes lack hierarchical structure and detail complexity
3. **Simplified individual differences**: Differences introduced only through random initialization
4. **Missing multimodality**: Purely symbolic representation, lacking perceptual and motor grounding

### Semantic Calculation Enhancement Directions
The current project uses relatively simplified semantic similarity calculations. Future work could integrate more complex Chinese semantic processing capabilities:
1. **Introduce Chinese meta-structures**: Utilize Chinese meta-concept mapping to capture multi-dimensional attributes of concepts
2. **Use detailed concept descriptions**: Utilize extended concept definitions for deeper semantic analysis
3. **Combine semantic relationships**: Utilize causal relationships, parallel relationships, etc., to build richer semantic networks
4. **Multi-level abstraction**: Apply concept abstraction hierarchy for cross-level semantic reasoning

### Integration with Embodied Intelligence Frontiers
The cognitive graph theory framework proposed in this project provides a theoretical foundation for the cognitive architecture of embodied intelligence. Future work could attempt:
1. **Integration with robotic platforms**: Deploy this cognitive model on robots to validate its learning capabilities in physical environments
2. **Multimodal integration**: Introduce visual, auditory, and other modalities to achieve learning processes closer to humans
3. **Online learning**: Dynamically update cognitive graphs from real-time interactions, simulating infants' continuous learning
4. **Principle transfer validation**: Test the effectiveness of first-principle transfer in embodied environments

### Publication and Academic Exchange
Given the project's high relevance to current embodied intelligence frontiers, it is recommended to:
1. **Conference submission**: Target high-level conferences similar to "2026 Greater Bay Area Embodied Intelligence International Forum," such as EI-OAHV 2026
2. **Paper rewriting**: Organize core results into English abstracts according to conference submission requirements, highlighting connections with embodied intelligence, cognitive architecture, and other frontier fields
3. **Theoretical expansion**: Further elaborate how cognitive graph theory provides principled solutions to core bottlenecks in embodied intelligence
4. **Experimental validation**: Design validation experiments in embodied environments to demonstrate the model's adaptability in the physical world

## 🤝 Contribution and Communication

### Project Status
- **Author**: Mingjia Zeng (sophomore student)
- **Time**: December 2025
- **Status**: Theoretical framework basically complete, experimental validation sufficient
- **Characteristics**: AI-assisted generation, but core ideas and experiments are original

### Relationship with Frontier Forums
This project's research direction is highly relevant to the core issues of the 2026 Greater Bay Area Embodied Intelligence International Forum. The forum points out that current embodied intelligence faces challenges such as "poor data foundation, high energy consumption" and "insufficient brain cognition and adaptive capacity." The energy minimization cognitive model proposed in this project is precisely a principled exploration attempting to address these challenges from the cognitive architecture level. We encourage interested researchers to build upon this foundation to jointly advance the development of cognitive models for embodied intelligence.

### How to Contribute
1. **Report issues**: Submit Issues on GitHub/Gitee
2. **Improve code**: Submit Pull Requests
3. **Extend experiments**: Try larger scales or new paradigms
4. **Theoretical discussion**: Discuss fundamental issues in cognitive science
5. **Publication support**: Assist in transforming research results into academic papers or conference presentations

## 📄 License and Citation

### License
This project uses the MIT license. See the `LICENSE` file for details.

### Related Projects
This project's semantic processing section references another independent project's Chinese semantic processing framework, which provides more complex Chinese meta-structure analysis and concept definitions. Related code is included in the provided files (`chinese_semantic.py`, `concept_definitions.py`, `semantic_config.py`).

### Citation
```
Zeng, M. (2025). Geometry, Algebra, and Dynamics of Cognition: A Unified Graph-Theoretic Model Based on the Principle of Energy Minimization.
```

### Acknowledgments
- Thanks to Karl Friston's free energy principle for providing philosophical foundations
- Thanks to Gärdenfors' conceptual space theory for providing geometric inspiration
- Thanks to all pioneering work in cognitive science and artificial intelligence
- Thanks to AI-assisted tools for help in writing and programming

## 🌟 Core Philosophy

> **Cognition knows no bounds, learning has no end**  
> Technology is merely our tool for understanding the world  
> True intelligence lies not in computational power, but in insight into principles  
> Energy minimization is not just an optimization strategy, but a fundamental law of cognitive organization

> **Our exploration walks with frontier science**: Starting from self-observation, the independently constructed cognitive graph theory model remarkably converges with critical consensus reached by leading global scholars through different paths. This reveals that deep thinking about the nature of cognition is the core driving force pushing artificial intelligence forward.

<div align="center">

**🚀 Exploring the Boundaries of Cognition, Understanding the Essence of Learning**  
**Transforming Introspection into Computation, Letting Principles Drive Intelligence**

*This is a project in progress, documenting a student's curiosity and attempts*  
*Progress has no end, and neither does human potential.*

</div>