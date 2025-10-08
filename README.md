# Topological Properties and Wavefunction Localization in Bianchi I Geometry
# 动态Bianchi I几何中准周期系统的拓扑特性与波函数局域化研究

[![MATLAB](https://img.shields.io/badge/MATLAB-R2022a+-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](#english) 

---

## English

### Overview
This project investigates the topological properties and wavefunction localization in quasiperiodic quantum systems under Bianchi I geometric constraints. By numerically simulating a three-dimensional lattice model, we analyze how expansion factors affect the Inverse Participation Ratio (IPR), energy band structure, Berry curvature, and Chern numbers.

### Key Features
- **Dynamic Geometry Analysis**: Study of expansion factor evolution over time
- **Directional Localization**: Anisotropic localization properties in X, Y, Z directions  
- **Topological Phase Transitions**: Critical point analysis in Z-direction (a₃≈1.442)
- **Geometric Resonance**: U-shaped localization characteristics in Y-direction (a₂≈1.113)
- **High-Precision Scanning**: Fine-grained parameter space exploration

### Physical Model

**Hamiltonian:**
H = -Σ(tₓ + tᵧ + tᵧ + h.c.) + ΣVᵢⱼₖ
**Geometric Constraints (Bianchi I):**
tₓ = t₀/a₁,  tᵧ = t₀/a₂,  tᵧ = t₀/a₃
**Quasiperiodic Potential:**
Vᵢⱼₖ = V₀[cos(2πβₓi/a₁ + φₓ) + cos(2πβᵧj/a₂ + φᵧ) + cos(2πβᵧk/a₃ + φᵧ)]
### Main Results

1. **Z-Direction Phase Transition**
   - Critical expansion factor: a₃ ≈ 1.442
   - Maximum IPR_z ≈ 11.3
   - Energy gap minimum with fidelity susceptibility peak

2. **Y-Direction Geometric Resonance**  
   - Optimal expansion factor: a₂ ≈ 1.113
   - U-shaped IPR behavior (IPR_y: 7 → 1.8 → 7)
   - Controllable localization-to-extended state transition

3. **Anisotropic Properties**
   - Strong directional dependence
   - Non-linear coupling between different directions
   - Direction-selective quantum states
### Key Functions

**Core Calculations:**
- `build_hamiltonian()` / `build_enhanced_hamiltonian()`: Construct system Hamiltonian
- `calculate_directional_IPR()` / `calculate_enhanced_IPR()`: Compute directional localization measures
- `calculate_berry_curvature_3D()`: Berry curvature in momentum space
- `wilson_loop_chern_numbers()`: Exact Chern number calculation

**Advanced Analysis:**
- `analytical_framework()`: Theoretical predictions  
- `construct_complete_phase_diagram()`: 3D parameter space mapping
- `finite_size_scaling_analysis()`: Thermodynamic limit extrapolation
- `calculate_correlation_length()`: Spatial correlation analysis

### System Requirements
- MATLAB R2022a or higher
- Required Toolboxes: None (uses built-in sparse matrix operations)
- Memory: ~8GB RAM for default 20×20×20 lattice
- CPU: Multi-core recommended for phase diagram construction

### Quick Start
```matlab
% Run basic dynamic analysis (untitled.m)
% Generates 3 figures and .mat results
run untitled.m

% Run enhanced analysis with topological invariants (untitled2.m)  
run untitled2.m

% Run geometric control study with phase search (untitled3.m)
run untitled3.m
###Parameters
###Default Configuration:
Lx = Ly = Lz = 20        % Lattice size
t₀ = 1.0                 % Base hopping amplitude
V₀ = 2.0                 % Quasiperiodic potential strength
βₓ = (√5-1)/2            % Golden ratio conjugate
βᵧ = (√3-1)/2            % Irrational modulation
βᵧ = (√2-1)/2            % Irrational modulation
time_steps = 50-60       % Temporal resolution
Output Files
Data Files:

Bianchi_I_完整分析结果.mat: Complete analysis results
Bianchi_I_增强分析结果.mat: Enhanced analysis data
几何调控量子局域化_完整研究结果.mat: Geometric control study
物理洞察总结.mat: Physical insights summary

Visualizations:

Phase diagrams (2D/3D)
IPR evolution plots
Energy spectrum dynamics
Chern number distributions
Correlation length analysis

Physical Insights
Geometric Resonance Mechanism:
The U-shaped IPR behavior in Y-direction reveals a competition between hopping amplitude and effective disorder. At the resonance point (a₂≈1.113), the geometric modulation optimally balances kinetic and potential energy contributions, enabling maximum wavefunction delocalization.
Topological Phase Transition:
Z-direction exhibits a localization-extended phase transition near a₃≈1.442, characterized by:

Energy gap closure

Fidelity susceptibility divergence
Anomalous critical exponent (ν≈-0.028)

This suggests a non-conventional quantum phase transition driven by geometric constraints rather than traditional disorder or interaction effects.
