# Nonergodic Development

**Paper:** *Nonergodic Development: How High-Dimensional Systems with Low-Dimensional Anchors Generate Phenotypes*

**Target journal:** BioSystems

**Author:** Ian Todd, University of Sydney

---

## Abstract

Biological development is a high-dimensional dynamical process that cannot explore its state space in finite time—it is *nonergodic*. We argue that this nonergodicity, combined with low-dimensional genetic anchors, is the fundamental reason why genotype does not algorithmically determine phenotype. The genome constrains which regions of developmental state space are reachable, but environmental history determines which attractor basin the system occupies. We formalize this via the **Dimensional Gap** (Δ_D), which quantifies when allele-based and trajectory-based models become non-identifiable from aggregate data.

## Repository Contents

```
├── nonergodic_development.tex  # Main paper (LaTeX, BioSystems format)
├── nonergodic_development.pdf  # Compiled paper
├── nonergodic_development.py   # Core simulation code (Figs 1-6)
├── multilevel_simulation.py    # Multilevel selection simulation (Figs 7-8)
├── cover_letter.tex            # Cover letter to editor
└── figures/                    # Generated figures (PDF + PNG)
    ├── fig1_same_genotype.*
    ├── fig2_population_patterns.*
    ├── fig3_causal_dags.*
    ├── fig4_projection_loss.*
    ├── fig5_twin_worlds.*
    ├── fig6_intervention_test.*
    ├── fig7_fractal_price.*      # Price equation decomposition
    └── fig8_regime_comparison.*  # Selection regime comparison
```

## Running the Code

```bash
pip install numpy matplotlib scipy scikit-learn
python nonergodic_development.py   # Generates figures 1-6
python multilevel_simulation.py    # Generates figures 7-8
```

This generates all 8 figures in the `figures/` directory.

---

## AI-Assisted Workflow (Full Transparency)

This paper was developed using an AI-assisted workflow. In the spirit of open science, here's exactly how it worked:

### Primary Drafting: Claude Code (Claude 4.5 Opus)

The paper was developed interactively using [Claude Code](https://claude.com/claude-code), Anthropic's CLI tool. The workflow was:

1. **Conceptual discussion:** I described the core argument (development is nonergodic; genomes anchor high-dimensional systems; trajectories determine phenotypes)

2. **Model development:** Claude helped formalize the dimensional gap concept (Δ_D) and write the developmental network simulation

3. **Figure design:** Each figure was designed iteratively:
   - I described what I wanted to show
   - Claude wrote the plotting code
   - We refined panel layouts, colors, and labels through back-and-forth

4. **LaTeX drafting:** Claude drafted sections based on our discussions; I edited and directed revisions

5. **Citation integration:** References to Waddington, Kauffman, Lissek, Rosen, etc. to ground the argument

### Feedback Loop: Gemini 3 Pro + GPT 5.1 Pro

After drafting, I fed the paper to Gemini and GPT for critical feedback. This revealed:

- **Notation inconsistencies** (e.g., `k` was overloaded for both trait dimension and environment dimension)
- **Code-text mismatches** (e.g., bias term in equation that wasn't in the code)
- **Missing parameter values** (now explicitly stated in Methods)
- **Figure numbering mismatches** (fixed to match paper order)

Each round of feedback was implemented via Claude Code.

### Figure Design Philosophy

The figures were designed to tell a specific story:

| Figure | Purpose |
|--------|---------|
| **Fig 1** | Same genotype → different phenotypes (trajectory divergence) |
| **Fig 2** | Population patterns arise from either mechanism |
| **Fig 3** | Causal DAGs showing structural difference |
| **Fig 4** | Information loss under projection (the dimensional gap) |
| **Fig 5** | "Twin Worlds" - the decisive demonstration |
| **Fig 6** | Where the models diverge (intervention test) |
| **Fig 7** | "Fractal Price" - multilevel selection with Price equation decomposition |
| **Fig 8** | Selection regime comparison - group selection enables cooperation |

Figure 5 is the key result: identical genotype distributions in different environments produce patterns a naive allele model would interpret as genetic differences.

Figures 7-8 extend the model to evolutionary time, showing that the same developmental architecture (coherence) that suppresses cancer also enables social cooperation - a "fractal" structure where the same mechanism operates at multiple scales.

### What the AI Did vs. What I Did

**AI contributed:**
- Code generation (Python simulation, LaTeX formatting)
- Literature search and citation formatting
- Draft text based on my descriptions
- Iterative refinement based on feedback

**I contributed:**
- Core theoretical argument (nonergodicity + anchoring)
- Decisions about what to include/exclude
- Final editing and approval of all content
- Integration with broader research program (this is paper III in a series on dimensional constraints in biology)

---

## The Model

The developmental network is a recurrent neural network treated as a gene regulatory network:

```
h_{t+1} = tanh(W_h · h_t + W_e · e_t + W_g · g)
```

Where:
- `g ∈ R^L` is the genotype (L=5, the low-dimensional *anchor*)
- `e_t ∈ R^p` is environmental input at time t (p=3)
- `h_t ∈ R^m` is developmental state (m=20, high-dimensional, *nonergodic*)

The phenotype is a linear readout: `x = W_out · h_T`

Cancer mortality emerges via: `μ_S = μ_0(1 - α · c)` where `c` is developmental coherence—how coordinated the trajectory was. Cancer is attractor bifurcation: cells diverging from the organismal trajectory.

## Multilevel Simulation (Fractal Cooperation)

The `multilevel_simulation.py` extends the model to evolutionary time with explicit multilevel selection:

- **Organisms** develop phenotypes via the developmental network
- **Groups** provide the developmental environment (more cooperative groups → more stable environments)
- **Selection** operates at both levels: between-group (cooperative groups outcompete) and within-group (individuals compete)

The **Price equation** decomposes selection into:
- `S_between`: Selection due to group differences (favors cooperation)
- `S_within`: Selection due to individual differences (can favor defection)

Key finding: **The same coherence parameter (c) that suppresses cancer also drives cooperation**. This is "fractal" because the same dynamical structure (high-D development + low-D anchor → attractor trapping) operates at multiple scales:
- Cells cooperating within organisms (coherence prevents cancer)
- Organisms cooperating within groups (coherence creates stable environment)

## Key Result

![Twin Worlds Experiment](figures/fig5_twin_worlds.png)

**Proposition 1 (Non-identifiability):** For any allele-based mortality pattern {μ, μ+δ}, there exists an environment pair that produces identical patterns via trajectory-based mechanisms. The models are distinguishable only through intervention experiments.

**The core insight:** Some of the "missing heritability" in GWAS may be missing trajectory information, not missing variants. The dimensional gap Δ_D quantifies how much information is lost when we project from (anchor, trajectory, attractor) to (genotype, phenotype).

---

## Related Work

This paper builds on:
- [Todd (2025a)](https://doi.org/10.1016/j.biosystems.2025.105608) - Falsifiability and dimensional constraints
- [Todd (2025b)](https://doi.org/10.1016/j.biosystems.2025.105632) - Maxwell's demon and projection bounds

And responds to:
- [Sierra et al. (2025)](https://doi.org/10.1126/sciadv.adw0685) - Cooperation and cancer in mammals
- [Lissek (2024)](https://doi.org/10.1016/j.biosystems.2024.105381) - Cancer memory hypothesis

---

## License

Code: MIT License

Paper: © 2025 Ian Todd. All rights reserved.
