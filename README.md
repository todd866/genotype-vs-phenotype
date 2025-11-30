# Genotype ≠ Phenotype

**Paper:** *Genotype ≠ Phenotype: High-Dimensional Development, Plasticity, and the Limits of Allele Stories*

**Target journal:** BioSystems

**Author:** Ian Todd, University of Sydney

---

## Abstract

The assumption that genotype algorithmically determines phenotype underlies much of evolutionary modeling. We prove this framing is incomplete: when phenotype emerges from low-dimensional genetic parameters passing through a high-dimensional developmental system, allele-based and plasticity-based models become *non-identifiable* from typical observational data.

## Repository Contents

```
├── genotype_phenotype.tex    # Main paper (LaTeX, BioSystems format)
├── genotype_phenotype.pdf    # Compiled paper
├── developmental_network.py  # All simulation code
└── figures/                  # Generated figures (PDF + PNG)
    ├── fig1_same_genotype.*
    ├── fig2_population_patterns.*
    ├── fig3_projection_loss.*
    ├── fig4_prediction_divergence.*
    ├── fig5_causal_dags.*
    └── fig6_twin_worlds.*
```

## Running the Code

```bash
pip install numpy matplotlib scipy scikit-learn
python developmental_network.py
```

This generates all 6 figures in the `figures/` directory and exactly reproduces the plots in the paper.

---

## AI-Assisted Workflow (Full Transparency)

This paper was developed using an AI-assisted workflow. In the spirit of open science, here's exactly how it worked:

### Primary Drafting: Claude Code (Claude 4.5 Opus)

The paper was developed interactively using [Claude Code](https://claude.com/claude-code), Anthropic's CLI tool. The workflow was:

1. **Conceptual discussion:** I described the core argument (genotype provides parameters to a high-dimensional developmental system; allele stories are non-identifiable from plasticity mechanisms)

2. **Model development:** Claude helped formalize the dimensional gap concept (ΔD) and write the developmental network simulation

3. **Figure design:** Each figure was designed iteratively:
   - I described what I wanted to show
   - Claude wrote the plotting code
   - We refined panel layouts, colors, and labels through back-and-forth

4. **LaTeX drafting:** Claude drafted sections based on our discussions; I edited and directed revisions

5. **Citation integration:** We added ~18 references (Pigliucci, West-Eberhard, Waddington, Kauffman, etc.) to ground the argument in existing literature

### Feedback Loop: Gemini 3 Pro + GPT 5.1 Pro

After drafting, I fed the paper to Gemini and GPT for critical feedback. This revealed:

- **Notation inconsistencies** (e.g., `k` was overloaded for both trait dimension and environment dimension)
- **Code-text mismatches** (e.g., bias term in equation that wasn't in the code)
- **Missing parameter values** (now explicitly stated in §3.3)
- **Weak section transitions** (added explanatory text)

Each round of feedback was implemented via Claude Code.

### Figure Design Philosophy

The figures were designed to tell a specific story:

| Figure | Purpose |
|--------|---------|
| **Fig 1** | Same genotype → different phenotypes (the core claim) |
| **Fig 2** | Population patterns arise from either mechanism |
| **Fig 3** | Information loss under projection (why we can't tell) |
| **Fig 4** | Where the models diverge (intervention predictions) |
| **Fig 5** | Causal DAGs showing structural difference |
| **Fig 6** | "Twin Worlds" - the decisive demonstration |

Figure 6 is the key result: identical genotype distributions in different environments produce patterns a naive allele model would interpret as genetic differences.

### What the AI Did vs. What I Did

**AI contributed:**
- Code generation (Python simulation, LaTeX formatting)
- Literature search and citation formatting
- Draft text based on my descriptions
- Iterative refinement based on feedback

**I contributed:**
- Core theoretical argument
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
- `g ∈ R^L` is the genotype (L=5, low-dimensional)
- `e_t ∈ R^p` is environmental input at time t (p=3)
- `h_t ∈ R^m` is developmental state (m=20, high-dimensional)

The phenotype is a linear readout: `x = W_out · h_T`

Cancer mortality emerges via: $\mu_S = \mu_0(1 - \alpha \cdot r)$ where $r$ is repair allocation extracted from the phenotype.

## Key Result

![Twin Worlds Experiment](figures/fig6_twin_worlds.png)

**Proposition 1 (Non-identifiability):** For any allele-based mortality pattern $\{\mu, \mu+\delta\}$, there exists an environment pair that produces identical patterns via plasticity. The models are distinguishable only through intervention experiments.

---

## Related Work

This paper builds on:
- [Todd (2025a)](https://doi.org/10.1016/j.biosystems.2025.105608) - Falsifiability and dimensional constraints
- [Todd (2025b)](https://doi.org/10.1016/j.biosystems.2025.105632) - Maxwell's demon and projection bounds

And responds to:
- [Sierra et al. (2025)](https://doi.org/10.1126/sciadv.adw0685) - Cooperation and cancer in mammals

---

## License

Code: MIT License

Paper: © 2025 Ian Todd. All rights reserved.
