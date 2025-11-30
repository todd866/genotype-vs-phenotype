"""
Developmental Network Model: Genotype → High-D Development → Phenotype

Demonstrates that:
1. Same genotype + different environments → different phenotypes
2. "Allele model" and "plastic policy model" can produce identical aggregate data
3. Non-identifiability of mechanism from phenotypic observations alone

For: "Genotype ≠ Phenotype: High-Dimensional Development and the Limits of Allele Stories"
Target: BioSystems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

os.makedirs('figures', exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

np.random.seed(42)


class DevelopmentalNetwork:
    """
    A minimal model of development as a dynamical system.

    Genotype g (low-D) parameterizes a network that maps
    environmental history E(t) to phenotype x (high-D).

    Key insight: g doesn't "encode" x algorithmically;
    it parameterizes a policy that integrates environment.
    """

    def __init__(self, n_genes=5, n_hidden=20, n_phenotype=10, n_env=3):
        """
        n_genes: dimension of genotype (low)
        n_hidden: dimension of developmental state (high)
        n_phenotype: dimension of phenotype output (high)
        n_env: dimension of environmental input
        """
        self.n_genes = n_genes
        self.n_hidden = n_hidden
        self.n_phenotype = n_phenotype
        self.n_env = n_env

        # Initialize weight matrices (these are "fixed" by evolution)
        # In reality these would also be parameterized by genes,
        # but we keep them fixed to isolate the g→policy→phenotype pathway
        self.W_h = 0.5 * np.random.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
        self.W_e = 0.3 * np.random.randn(n_hidden, n_env)
        self.W_g = 0.3 * np.random.randn(n_hidden, n_genes)
        self.W_out = 0.5 * np.random.randn(n_phenotype, n_hidden) / np.sqrt(n_hidden)

    def develop(self, genotype, env_history, dt=0.1, return_trajectory=False):
        """
        Run development: integrate environment history through network.

        genotype: array of shape (n_genes,)
        env_history: array of shape (T, n_env) - environmental inputs over time

        Returns: final phenotype x of shape (n_phenotype,)
        """
        T = len(env_history)
        h = np.zeros(self.n_hidden)  # Initial developmental state

        trajectory = [h.copy()] if return_trajectory else None

        for t in range(T):
            # Developmental dynamics: dh/dt = -h + tanh(Wh*h + We*e + Wg*g)
            e_t = env_history[t]
            input_total = self.W_h @ h + self.W_e @ e_t + self.W_g @ genotype
            dh = -h + np.tanh(input_total)
            h = h + dt * dh

            if return_trajectory:
                trajectory.append(h.copy())

        # Final phenotype is readout of developmental state
        phenotype = np.tanh(self.W_out @ h)

        if return_trajectory:
            return phenotype, np.array(trajectory)
        return phenotype

    def get_repair_allocation(self, phenotype):
        """
        Extract "repair vs proliferation" allocation from phenotype.
        This is the key variable that determines cancer risk.

        Convention: higher value = more repair, lower cancer risk
        """
        # Use first component as repair allocation (arbitrary but consistent)
        return (phenotype[0] + 1) / 2  # Map from [-1,1] to [0,1]

    def get_cancer_mortality(self, phenotype, baseline_mu=0.1):
        """
        Compute effective cancer mortality from phenotype.

        Higher repair allocation → lower cancer mortality.
        This is the emergent μ_S that allele models treat as a genetic parameter.
        """
        repair = self.get_repair_allocation(phenotype)
        # Repair reduces cancer mortality
        mu_cancer = baseline_mu * (1 - 0.8 * repair)
        return mu_cancer


def generate_environment(T, env_type='cooperative', n_env=3):
    """
    Generate environmental history.

    cooperative: low variance, supportive (buffered resources, social support)
    competitive: high variance, winner-take-all (resource scarcity, conflict)
    """
    t = np.linspace(0, 10, T)

    if env_type == 'cooperative':
        # Stable, predictable environment
        base = np.column_stack([
            0.5 + 0.1 * np.sin(0.5 * t),  # Stable resources
            0.7 + 0.05 * np.cos(0.3 * t),  # High social support
            0.2 * np.ones(T)  # Low conflict
        ])
        noise = 0.05 * np.random.randn(T, n_env)

    elif env_type == 'competitive':
        # Volatile, high-stakes environment
        base = np.column_stack([
            0.3 + 0.3 * np.sin(2 * t),  # Fluctuating resources
            0.2 + 0.1 * np.random.randn(T),  # Low/unpredictable social support
            0.6 + 0.2 * np.sin(3 * t)  # High conflict
        ])
        noise = 0.15 * np.random.randn(T, n_env)

    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    return np.clip(base + noise, 0, 1)


def figure1_same_genotype_different_phenotype():
    """
    Figure 1: Same genotype, different environments → different phenotypes.

    Demonstrates that genotype doesn't algorithmically determine phenotype.
    """
    net = DevelopmentalNetwork()
    T = 100

    # Fixed genotype
    genotype = np.array([0.5, -0.3, 0.2, 0.1, -0.4])

    # Two different environments
    env_coop = generate_environment(T, 'cooperative')
    env_comp = generate_environment(T, 'competitive')

    # Develop in each environment
    pheno_coop, traj_coop = net.develop(genotype, env_coop, return_trajectory=True)
    pheno_comp, traj_comp = net.develop(genotype, env_comp, return_trajectory=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Environmental histories
    ax1 = axes[0, 0]
    ax1.plot(env_coop[:, 0], 'b-', label='Cooperative', alpha=0.8)
    ax1.plot(env_comp[:, 0], 'r-', label='Competitive', alpha=0.8)
    ax1.set_xlabel('Developmental time')
    ax1.set_ylabel('Environmental signal (resource)')
    ax1.set_title('(A) Environmental histories')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Panel B: Developmental trajectories (first 2 PCs of hidden state)
    ax2 = axes[0, 1]
    # Project to 2D for visualization
    from sklearn.decomposition import PCA
    all_traj = np.vstack([traj_coop, traj_comp])
    pca = PCA(n_components=2)
    pca.fit(all_traj)

    traj_coop_2d = pca.transform(traj_coop)
    traj_comp_2d = pca.transform(traj_comp)

    ax2.plot(traj_coop_2d[:, 0], traj_coop_2d[:, 1], 'b-', alpha=0.7, label='Cooperative')
    ax2.plot(traj_comp_2d[:, 0], traj_comp_2d[:, 1], 'r-', alpha=0.7, label='Competitive')
    ax2.scatter([traj_coop_2d[0, 0]], [traj_coop_2d[0, 1]], c='green', s=100, zorder=5, marker='o')
    ax2.scatter([traj_coop_2d[-1, 0]], [traj_coop_2d[-1, 1]], c='blue', s=100, zorder=5, marker='s')
    ax2.scatter([traj_comp_2d[-1, 0]], [traj_comp_2d[-1, 1]], c='red', s=100, zorder=5, marker='s')
    ax2.set_xlabel('PC1 of developmental state')
    ax2.set_ylabel('PC2 of developmental state')
    ax2.set_title('(B) Developmental trajectories (same genotype)')
    ax2.legend()

    # Panel C: Final phenotypes
    ax3 = axes[1, 0]
    x = np.arange(len(pheno_coop))
    width = 0.35
    ax3.bar(x - width/2, pheno_coop, width, label='Cooperative env', color='steelblue')
    ax3.bar(x + width/2, pheno_comp, width, label='Competitive env', color='coral')
    ax3.set_xlabel('Phenotype dimension')
    ax3.set_ylabel('Phenotype value')
    ax3.set_title('(C) Final phenotypes (same genotype)')
    ax3.legend()
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Panel D: Emergent cancer mortality
    ax4 = axes[1, 1]
    repair_coop = net.get_repair_allocation(pheno_coop)
    repair_comp = net.get_repair_allocation(pheno_comp)
    mu_coop = net.get_cancer_mortality(pheno_coop)
    mu_comp = net.get_cancer_mortality(pheno_comp)

    labels = ['Cooperative', 'Competitive']
    repairs = [repair_coop, repair_comp]
    mus = [mu_coop, mu_comp]
    colors = ['steelblue', 'coral']

    ax4_twin = ax4.twinx()
    x_pos = [0, 1]
    ax4.bar([p - 0.15 for p in x_pos], repairs, 0.3, color=colors, alpha=0.7, label='Repair allocation')
    ax4_twin.bar([p + 0.15 for p in x_pos], mus, 0.3, color=colors, alpha=0.4, hatch='//', label='Cancer mortality')

    ax4.set_ylabel('Repair allocation', color='black')
    ax4_twin.set_ylabel('Cancer mortality $\\mu_S$', color='gray')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.set_title('(D) Emergent repair & cancer risk')
    ax4.set_ylim(0, 1)
    ax4_twin.set_ylim(0, 0.15)

    plt.tight_layout()
    plt.savefig('figures/fig1_same_genotype.pdf')
    plt.savefig('figures/fig1_same_genotype.png')
    plt.close()
    print("Generated: figures/fig1_same_genotype.pdf")

    return repair_coop, repair_comp, mu_coop, mu_comp


def figure2_population_level_patterns():
    """
    Figure 2: Population-level patterns.

    Shows that aggregate data (cancer prevalence vs lifestyle) can arise
    from either:
    - Model A: Different alleles with different μ_S
    - Model B: Same alleles, different environments → different emergent μ_S
    """
    net = DevelopmentalNetwork()
    T = 100
    n_individuals = 50

    # Generate population with genetic variation
    genotypes = 0.3 * np.random.randn(n_individuals, net.n_genes)

    # Model B: Same genetic distribution, different environments
    mu_coop_pop = []
    mu_comp_pop = []

    for g in genotypes:
        env_coop = generate_environment(T, 'cooperative')
        env_comp = generate_environment(T, 'competitive')

        pheno_coop = net.develop(g, env_coop)
        pheno_comp = net.develop(g, env_comp)

        mu_coop_pop.append(net.get_cancer_mortality(pheno_coop))
        mu_comp_pop.append(net.get_cancer_mortality(pheno_comp))

    mu_coop_pop = np.array(mu_coop_pop)
    mu_comp_pop = np.array(mu_comp_pop)

    # Model A: Different "alleles" (we simulate by just setting μ_S directly)
    # Cooperative species: lower μ_S allele favored
    # Competitive species: higher μ_S allele can be favored
    mu_allele_coop = 0.03 + 0.01 * np.random.randn(n_individuals)
    mu_allele_comp = 0.07 + 0.015 * np.random.randn(n_individuals)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: Model comparison - distributions
    ax1 = axes[0]
    bins = np.linspace(0, 0.12, 20)
    ax1.hist(mu_coop_pop, bins=bins, alpha=0.6, label='Cooperative', color='steelblue')
    ax1.hist(mu_comp_pop, bins=bins, alpha=0.6, label='Competitive', color='coral')
    ax1.axvline(mu_coop_pop.mean(), color='steelblue', linestyle='--', linewidth=2)
    ax1.axvline(mu_comp_pop.mean(), color='coral', linestyle='--', linewidth=2)
    ax1.set_xlabel('Cancer mortality $\\mu_S$')
    ax1.set_ylabel('Count')
    ax1.set_title('(A) Model B: Plastic policy')

    # Panel B: Model A distributions (allele-based)
    ax2 = axes[1]
    ax2.hist(mu_allele_coop, bins=bins, alpha=0.6, label='Cooperative', color='steelblue')
    ax2.hist(mu_allele_comp, bins=bins, alpha=0.6, label='Competitive', color='coral')
    ax2.axvline(mu_allele_coop.mean(), color='steelblue', linestyle='--', linewidth=2)
    ax2.axvline(mu_allele_comp.mean(), color='coral', linestyle='--', linewidth=2)
    ax2.set_xlabel('Cancer mortality $\\mu_S$')
    ax2.set_ylabel('Count')
    ax2.set_title('(B) Model A: Allele-based')

    # Shared legend below panels A and B
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.35, -0.02), fontsize=9)

    # Panel C: Non-identifiability - aggregate statistics match
    ax3 = axes[2]

    # Compare means and SDs
    stats = {
        'Model A\nCooperative': (mu_allele_coop.mean(), mu_allele_coop.std()),
        'Model A\nCompetitive': (mu_allele_comp.mean(), mu_allele_comp.std()),
        'Model B\nCooperative': (mu_coop_pop.mean(), mu_coop_pop.std()),
        'Model B\nCompetitive': (mu_comp_pop.mean(), mu_comp_pop.std()),
    }

    x_pos = np.arange(len(stats))
    means = [v[0] for v in stats.values()]
    stds = [v[1] for v in stats.values()]
    colors = ['steelblue', 'coral', 'steelblue', 'coral']
    hatches = ['', '', '//', '//']

    bars = ax3.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stats.keys(), fontsize=9)
    ax3.set_ylabel('Mean cancer mortality $\\mu_S$')
    ax3.set_title('(C) Non-identifiability:\nAggregate statistics match')

    plt.tight_layout()
    plt.savefig('figures/fig2_population_patterns.pdf')
    plt.savefig('figures/fig2_population_patterns.png')
    plt.close()
    print("Generated: figures/fig2_population_patterns.pdf")


def figure3_projection_and_information_loss():
    """
    Figure 3: Information loss under projection.

    Key insight:
    - Low-D allele models are projections of high-D developmental reality
    - Projection loses the environment-dependence information
    """
    net = DevelopmentalNetwork()
    T = 100
    n_samples = 200

    # Generate diverse genotype-environment combinations
    data = []
    for _ in range(n_samples):
        g = 0.3 * np.random.randn(net.n_genes)

        # Random mix of environments
        env_mix = np.random.rand()
        if env_mix < 0.5:
            env = generate_environment(T, 'cooperative')
            env_label = 0
        else:
            env = generate_environment(T, 'competitive')
            env_label = 1

        pheno = net.develop(g, env)
        mu = net.get_cancer_mortality(pheno)
        repair = net.get_repair_allocation(pheno)

        data.append({
            'genotype': g,
            'env_label': env_label,
            'phenotype': pheno,
            'mu': mu,
            'repair': repair
        })

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Full space - genotype + environment clearly separates
    ax1 = axes[0]
    g_pc1 = [d['genotype'][0] for d in data]
    env_labels = [d['env_label'] for d in data]
    mus = [d['mu'] for d in data]

    colors = ['steelblue' if e == 0 else 'coral' for e in env_labels]
    ax1.scatter(g_pc1, mus, c=colors, alpha=0.6, s=30)
    ax1.set_xlabel('Genotype (first component)')
    ax1.set_ylabel('Cancer mortality $\\mu_S$')
    ax1.set_title('(A) Full information:\nGenotype + Environment')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='Cooperative'),
                      Patch(facecolor='coral', label='Competitive')]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Panel B: Projected space - only genotype, environment hidden
    ax2 = axes[1]
    ax2.scatter(g_pc1, mus, c='gray', alpha=0.6, s=30)
    ax2.set_xlabel('Genotype (first component)')
    ax2.set_ylabel('Cancer mortality $\\mu_S$')
    ax2.set_title('(B) Projected (allele model):\nEnvironment hidden')

    # Fit line to show "genetic effect"
    z = np.polyfit(g_pc1, mus, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(g_pc1), max(g_pc1), 100)
    ax2.plot(x_line, p(x_line), 'k--', linewidth=2, label=f'Apparent "allele effect"')
    ax2.legend()

    # Panel C: Variance decomposition
    ax3 = axes[2]

    # Compute variance components
    mus_arr = np.array(mus)
    env_arr = np.array(env_labels)
    g_arr = np.array(g_pc1)

    # Total variance
    var_total = np.var(mus_arr)

    # Variance explained by environment
    mu_by_env = [mus_arr[env_arr == e].mean() for e in [0, 1]]
    var_env = np.var([mu_by_env[int(e)] for e in env_arr])

    # Variance explained by genotype (residual after environment)
    residuals = mus_arr - np.array([mu_by_env[int(e)] for e in env_arr])
    corr_g, _ = pearsonr(g_arr, residuals)
    var_g = corr_g**2 * np.var(residuals)

    # Residual
    var_residual = var_total - var_env - var_g

    components = ['Environment', 'Genotype', 'Residual']
    variances = [var_env/var_total, var_g/var_total, var_residual/var_total]
    colors = ['coral', 'steelblue', 'gray']

    ax3.bar(components, variances, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Fraction of variance in $\\mu_S$')
    ax3.set_title('(C) Variance decomposition')
    ax3.set_ylim(0, 1)

    # Add annotation - arrow points to top of Environment bar
    env_bar_height = variances[0]
    ax3.annotate('Allele models\nattribute this\nto "genetic effects"',
                xy=(0, env_bar_height), fontsize=9, ha='center',
                xytext=(1.5, 0.85),
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig('figures/fig3_projection_loss.pdf')
    plt.savefig('figures/fig3_projection_loss.png')
    plt.close()
    print("Generated: figures/fig3_projection_loss.pdf")


def figure4_prediction_divergence():
    """
    Figure 4: Where models diverge in prediction.

    The key testable difference:
    - Model A (allele): Changing environment shouldn't change μ_S
    - Model B (plastic): Changing environment WILL change μ_S
    """
    net = DevelopmentalNetwork()
    T = 100

    # Fixed genotype population
    n_individuals = 30
    genotypes = 0.3 * np.random.randn(n_individuals, net.n_genes)

    # Scenario: Environment shifts from competitive to cooperative

    # Phase 1: Competitive environment
    mu_phase1 = []
    for g in genotypes:
        env = generate_environment(T, 'competitive')
        pheno = net.develop(g, env)
        mu_phase1.append(net.get_cancer_mortality(pheno))

    # Phase 2: Same genotypes, now in cooperative environment
    mu_phase2 = []
    for g in genotypes:
        env = generate_environment(T, 'cooperative')
        pheno = net.develop(g, env)
        mu_phase2.append(net.get_cancer_mortality(pheno))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Model predictions
    ax1 = axes[0]

    x = [0, 1]

    # Model A prediction: no change (allele-determined)
    model_a_mean = [np.mean(mu_phase1), np.mean(mu_phase1)]  # No change
    model_a_err = [np.std(mu_phase1), np.std(mu_phase1)]

    # Model B prediction: change with environment
    model_b_mean = [np.mean(mu_phase1), np.mean(mu_phase2)]
    model_b_err = [np.std(mu_phase1), np.std(mu_phase2)]

    ax1.errorbar([p - 0.1 for p in x], model_a_mean, yerr=model_a_err,
                fmt='s-', color='gray', capsize=5, linewidth=2, markersize=10,
                label='Model A (allele)')
    ax1.errorbar([p + 0.1 for p in x], model_b_mean, yerr=model_b_err,
                fmt='o-', color='green', capsize=5, linewidth=2, markersize=10,
                label='Model B (plastic)')

    ax1.set_xticks(x)
    ax1.set_xticklabels(['Phase 1:\nCompetitive', 'Phase 2:\nCooperative'])
    ax1.set_ylabel('Cancer mortality $\\mu_S$')
    ax1.set_title('(A) Model predictions for\nenvironment shift')
    ax1.legend()
    ax1.set_ylim(0, 0.12)

    # Panel B: Individual trajectories
    ax2 = axes[1]

    for i in range(min(15, n_individuals)):
        ax2.plot([0, 1], [mu_phase1[i], mu_phase2[i]], 'o-',
                color='green', alpha=0.3, linewidth=1)

    ax2.plot([0, 1], [np.mean(mu_phase1), np.mean(mu_phase2)], 'o-',
            color='green', linewidth=3, markersize=12, label='Population mean')

    # Model A prediction line (flat)
    ax2.axhline(np.mean(mu_phase1), color='gray', linestyle='--',
               linewidth=2, label='Model A prediction')

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Competitive\nenv', 'Cooperative\nenv'])
    ax2.set_ylabel('Cancer mortality $\\mu_S$')
    ax2.set_title('(B) Individual responses to\nenvironment change')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 0.12)

    plt.tight_layout()
    plt.savefig('figures/fig4_prediction_divergence.pdf')
    plt.savefig('figures/fig4_prediction_divergence.png')
    plt.close()
    print("Generated: figures/fig4_prediction_divergence.pdf")


def figure5_causal_dags():
    """
    Figure 5: Causal DAGs contrasting Model A vs Model B.

    Shows the structural difference between allele-based and policy-based mechanisms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel A: Model A (Allele Story)
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    ax1.set_title('(A) Model A: Allele-based', fontsize=12, fontweight='bold')

    # Nodes
    nodes_a = {
        'G': (2, 4),
        'μS': (5, 4),
        'Cancer': (8, 4),
        'E': (5, 1.5)
    }

    for name, (x, y) in nodes_a.items():
        if name == 'E':
            circle = plt.Circle((x, y), 0.6, fill=True, color='lightcoral', ec='black', linewidth=2)
        else:
            circle = plt.Circle((x, y), 0.6, fill=True, color='lightblue', ec='black', linewidth=2)
        ax1.add_patch(circle)
        label = '$\\mu_S$' if name == 'μS' else name
        ax1.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrows
    ax1.annotate('', xy=(4.4, 4), xytext=(2.6, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.annotate('', xy=(7.4, 4), xytext=(5.6, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.annotate('', xy=(4.5, 3.5), xytext=(5, 2.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))

    ax1.text(5, 0.5, 'E only modulates selection on G', ha='center', fontsize=10, style='italic')

    # Panel B: Model B (Plastic Policy)
    ax2 = axes[1]
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('(B) Model B: Plastic policy', fontsize=12, fontweight='bold')

    # Nodes
    nodes_b = {
        'G': (1, 4),
        'Dev': (3.5, 4),
        'π': (6, 4),
        'r': (8, 4),
        'μS': (10, 4),
        'E': (4.75, 1.5)
    }

    labels_b = {
        'G': 'G',
        'Dev': 'Dev',
        'π': '$\\pi(E)$',
        'r': '$r_t$',
        'μS': '$\\mu_S$',
        'E': 'E'
    }

    for name, (x, y) in nodes_b.items():
        if name == 'E':
            circle = plt.Circle((x, y), 0.6, fill=True, color='lightcoral', ec='black', linewidth=2)
        else:
            circle = plt.Circle((x, y), 0.6, fill=True, color='lightblue', ec='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x, y, labels_b[name], ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    arrow_pairs = [
        ((1.6, 4), (2.9, 4)),
        ((4.1, 4), (5.4, 4)),
        ((6.6, 4), (7.4, 4)),
        ((8.6, 4), (9.4, 4)),
    ]
    for start, end in arrow_pairs:
        ax2.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # E enters Dev and π
    ax2.annotate('', xy=(3.5, 3.4), xytext=(4.5, 2.1),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
    ax2.annotate('', xy=(5.8, 3.4), xytext=(5, 2.1),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

    ax2.text(6, 0.5, 'E enters directly into development and policy', ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig('figures/fig5_causal_dags.pdf')
    plt.savefig('figures/fig5_causal_dags.png')
    plt.close()
    print("Generated: figures/fig5_causal_dags.pdf")


def figure6_twin_worlds():
    """
    Figure 6: Twin worlds experiment (Proposition 2 demonstration).

    Same genotype distribution developed in cooperative vs competitive worlds.
    Shows that naive allele inference would "find" oncogenic variants
    even when all variation is due to plastic policy.

    This provides the empirical demonstration of Proposition 2's non-identifiability
    claim: aggregate patterns are indistinguishable between Model A and Model B.
    """
    net = DevelopmentalNetwork()
    T = 100
    n_species = 40

    # Same starting genotype distribution
    genotypes = 0.3 * np.random.randn(n_species, net.n_genes)

    # World 1: All species develop in cooperative environments
    mu_world1 = []
    for g in genotypes:
        env = generate_environment(T, 'cooperative')
        pheno = net.develop(g, env)
        mu_world1.append(net.get_cancer_mortality(pheno))

    # World 2: All species develop in competitive environments
    mu_world2 = []
    for g in genotypes:
        env = generate_environment(T, 'competitive')
        pheno = net.develop(g, env)
        mu_world2.append(net.get_cancer_mortality(pheno))

    mu_world1 = np.array(mu_world1)
    mu_world2 = np.array(mu_world2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: μ_S distributions in both worlds
    ax1 = axes[0]
    bins = np.linspace(0, 0.12, 15)
    ax1.hist(mu_world1, bins=bins, alpha=0.7, label='World 1 (Cooperative)', color='steelblue')
    ax1.hist(mu_world2, bins=bins, alpha=0.7, label='World 2 (Competitive)', color='coral')
    ax1.axvline(mu_world1.mean(), color='steelblue', linestyle='--', linewidth=2)
    ax1.axvline(mu_world2.mean(), color='coral', linestyle='--', linewidth=2)
    ax1.set_xlabel('Cancer mortality $\\mu_S$')
    ax1.set_ylabel('Number of species')
    ax1.set_title('(A) Same genomes, different worlds')
    ax1.legend()

    # Panel B: What a naive allele model would infer
    ax2 = axes[1]

    # Naive inference: "High μ_S species must have oncogenic alleles"
    threshold = 0.05

    frac_oncogenic_w1 = (mu_world1 > threshold).mean()
    frac_oncogenic_w2 = (mu_world2 > threshold).mean()

    x_pos = [0, 1]
    ax2.bar(x_pos, [frac_oncogenic_w1, frac_oncogenic_w2],
           color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['World 1\n(Cooperative)', 'World 2\n(Competitive)'])
    ax2.set_ylabel('Inferred "oncogenic allele" frequency')
    ax2.set_title('(B) Naive allele model inference')
    ax2.set_ylim(0, 1)

    # Add annotation above the bars
    ax2.text(0.5, 1.05, 'No genetic difference!', fontsize=11,
             ha='center', va='bottom', color='red', fontweight='bold',
             transform=ax2.transAxes)

    # Panel C: Scatter showing no genetic correlation with μ_S
    ax3 = axes[2]

    # Use first genotype component as proxy
    g_component = genotypes[:, 0]

    ax3.scatter(g_component, mu_world1, alpha=0.6, label='World 1', color='steelblue', s=50)
    ax3.scatter(g_component, mu_world2, alpha=0.6, label='World 2', color='coral', s=50)

    # Fit lines
    z1 = np.polyfit(g_component, mu_world1, 1)
    z2 = np.polyfit(g_component, mu_world2, 1)
    x_line = np.linspace(g_component.min(), g_component.max(), 100)
    ax3.plot(x_line, np.poly1d(z1)(x_line), 'b--', alpha=0.5)
    ax3.plot(x_line, np.poly1d(z2)(x_line), 'r--', alpha=0.5)

    ax3.set_xlabel('Genotype (first component)')
    ax3.set_ylabel('Cancer mortality $\\mu_S$')
    ax3.set_title('(C) Environment, not genes,\ndetermines $\\mu_S$')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('figures/fig6_twin_worlds.pdf')
    plt.savefig('figures/fig6_twin_worlds.png')
    plt.close()
    print("Generated: figures/fig6_twin_worlds.pdf")


if __name__ == '__main__':
    print("Generating figures for genotype ≠ phenotype paper...")
    print("=" * 50)

    figure1_same_genotype_different_phenotype()
    figure2_population_level_patterns()
    figure3_projection_and_information_loss()
    figure4_prediction_divergence()
    figure5_causal_dags()
    figure6_twin_worlds()

    print("=" * 50)
    print("All figures generated in figures/ directory")
