"""
Multilevel Simulation: The Fractal Cooperation Engine
Extension of Nonergodic Development to Evolutionary Time

Demonstrates:
1. Developmental Coherence (c) acts as a "fractal" trait.
2. Cooperative regimes amplify Between-Group selection.
3. Competitive regimes amplify Within-Group selection.

Note: This script contains a local copy of DevelopmentalNetwork with a simplified
interface for the evolutionary simulation. The core developmental dynamics
(Equation 2 in the paper) are identical to nonergodic_development.py.
Parameters: n_genes=5, n_hidden=20, n_phenotype=10, n_env=3.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os

os.makedirs('figures', exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.figsize': (12, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


# --- Core Physics (same dynamics as nonergodic_development.py) ---

class DevelopmentalNetwork:
    """
    Minimal developmental network - the "physics" of the system.
    Implements Equation 1: dh/dt = -h + tanh(W_h·h + W_e·e_t + W_g·g)
    Discretized with Euler's method (dt=0.1).

    Parameters match nonergodic_development.py exactly.
    """

    def __init__(self, n_genes=5, n_hidden=20, n_phenotype=10, n_env=3):
        self.n_genes = n_genes
        self.n_hidden = n_hidden
        self.n_env = n_env
        self.W_h = 0.5 * np.random.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
        self.W_e = 0.3 * np.random.randn(n_hidden, n_env)
        self.W_g = 0.3 * np.random.randn(n_hidden, n_genes)
        self.W_out = 0.5 * np.random.randn(n_phenotype, n_hidden) / np.sqrt(n_hidden)

    def develop(self, genotype, env_history, dt=0.1):
        h = np.zeros(self.n_hidden)
        for e_t in env_history:
            input_total = self.W_h @ h + self.W_e @ e_t + self.W_g @ genotype
            h += dt * (-h + np.tanh(input_total))
        return np.tanh(self.W_out @ h)

    def get_coherence(self, phenotype):
        """Maps phenotype to [0,1]. High coherence = Low Cancer AND High Cooperation."""
        return (phenotype[0] + 1) / 2

    def get_cancer_mortality(self, coherence, baseline_mu=0.1):
        """Simplified interface: takes coherence directly (already extracted)."""
        return baseline_mu * (1 - 0.8 * coherence)


def generate_group_environment(T, coherence_level, n_env=3):
    """
    Higher group coherence = More stable environment.
    This is how social cooperation shapes developmental context.
    """
    stability = coherence_level  # 0 = Chaos, 1 = Stable

    # Resource buffering (high stability = low noise, better resources)
    noise_mag = 0.2 * (1 - stability)
    base = 0.5 + 0.3 * stability

    env = base + noise_mag * np.random.randn(T, n_env)
    return np.clip(env, 0, 1)


# --- 2. The Evolutionary Agents ---

class Organism:
    """An individual organism with genome, phenotype, and fitness."""

    def __init__(self, genome=None, n_genes=5):
        if genome is None:
            self.genome = 0.1 * np.random.randn(n_genes)
        else:
            self.genome = genome

        self.phenotype = None
        self.coherence = 0.0
        self.cancer_mu = 0.0
        self.fitness = 0.0

    def develop(self, group_coherence, net):
        """Develop phenotype in environment shaped by group's state."""
        env = generate_group_environment(50, group_coherence)
        self.phenotype = net.develop(self.genome, env)

        self.coherence = net.get_coherence(self.phenotype)
        self.cancer_mu = net.get_cancer_mortality(self.coherence)

    def calculate_fitness(self, group_coherence):
        """
        Individual fitness w_i:
        - Benefit from group cooperation (beta * C_g * c_i) - synergy!
        - Cost of cancer mortality (k * mu_i)

        Key insight: Cooperation creates synergy - high c pays off MORE in high C groups.
        This is the multilevel selection mechanism.
        """
        base_w = 1.0
        # Synergistic benefit: cooperation pays off more in cooperative groups
        synergy = 3.0 * group_coherence * self.coherence
        # Cancer kills
        cancer_cost = 8.0 * self.cancer_mu

        self.fitness = max(0.01, base_w + synergy - cancer_cost)
        return self.fitness


class Group:
    """A group of organisms with emergent group-level properties."""

    def __init__(self, organisms):
        self.organisms = organisms
        self.group_coherence = 0.5  # Initial "climate"
        self.fitness = 0.0

    def update_state(self, net):
        """
        1. Develop all organisms based on current group climate
        2. Update group climate (leaky integrator of member behavior)
        3. Calculate fitnesses
        """
        # 1. Develop organisms
        trait_sum = 0
        for org in self.organisms:
            org.develop(self.group_coherence, net)
            trait_sum += org.coherence

        # 2. Update group climate - creates the social attractor
        current_mean_c = trait_sum / len(self.organisms)
        self.group_coherence = 0.8 * self.group_coherence + 0.2 * current_mean_c

        # 3. Calculate fitnesses
        fit_sum = 0
        for org in self.organisms:
            fit_sum += org.calculate_fitness(self.group_coherence)

        # Group fitness (group selection acts on this)
        self.fitness = fit_sum / len(self.organisms)


# --- 3. The Evolutionary Engine (Price Equation) ---

def run_simulation(n_groups=30, orgs_per_group=15, generations=80,
                   mutation_rate=0.1, seed=42):
    """
    Run multilevel selection simulation with Price equation decomposition.
    """
    np.random.seed(seed)

    net = DevelopmentalNetwork()

    # Initialize population
    groups = [Group([Organism() for _ in range(orgs_per_group)])
              for _ in range(n_groups)]

    # Data logging
    history = {
        'gen': [],
        'mean_coherence': [],
        'mean_cancer': [],
        'price_between': [],
        'price_within': [],
        'price_total': []
    }

    print(f"Running simulation: {n_groups} groups, {generations} generations...")

    for gen in range(generations):
        # A. Development & Evaluation
        all_fitness = []
        all_coherence = []
        all_cancer = []

        group_W = []      # Group fitnesses
        group_Z = []      # Group mean traits (coherence)
        group_cov_wz = [] # Within-group covariances

        for g in groups:
            g.update_state(net)

            # Collect data for Price equation
            ws = np.array([o.fitness for o in g.organisms])
            zs = np.array([o.coherence for o in g.organisms])
            mus = np.array([o.cancer_mu for o in g.organisms])

            group_W.append(g.fitness)
            group_Z.append(np.mean(zs))

            # Within-group covariance Cov(w, z)
            if np.var(zs) > 0 and len(zs) > 1:
                group_cov_wz.append(np.cov(ws, zs)[0, 1])
            else:
                group_cov_wz.append(0.0)

            all_fitness.extend(ws)
            all_coherence.extend(zs)
            all_cancer.extend(mus)

        # B. Calculate Price Equation Terms
        # Δz̄ = Cov(W_g, Z_g)/W̄ + E[Cov(w_ig, z_ig)]/W̄
        W_bar = np.mean(all_fitness)

        if W_bar > 0:
            # Between-group selection
            if len(group_W) > 1:
                cov_between = np.cov(group_W, group_Z)[0, 1]
            else:
                cov_between = 0
            term_between = cov_between / W_bar

            # Within-group selection (average of within-group covariances)
            term_within = np.mean(group_cov_wz) / W_bar
        else:
            term_between = 0
            term_within = 0

        # Log statistics
        history['gen'].append(gen)
        history['mean_coherence'].append(np.mean(all_coherence))
        history['mean_cancer'].append(np.mean(all_cancer))
        history['price_between'].append(term_between)
        history['price_within'].append(term_within)
        history['price_total'].append(term_between + term_within)

        # C. Reproduction

        # Group selection: sample groups proportional to fitness
        group_probs = np.array(group_W)
        if group_probs.sum() <= 0:
            group_probs = np.ones_like(group_probs)
        group_probs = group_probs / group_probs.sum()

        new_groups = []

        for _ in range(n_groups):
            # Pick parent group
            parent_g_idx = np.random.choice(len(groups), p=group_probs)
            parent_g = groups[parent_g_idx]

            # Individual selection within chosen group
            org_probs = np.array([o.fitness for o in parent_g.organisms])
            if org_probs.sum() <= 0:
                org_probs = np.ones_like(org_probs)
            org_probs = org_probs / org_probs.sum()

            new_orgs = []
            for _ in range(orgs_per_group):
                parent_idx = np.random.choice(len(parent_g.organisms), p=org_probs)
                parent_o = parent_g.organisms[parent_idx]

                # Offspring with mutation
                child_genome = deepcopy(parent_o.genome)
                child_genome += mutation_rate * np.random.randn(len(child_genome))
                new_orgs.append(Organism(child_genome))

            new_g = Group(new_orgs)
            new_g.group_coherence = parent_g.group_coherence  # Inherit climate
            new_groups.append(new_g)

        groups = new_groups

        if gen % 10 == 0:
            print(f"  Gen {gen:3d}: c̄={history['mean_coherence'][-1]:.3f}, "
                  f"μ̄={history['mean_cancer'][-1]:.4f}, "
                  f"S_between={term_between:+.4f}, S_within={term_within:+.4f}")

    return history


def plot_results(history):
    """Generate Figure 7: Fractal Cooperation and Price Equation."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    gens = history['gen']

    # Panel A: Evolution of coherence
    ax1 = axes[0]
    ax1.plot(gens, history['mean_coherence'], 'g-', lw=2.5)
    ax1.fill_between(gens, 0, history['mean_coherence'], alpha=0.3, color='green')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean developmental coherence $\\bar{c}$')
    ax1.set_title('(A) Evolution of fractal coherence')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Panel B: Price equation decomposition
    ax2 = axes[1]
    ax2.plot(gens, history['price_between'], 'b-', lw=2, label='$S_{between}$ (group)')
    ax2.plot(gens, history['price_within'], 'r--', lw=2, label='$S_{within}$ (individual)')
    ax2.plot(gens, history['price_total'], 'k:', lw=1.5, label='Total $\\Delta\\bar{c}$')
    ax2.axhline(0, color='gray', lw=1, alpha=0.5)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Selection gradient')
    ax2.set_title('(B) Price equation decomposition')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel C: Cancer mortality decreases
    ax3 = axes[2]
    ax3.plot(gens, history['mean_cancer'], 'purple', lw=2.5)
    ax3.fill_between(gens, 0, history['mean_cancer'], alpha=0.3, color='purple')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Mean cancer mortality $\\bar{\\mu}_S$')
    ax3.set_title('(C) Cancer suppression follows cooperation')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig7_fractal_price.pdf')
    plt.savefig('figures/fig7_fractal_price.png')
    plt.close()
    print("Generated: figures/fig7_fractal_price.pdf")


def run_regime_comparison():
    """
    Compare evolution under different selection regimes.
    Shows that group selection is necessary for cooperation to evolve.
    """
    print("\nRunning regime comparison...")

    regimes = {
        'Strong group selection': {'group_weight': 2.0, 'ind_weight': 0.3},
        'Balanced selection': {'group_weight': 1.0, 'ind_weight': 1.0},
        'Strong individual selection': {'group_weight': 0.1, 'ind_weight': 2.0}
    }

    results = {}

    for name, params in regimes.items():
        print(f"  Running: {name}")
        np.random.seed(42)

        net = DevelopmentalNetwork()
        n_groups = 25
        orgs_per_group = 12
        generations = 70

        groups = [Group([Organism() for _ in range(orgs_per_group)])
                  for _ in range(n_groups)]

        coherence_history = []
        cancer_history = []

        for gen in range(generations):
            all_coherence = []
            all_cancer = []
            group_W = []

            for g in groups:
                g.update_state(net)
                group_W.append(g.fitness)
                for o in g.organisms:
                    all_coherence.append(o.coherence)
                    all_cancer.append(o.cancer_mu)

            coherence_history.append(np.mean(all_coherence))
            cancer_history.append(np.mean(all_cancer))

            # Modified selection based on regime
            gw = params['group_weight']
            iw = params['ind_weight']

            group_probs = np.array(group_W) ** gw
            if group_probs.sum() <= 0:
                group_probs = np.ones_like(group_probs)
            group_probs = group_probs / group_probs.sum()

            new_groups = []
            for _ in range(n_groups):
                parent_g_idx = np.random.choice(len(groups), p=group_probs)
                parent_g = groups[parent_g_idx]

                org_fits = np.array([o.fitness for o in parent_g.organisms]) ** iw
                if org_fits.sum() <= 0:
                    org_fits = np.ones_like(org_fits)
                org_probs = org_fits / org_fits.sum()

                new_orgs = []
                for _ in range(orgs_per_group):
                    parent_idx = np.random.choice(len(parent_g.organisms), p=org_probs)
                    child_genome = deepcopy(parent_g.organisms[parent_idx].genome)
                    child_genome += 0.05 * np.random.randn(len(child_genome))
                    new_orgs.append(Organism(child_genome))

                new_g = Group(new_orgs)
                new_g.group_coherence = parent_g.group_coherence
                new_groups.append(new_g)

            groups = new_groups

        results[name] = {
            'coherence': coherence_history,
            'cancer': cancer_history
        }

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = {
        'Strong group selection': 'green',
        'Balanced selection': 'orange',
        'Strong individual selection': 'gray'
    }

    gens = range(generations)

    # Panel A: Coherence trajectories
    ax1 = axes[0]
    for name, data in results.items():
        ax1.plot(gens, data['coherence'], color=colors[name], lw=2, label=name)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean coherence $\\bar{c}$')
    ax1.set_title('(A) Cooperation requires group selection')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Panel B: Cancer trajectories
    ax2 = axes[1]
    for name, data in results.items():
        ax2.plot(gens, data['cancer'], color=colors[name], lw=2, label=name)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Mean cancer mortality $\\bar{\\mu}_S$')
    ax2.set_title('(B) Cancer suppression follows cooperation')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig8_regime_comparison.pdf')
    plt.savefig('figures/fig8_regime_comparison.png')
    plt.close()
    print("Generated: figures/fig8_regime_comparison.pdf")

    return results


if __name__ == "__main__":
    # Main simulation with Price equation
    history = run_simulation(n_groups=30, orgs_per_group=10, generations=60)
    plot_results(history)

    # Regime comparison
    run_regime_comparison()

    print("\nDone! Check figures/ for fig7_fractal_price.pdf and fig8_regime_comparison.pdf")
