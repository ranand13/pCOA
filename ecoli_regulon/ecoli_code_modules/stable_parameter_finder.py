"""
Stable Parameter Finder for E. coli CRP Regulon

Searches for ODE parameter sets that:
1. Produce stable steady states (negative eigenvalues)
2. Are within biologically realistic ranges (literature priors)
3. Remain stable under various interventions
4. Can be used to generate synthetic training data for SCM

Uses forward ODE simulation (ODESimulator) to reach steady state.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import sys, os

from ode_simulator import ODESimulator


class StableParameterFinder:
    
    def __init__(self, pkn, form='hill', cache_dir='../ecoli_crp_data/cache'):
        """
        Initialize parameter finder
        
        Args:
            pkn: PKN dictionary with network structure
            form: Kinetic form ('hill' for transcriptional regulation)
            cache_dir: Where to save/load stable parameters
        """
        self.pkn = pkn
        self.form = form
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ode_sim = ODESimulator(pkn)
        
    def get_literature_ranges(self):
        """
        Get literature-based parameter ranges for E. coli transcriptional regulation
        
        Returns:
            dict: Parameter ranges based on literature
        """
        ranges = {}
        
        # Basal transcription rates (mRNA/min)
        # Literature: 0.1-0.5 mRNA/min for bacterial genes
        for gene in self.pkn['variables']:
            ranges[f'k_basal_{gene}'] = (0.05, 0.8)
        
        # Degradation rates (1/min)
        # Literature: E. coli mRNA half-life ~3-8 min → k_deg = ln(2)/t_half ≈ 0.09-0.23
        for gene in self.pkn['variables']:
            ranges[f'k_deg_{gene}'] = (0.05, 0.5)
        
        # Hill parameters for TF regulation
        for tf, target, reg_type in self.pkn['edges']:
            # Max transcription rate (fold change over basal)
            # Literature: TF-mediated activation typically 2-20 fold
            ranges[f'V_{tf}_{target}'] = (0.5, 3.0)
            
            # Half-saturation constant (TF concentration at half-max)
            # Literature: typical KD for TF-DNA binding ~0.1-10 nM, normalized ~0.3-2.0
            ranges[f'K_{tf}_{target}'] = (0.2, 2.5)
            
            # Hill coefficient (cooperativity)
            # Literature: n=1 (no cooperativity) to n=4 (high cooperativity)
            # Most bacterial TFs have n=2-3
            ranges[f'n_{tf}_{target}'] = (1.5, 4.0)
        
        return ranges
    
    def sample_parameters(self, param_ranges, seed=None):
        """
        Sample one parameter set from ranges
        
        Args:
            param_ranges: Dict of parameter ranges
            seed: Random seed
        
        Returns:
            dict: Sampled parameters
        """
        if seed is not None:
            np.random.seed(seed)
        
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name.startswith('n_'):
                # Hill coefficients: discrete values
                params[param_name] = np.random.choice([2.0, 3.0, 4.0])
            else:
                # Other parameters: continuous uniform
                params[param_name] = np.random.uniform(min_val, max_val)
        
        return params
    
    def _simulate_to_steady_state(self, params, interventions, t_max=100, dt=0.1):
        """
        Run forward ODE simulation to reach steady state
        
        Uses ODESimulator.build_solver() interface
        
        Args:
            params: Parameter dictionary
            interventions: Intervention dictionary (includes exogenous + I_X values)
            t_max: Maximum simulation time (not used with build_solver)
            dt: Time step (not used with build_solver)
        
        Returns:
            dict: Steady-state concentrations or None if failed
        """
        try:
            # Build solver function
            solver = self.ode_sim.build_solver(self.form, params)
            
            # Extract k_val and I values from interventions
            k_val = interventions.get(self.pkn['exogenous'], 1.0)
            I_kwargs = {key: val for key, val in interventions.items() 
                       if key != self.pkn['exogenous']}
            
            # Call solver to get steady state
            steady_state = solver(k_val, **I_kwargs)
            
            return steady_state
            
        except Exception as e:
            return None
    
    def _compute_jacobian(self, params, steady_state):
        """
        Compute Jacobian matrix at steady state
        
        Args:
            params: Parameter dictionary
            steady_state: Steady-state concentrations
        
        Returns:
            np.array: Jacobian matrix
        """
        n = len(self.pkn['variables'])
        J = np.zeros((n, n))
        
        gene_to_idx = {gene: i for i, gene in enumerate(self.pkn['variables'])}
        
        for i, gene_i in enumerate(self.pkn['variables']):
            # Degradation (diagonal)
            k_deg = params.get(f'k_deg_{gene_i}', 0.1)
            J[i, i] -= k_deg
            
            # Regulation terms
            for src, tgt, reg_type in self.pkn['edges']:
                if tgt != gene_i or src not in self.pkn['variables']:
                    continue
                
                j = gene_to_idx[src]
                src_conc = steady_state[src]
                
                V = params.get(f'V_{src}_{tgt}', 0)
                K = params.get(f'K_{src}_{tgt}', 1.0)
                n = params.get(f'n_{src}_{tgt}', 2.0)
                
                # Derivative of Hill function
                if reg_type == 'activate':
                    # d/dx[V * x^n / (K^n + x^n)]
                    numerator = V * n * (src_conc**(n-1)) * (K**n)
                    denominator = (K**n + src_conc**n)**2
                else:  # repress
                    # d/dx[V * K^n / (K^n + x^n)]
                    numerator = -V * (K**n) * n * (src_conc**(n-1))
                    denominator = (K**n + src_conc**n)**2
                
                J[i, j] += numerator / denominator
        
        return J
    
    def check_stability(self, params, test_interventions=None, verbose=False):
        """
        Check if parameter set produces stable steady states
        
        Uses forward ODE simulation to reach steady state.
        Tests stability under baseline AND interventions.
        
        Args:
            params: Parameter dictionary
            test_interventions: List of intervention dicts to test
            verbose: Print diagnostics
        
        Returns:
            tuple: (is_stable, baseline_state, eigenvalues)
        """
        # Default test cases: baseline + a few knockdowns
        if test_interventions is None:
            test_interventions = [
                {self.pkn['exogenous']: 1.0},  # Baseline
            ]
            # Add a few test knockdowns
            test_genes = list(self.pkn['inhibitions'].keys())[:3]
            for gene in test_genes:
                test_interventions.append({
                    self.pkn['exogenous']: 1.0,
                    self.pkn['inhibitions'][gene]: 0.5  # 50% knockdown
                })
        
        # Test each intervention
        for interventions in test_interventions:
            # Run forward simulation to steady state
            steady_state = self._simulate_to_steady_state(params, interventions)
            
            if steady_state is None:
                if verbose:
                    print(f"  ✗ Failed to reach steady state")
                return False, None, None
            
            # Check all concentrations are positive and reasonable
            for gene in self.pkn['variables']:
                if gene not in steady_state or steady_state[gene] <= 0:
                    if verbose:
                        print(f"  ✗ Invalid concentration for {gene}")
                    return False, steady_state, None
                
                # Check not too high (unrealistic)
                if steady_state[gene] > 100:
                    if verbose:
                        print(f"  ✗ Unrealistically high: {gene}={steady_state[gene]:.1f}")
                    return False, steady_state, None
            
            # Compute Jacobian eigenvalues
            jacobian = self._compute_jacobian(params, steady_state)
            eigenvalues = np.linalg.eigvals(jacobian)
            max_real = np.max(np.real(eigenvalues))
            
            if max_real >= -1e-6:
                if verbose:
                    print(f"  ✗ Unstable (max eigenvalue: {max_real:.6f})")
                return False, steady_state, eigenvalues
        
        # If all tests passed, return baseline steady state
        baseline_interventions = {self.pkn['exogenous']: 1.0}
        baseline_state = self._simulate_to_steady_state(params, baseline_interventions)
        jacobian = self._compute_jacobian(params, baseline_state)
        eigenvalues = np.linalg.eigvals(jacobian)
        
        if verbose:
            max_real = np.max(np.real(eigenvalues))
            print(f"  ✓ Stable under all {len(test_interventions)} conditions (max eig: {max_real:.6f})")
        
        return True, baseline_state, eigenvalues
    
    def find_stable_parameters(self, n_attempts=1000, seed=42, verbose=True):
        """
        Search for stable parameter set
        
        Args:
            n_attempts: Maximum number of attempts
            seed: Random seed for reproducibility
            verbose: Print progress
        
        Returns:
            dict: Stable parameters (or None if not found)
        """
        if verbose:
            print(f"Searching for stable parameters...")
            print(f"  Network: {len(self.pkn['variables'])} genes, {len(self.pkn['edges'])} edges")
            print(f"  Attempting up to {n_attempts} parameter sets")
        
        param_ranges = self.get_literature_ranges()
        
        for attempt in tqdm(range(n_attempts), disable=not verbose):
            # Sample parameters
            params = self.sample_parameters(param_ranges, seed=seed+attempt)
            
            # Check stability under baseline + test interventions
            is_stable, baseline, eigenvalues = self.check_stability(params, verbose=False)
            
            if is_stable:
                if verbose:
                    print(f"\n✓ Found stable parameters on attempt {attempt+1}")
                    print(f"  Baseline concentrations: min={min(baseline.values()):.3f}, "
                          f"max={max(baseline.values()):.3f}")
                    max_eig = np.max(np.real(eigenvalues))
                    print(f"  Max eigenvalue: {max_eig:.6f}")
                
                return {
                    'parameters': params,
                    'baseline_state': baseline,
                    'eigenvalues': eigenvalues.tolist(),
                    'attempt': attempt + 1,
                    'seed': seed
                }
        
        if verbose:
            print(f"\n✗ No stable parameters found in {n_attempts} attempts")
        
        return None
    
    def save_stable_parameters(self, stable_params, filename='stable_params.pkl'):
        """Save stable parameters to cache"""
        cache_file = self.cache_dir / filename
        
        with open(cache_file, 'wb') as f:
            pickle.dump(stable_params, f)
        
        print(f"✓ Saved stable parameters to: {cache_file}")
        
        # Also save as JSON (without numpy arrays)
        json_file = cache_file.with_suffix('.json')
        json_data = {
            'parameters': stable_params['parameters'],
            'baseline_state': stable_params['baseline_state'],
            'attempt': stable_params['attempt'],
            'seed': stable_params['seed']
        }
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✓ Saved parameters (JSON) to: {json_file}")
    
    def load_stable_parameters(self, filename='stable_params.pkl'):
        """Load stable parameters from cache"""
        cache_file = self.cache_dir / filename
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'rb') as f:
            stable_params = pickle.load(f)
        
        print(f"✓ Loaded stable parameters from: {cache_file}")
        return stable_params


def find_and_cache_stable_params(pkn, cache_dir='../ecoli_crp_data/cache', 
                                 n_attempts=1000, force_recompute=False):
    """
    Convenience function to find and cache stable parameters
    
    Args:
        pkn: PKN dictionary
        cache_dir: Cache directory
        n_attempts: Max attempts
        force_recompute: Force new search even if cached
    
    Returns:
        dict: Stable parameters
    """
    finder = StableParameterFinder(pkn, cache_dir=cache_dir)
    
    # Try to load cached
    if not force_recompute:
        cached = finder.load_stable_parameters()
        if cached is not None:
            print("Using cached stable parameters")
            return cached
    
    # Search for new stable parameters
    print("Searching for new stable parameters...")
    stable_params = finder.find_stable_parameters(n_attempts=n_attempts)
    
    if stable_params is None:
        raise RuntimeError(f"Could not find stable parameters in {n_attempts} attempts. "
                         f"Try increasing n_attempts or adjusting parameter ranges.")
    
    # Cache for future use
    finder.save_stable_parameters(stable_params)
    
    return stable_params