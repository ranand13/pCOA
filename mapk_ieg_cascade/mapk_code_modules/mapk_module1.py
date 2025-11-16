"""
Module 1: Ground Truth Data Generation

FEATURES:
1. k_in test range is INTERPOLATION (within training range)
2. Parameter perturbation for robustness testing (optional)
3. No noise on test data (clean ground truth)
4. Proper cache handling for perturbed parameters
"""

import numpy as np
import pickle
from pathlib import Path


class GroundTruthDataGenerator:
    def __init__(self, ode_module, cache_dir='cache'):
        self.ode_module = ode_module
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.config = ode_module.get_cascade_config()
    
    def find_stable_k_in_range(self, config, k_in_candidates=None, verbose=True):
        """
        Find k_in range stable across all 3 formulations and all interventions
        
        Args:
            config: System configuration (may have perturbed parameters)
            k_in_candidates: Candidate k_in values to test
        """
        if k_in_candidates is None:
            k_in_candidates = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        
        if verbose:
            print("[Module 1] Finding stable k_in range...")
        
        formulations = ['mass_action', 'mm', 'hill']
        stable_k_in = set(k_in_candidates)
        
        for form in formulations:
            solver = self.ode_module.build_ode_solver(form, config)
            
            for k_in in k_in_candidates:
                if k_in not in stable_k_in:
                    continue
                
                # Test wild-type
                ss = solver(k_in)
                if not ss or not self._is_stable(ss):
                    stable_k_in.discard(k_in)
                    continue
                
                # Test all inhibitions
                for var in config['inhibitions'].keys():
                    I_name = config['inhibitions'][var]
                    ss_inh = solver(k_in, **{I_name: 0.2})
                    if not ss_inh or not self._is_stable(ss_inh):
                        stable_k_in.discard(k_in)
                        break
        
        stable_sorted = sorted(stable_k_in)
        if len(stable_sorted) < 2:
            raise ValueError(f"Insufficient stable k_in values: {stable_sorted}")
        
        k_in_range = (stable_sorted[0], stable_sorted[-1])
        
        if verbose:
            print(f"  Tested: {k_in_candidates}")
            print(f"  Stable: {stable_sorted}")
            print(f"  Range: {k_in_range}")
        
        return k_in_range
    
    def _is_stable(self, steady_state, min_val=0.001, max_val=50):
        """Check if steady state is in valid range"""
        if steady_state is None:
            return False
        values = list(steady_state.values())
        return all(min_val < v < max_val for v in values)
    
    def run(self, formulation, N_obs, k_in_range=None, 
            intervention_strengths=None, N_test_reps=5,
            noise_cv=0.05,
            perturb_gt_params=False,
            perturbation_cv=0.10,
            seed=42, force_refresh=False, verbose=True):
        """
        Complete Module 1 pipeline for one ground truth formulation
        
        PARAMETER PERTURBATION:
            perturb_gt_params: If True, sample GT parameters from log-normal distributions
            perturbation_cv: Coefficient of variation for sampling (0.10 = 10% typical deviation)
        
        With perturbation enabled:
        - Each run (different seed) generates different GT system
        - Tests robustness across parameter space
        - Run 10-25 times for error bars
        
        Args:
            formulation: Ground truth formulation
            N_obs: Number of observational samples
            k_in_range: k_in range (auto-computed if None)
            intervention_strengths: I values for test
            N_test_reps: Replicates per intervention
            noise_cv: Measurement noise (obs data only)
            perturb_gt_params: Sample GT parameters
            perturbation_cv: Parameter sampling CV
            seed: Random seed
        """
        # Cache file includes seed when perturbing (different params per seed)
        if perturb_gt_params:
            cache_file = self.cache_dir / f'gt_data_{formulation}_perturbed_seed{seed}.pkl'
        else:
            cache_file = self.cache_dir / f'gt_data_{formulation}.pkl'
        
        if not force_refresh and cache_file.exists():
            if verbose:
                perturb_str = f"(perturbed, seed={seed})" if perturb_gt_params else ""
                print(f"[Module 1] Loading cached GT data for {formulation} {perturb_str}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        if verbose:
            print(f"\n[Module 1] Generating GT data for {formulation.upper()}")
        
        np.random.seed(seed)
        
        # ==== PARAMETER PERTURBATION ====
        config = self.ode_module.get_cascade_config()
        base_params = config['parameters'][formulation]
        
        if perturb_gt_params:
            if verbose:
                print(f"  Perturbing GT parameters (CV={perturbation_cv})...")
            
            # Sample multiplicative factors from log-normal
            # median=1.0, CV=perturbation_cv
            perturbed_params = {}
            sigma = np.sqrt(np.log(1 + perturbation_cv**2))
            
            for name, base_value in base_params.items():
                factor = np.random.lognormal(mean=0, sigma=sigma)
                perturbed_params[name] = base_value * factor
            
            # Show largest deviations
            if verbose:
                deviations = [(name, perturbed_params[name]/base_params[name]) 
                             for name in base_params.keys()]
                deviations.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)
                print(f"    Largest deviations:")
                for name, factor in deviations[:5]:
                    print(f"      {name}: {factor:.3f}×")
            
            # Update config
            config['parameters'][formulation] = perturbed_params
            gt_params_used = perturbed_params
        else:
            gt_params_used = base_params
        
        # Find stable k_in range with current config
        if k_in_range is None:
            k_in_range = self.find_stable_k_in_range(config, verbose=verbose)
        
        # Test k_in range (interpolation)
        train_width = k_in_range[1] - k_in_range[0]
        margin = 0.1 * train_width
        k_in_test_range = (k_in_range[0] + margin, k_in_range[1] - margin)
        
        if verbose:
            print(f"\n  k_in training range: {k_in_range}")
            print(f"  k_in test range: {k_in_test_range} (interpolation only)")
        
        # ==== OBSERVATIONAL DATA ====
        if verbose:
            print(f"\n  Generating observational data (N={N_obs})...")
        
        solver = self.ode_module.build_ode_solver(formulation, config)
        variables = config['variables']
        
        obs_data = []
        k_in_samples = np.random.uniform(k_in_range[0], k_in_range[1], N_obs)
        
        for k_in in k_in_samples:
            ss = solver(k_in)
            if ss and self._is_stable(ss):
                row = [k_in] + [ss[v] for v in variables]
                obs_data.append(row)
        
        obs_data = np.array(obs_data)
        
        # Add measurement noise
        if noise_cv > 0:
            noise = np.random.normal(1, noise_cv, obs_data[:, 1:].shape)
            obs_data[:, 1:] *= noise
            obs_data = np.maximum(obs_data, 1e-6)
        
        if verbose:
            print(f"    Generated {len(obs_data)} samples")
        
        # ==== TEST INTERVENTIONAL DATA ====
        if verbose:
            print(f"\n  Generating test interventional data...")
        
        if intervention_strengths is None:
            intervention_strengths = [0.2]
        
        inhibitions = config['inhibitions']
        test_data = []
        
        for target_var in inhibitions.keys():
            I_name = inhibitions[target_var]
            
            for I_val in intervention_strengths:
                for rep in range(N_test_reps):
                    k_in = np.random.uniform(k_in_test_range[0], k_in_test_range[1])
                    
                    # Baseline (wild-type)
                    ss_baseline = solver(k_in)
                    if not ss_baseline or not self._is_stable(ss_baseline):
                        continue
                    
                    initial_state = {v: ss_baseline[v] for v in variables}
                    
                    # Perturbed
                    ss_perturbed = solver(k_in, **{I_name: I_val})
                    if not ss_perturbed or not self._is_stable(ss_perturbed):
                        continue
                    
                    final_state = {v: ss_perturbed[v] for v in variables}
                    
                    I_vec = {inh: 1.0 for inh in inhibitions.values()}
                    I_vec[I_name] = I_val
                    
                    test_data.append({
                        'k_in': k_in,
                        'interventions': I_vec,
                        'target': target_var,
                        'I_value': I_val,
                        'initial_state': initial_state,
                        'final_state': final_state,
                    })
        
        if verbose:
            print(f"    Generated {len(test_data)} interventions")
        
        # Package results
        result = {
            'formulation': formulation,
            'obs_data': obs_data,
            'test_data': test_data,
            'k_in_train_range': k_in_range,
            'k_in_test_range': k_in_test_range,
            'gt_params_used': gt_params_used,
            'perturbed': perturb_gt_params,
            'metadata': {
                'N_obs': N_obs,
                'N_test': len(test_data),
                'noise_cv': noise_cv,
                'perturbation_cv': perturbation_cv if perturb_gt_params else 0.0,
                'intervention_strengths': intervention_strengths,
                'seed': seed,
            }
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        if verbose:
            print(f"\n  ✓ Cached to {cache_file.name}")
        
        return result


if __name__ == "__main__":
    pass