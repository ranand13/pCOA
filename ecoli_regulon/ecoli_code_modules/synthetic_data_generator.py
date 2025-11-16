"""
Synthetic Interventional Data Generator

Generates synthetic steady-state data from ODE model with stable parameters.
Uses forward ODE simulation (ODESimulator) to reach steady state.
Used to train SCM via PCOA structure discovery.

Similar to Investigation B and C approach.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from itertools import product
from tqdm import tqdm
import sys, os

sys.path.append(os.path.abspath('../../code_modules/'))
from ode_simulator import ODESimulator


class SyntheticDataGenerator:
    
    def __init__(self, pkn, stable_params, form='hill', cache_dir='../ecoli_crp_data/cache'):
        """
        Initialize synthetic data generator
        
        Args:
            pkn: PKN dictionary
            stable_params: Stable parameter set from parameter finder
            form: Kinetic form
            cache_dir: Where to save generated data
        """
        self.pkn = pkn
        self.params = stable_params['parameters']
        self.baseline_state = stable_params['baseline_state']
        self.form = form
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ode_sim = ODESimulator(pkn)
    
    def generate_observational_data(self, n_samples=100, noise_std=0.05, seed=42):
        """
        Generate observational (wild-type) data with measurement noise
        
        Args:
            n_samples: Number of samples to generate
            noise_std: Standard deviation of log-normal noise
            seed: Random seed
        
        Returns:
            DataFrame: Observational data (n_samples × n_genes)
        """
        np.random.seed(seed)
        
        genes = self.pkn['variables']
        baseline_values = np.array([self.baseline_state[g] for g in genes])
        
        # Add log-normal noise to simulate biological variability
        log_baseline = np.log(baseline_values)
        
        samples = []
        for i in range(n_samples):
            noisy_log = log_baseline + np.random.normal(0, noise_std, len(genes))
            noisy_values = np.exp(noisy_log)
            samples.append(noisy_values)
        
        df = pd.DataFrame(samples, columns=genes)
        
        return df
    
    def generate_single_intervention(self, target_gene, intervention_strength):
        """
        Generate steady state for a single intervention using ODESimulator
        
        Args:
            target_gene: Gene to intervene on
            intervention_strength: I value (0 = knockout, 0.5 = 50% knockdown, 1 = no effect)
        
        Returns:
            dict: Steady-state concentrations under intervention
        """
        try:
            # Build solver
            solver = self.ode_sim.build_solver(self.form, self.params)
            
            # Set up interventions
            k_val = 1.0  # Baseline exogenous input
            
            # Build I_kwargs
            I_kwargs = {self.pkn['inhibitions'][g]: 1.0 for g in self.pkn['inhibitions']}
            
            # Set target intervention
            if target_gene in self.pkn['inhibitions']:
                I_kwargs[self.pkn['inhibitions'][target_gene]] = intervention_strength
            
            # Call solver to get steady state
            steady_state = solver(k_val, **I_kwargs)
            
            return steady_state
            
        except:
            return None
    
    def generate_interventional_data(self, intervention_strengths=[0.2, 0.5, 0.8],
                                    target_genes=None, verbose=True):
        """
        Generate interventional data for training SCM
        
        Args:
            intervention_strengths: List of I values to test (e.g., [0.2, 0.5, 0.8])
            target_genes: Which genes to intervene on (default: all genes with inhibitions)
            verbose: Print progress
        
        Returns:
            DataFrame: Interventional data with columns for genes + intervention info
        """
        if target_genes is None:
            target_genes = list(self.pkn['inhibitions'].keys())
        
        if verbose:
            print(f"Generating interventional data...")
            print(f"  Target genes: {len(target_genes)}")
            print(f"  Intervention strengths: {intervention_strengths}")
            print(f"  Total experiments: {len(target_genes) * len(intervention_strengths)}")
        
        data_rows = []
        genes = self.pkn['variables']
        
        # Generate data for each combination
        for target in tqdm(target_genes, disable=not verbose):
            for strength in intervention_strengths:
                steady_state = self.generate_single_intervention(target, strength)
                
                if steady_state is not None:
                    row = {gene: steady_state.get(gene, np.nan) for gene in genes}
                    row['target_gene'] = target
                    row['intervention_strength'] = strength
                    row[self.pkn['exogenous']] = 1.0
                    data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        if verbose:
            print(f"  Generated {len(df)} interventional samples")
            print(f"  Columns: {list(df.columns[:5])}... + intervention info")
        
        return df
    
    def generate_training_dataset(self, n_obs_samples=100,
                                 intervention_strengths=[0.2, 0.5, 0.8],
                                 target_genes=None,
                                 obs_noise_std=0.05,
                                 seed=42,
                                 verbose=True):
        """
        Generate complete training dataset (observational + interventional)
        
        Args:
            n_obs_samples: Number of observational samples
            intervention_strengths: Intervention I values
            target_genes: Which genes to intervene on
            obs_noise_std: Noise level for observational data
            seed: Random seed
            verbose: Print info
        
        Returns:
            tuple: (obs_data, interventional_data)
        """
        if verbose:
            print("="*60)
            print("GENERATING SYNTHETIC TRAINING DATASET")
            print("="*60)
        
        # Generate observational data
        if verbose:
            print(f"\n1. Generating observational data...")
        obs_data = self.generate_observational_data(
            n_samples=n_obs_samples,
            noise_std=obs_noise_std,
            seed=seed
        )
        if verbose:
            print(f"   ✓ {len(obs_data)} observational samples")
        
        # Generate interventional data
        if verbose:
            print(f"\n2. Generating interventional data...")
        int_data = self.generate_interventional_data(
            intervention_strengths=intervention_strengths,
            target_genes=target_genes,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n" + "="*60)
            print(f"DATASET SUMMARY")
            print(f"="*60)
            print(f"Observational: {len(obs_data)} samples")
            print(f"Interventional: {len(int_data)} samples")
            print(f"Total: {len(obs_data) + len(int_data)} samples")
        
        return obs_data, int_data
    
    def save_dataset(self, obs_data, int_data, prefix='crp_regulon'):
        """
        Save generated dataset to cache
        
        Args:
            obs_data: Observational DataFrame
            int_data: Interventional DataFrame
            prefix: Filename prefix
        """
        obs_file = self.cache_dir / f'{prefix}_observational.csv'
        int_file = self.cache_dir / f'{prefix}_interventional.csv'
        
        obs_data.to_csv(obs_file, index=False)
        int_data.to_csv(int_file, index=False)
        
        print(f"\n✓ Saved datasets:")
        print(f"  Observational: {obs_file}")
        print(f"  Interventional: {int_file}")
        
        # Also save metadata
        metadata = {
            'n_genes': len(self.pkn['variables']),
            'n_edges': len(self.pkn['edges']),
            'n_obs_samples': len(obs_data),
            'n_int_samples': len(int_data),
            'genes': self.pkn['variables'],
            'intervention_type': self.pkn['inhibition_type']
        }
        
        metadata_file = self.cache_dir / f'{prefix}_metadata.json'
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Metadata: {metadata_file}")
    
    def load_dataset(self, prefix='crp_regulon'):
        """
        Load cached dataset
        
        Args:
            prefix: Filename prefix
        
        Returns:
            tuple: (obs_data, int_data) or (None, None) if not found
        """
        obs_file = self.cache_dir / f'{prefix}_observational.csv'
        int_file = self.cache_dir / f'{prefix}_interventional.csv'
        
        if not obs_file.exists() or not int_file.exists():
            return None, None
        
        obs_data = pd.read_csv(obs_file)
        int_data = pd.read_csv(int_file)
        
        print(f"✓ Loaded cached datasets:")
        print(f"  Observational: {len(obs_data)} samples")
        print(f"  Interventional: {len(int_data)} samples")
        
        return obs_data, int_data


def generate_and_cache_data(pkn, stable_params, 
                           n_obs_samples=100,
                           intervention_strengths=[0.2, 0.5, 0.8],
                           cache_dir='../ecoli_crp_data/cache',
                           prefix='crp_regulon',
                           force_recompute=False):
    """
    Convenience function to generate and cache synthetic data
    
    Args:
        pkn: PKN dictionary
        stable_params: Stable parameters from parameter finder
        n_obs_samples: Number of observational samples
        intervention_strengths: I values for interventions
        cache_dir: Cache directory
        prefix: Filename prefix
        force_recompute: Force regeneration even if cached
    
    Returns:
        tuple: (obs_data, int_data)
    """
    generator = SyntheticDataGenerator(pkn, stable_params, cache_dir=cache_dir)
    
    # Try to load cached
    if not force_recompute:
        obs_data, int_data = generator.load_dataset(prefix=prefix)
        if obs_data is not None:
            return obs_data, int_data
    
    # Generate new data
    obs_data, int_data = generator.generate_training_dataset(
        n_obs_samples=n_obs_samples,
        intervention_strengths=intervention_strengths,
        verbose=True
    )
    
    # Cache for future use
    generator.save_dataset(obs_data, int_data, prefix=prefix)
    
    return obs_data, int_data