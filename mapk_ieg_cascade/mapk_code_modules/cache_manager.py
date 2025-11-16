"""
MODULE 9: Caching System - WITH INTERVENTION-AWARE CACHING

Provides intelligent caching for:
1. PCOA structure discovery (independent of intervention strengths)
2. Ground truth obs data + fitted ODE models (independent of intervention strengths)
3. Interventional data generation (DEPENDS on intervention strengths - cached separately)

Critical: Interventional data cache keys include I_grid to ensure different
intervention designs don't share cache entries.

Usage:
  from cache_manager import CacheManager
  cache = CacheManager()
  
  structure = cache.get_pcoa_structure(pkn, form, eq_gen, pcoa)
  data = cache.get_training_data(pkn, base_params, gt_form, seed, N_obs, ode)
  interv = cache.get_interventional_data(..., I_grid=[0.15, 0.3, ...])
"""

import os
import pickle
import hashlib
import json
from pathlib import Path


class CacheManager:
    def __init__(self, cache_dir='.cache_investigation_b'):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.pcoa_dir = self.cache_dir / 'pcoa_structures'
        self.data_dir = self.cache_dir / 'training_data'
        self.interv_dir = self.cache_dir / 'interventional_data'
        
        self.pcoa_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.interv_dir.mkdir(exist_ok=True)
    
    def _hash_dict(self, d):
        """Create hash of dictionary for cache key"""
        json_str = json.dumps(d, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()[:16]
    
    def _hash_list(self, lst):
        """Create hash of list for cache key"""
        json_str = json.dumps(sorted([float(x) for x in lst]), default=str)
        return hashlib.md5(json_str.encode()).hexdigest()[:8]
    
    def _get_pcoa_cache_key(self, pkn, formulation):
        """Generate cache key for PCOA structure (intervention-independent)"""
        pkn_hash = self._hash_dict({
            'variables': pkn['variables'],
            'edges': pkn['edges'],
            'exogenous': pkn['exogenous'],
            'inhibitions': pkn['inhibitions'],
            'no_degradation': pkn.get('no_degradation', []),
            'saturation': pkn.get('saturation', {})
        })
        return f"pcoa_{pkn_hash}_{formulation}.pkl"
    
    def _get_data_cache_key(self, pkn, gt_form, seed, N_obs):
        """Generate cache key for training data (intervention-independent)"""
        pkn_hash = self._hash_dict({
            'variables': pkn['variables'],
            'edges': pkn['edges'],
            'exogenous': pkn['exogenous'],
            'inhibitions': pkn['inhibitions'],
            'no_degradation': pkn.get('no_degradation', [])
        })
        return f"data_{pkn_hash}_{gt_form}_seed{seed}_N{N_obs}.pkl"
    
    def _get_interv_cache_key(self, pkn, gt_form, model_form, N_per_combo, I_grid, seed):
        """
        Generate cache key for interventional data
        
        CRITICAL: Includes I_grid hash to ensure different intervention strengths
        generate separate cache entries
        """
        pkn_hash = self._hash_dict({
            'variables': pkn['variables'],
            'edges': pkn['edges'],
            'exogenous': pkn['exogenous'],
            'inhibitions': pkn['inhibitions'],
        })
        # Hash the I_grid to make it part of the key
        I_hash = self._hash_list(I_grid)
        return f"interv_{pkn_hash}_{gt_form}_{model_form}_seed{seed}_Nper{N_per_combo}_I{I_hash}.pkl"
    
    def get_pcoa_structure(self, pkn, formulation, eq_gen, pcoa, 
                          use_cache=True, verbose=True):
        """
        Get PCOA structure from cache or compute
        
        Args:
            pkn: Prior knowledge network
            formulation: 'mass_action', 'mm', or 'hill'
            eq_gen: EquilibriumGenerator instance
            pcoa: PCOADiscovery instance
            use_cache: Whether to use cached result
            verbose: Print cache status
        
        Returns:
            (structure, topo_order) tuple
        """
        cache_key = self._get_pcoa_cache_key(pkn, formulation)
        cache_path = self.pcoa_dir / cache_key
        
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                if verbose:
                    print(f"  ✓ Loaded PCOA structure from cache ({formulation})")
                return result
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Cache load failed: {e}, recomputing...")
        
        # Compute structure
        if verbose:
            print(f"  Computing PCOA structure ({formulation})...")
        
        eqs = eq_gen.generate_equations(formulation, with_inhibitions=False)
        structure, topo = pcoa.discover_structure(eqs, debug=False)
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((structure, topo), f)
            if verbose:
                print(f"  ✓ Saved PCOA structure to cache")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Cache save failed: {e}")
        
        return structure, topo
    
    def get_training_data(self, pkn, base_params, gt_form, seed, N_obs, ode,
                         use_cache=True, verbose=True):
        """
        Get training data bundle from cache or generate
        
        Args:
            pkn: Prior knowledge network
            base_params: Base parameters dict
            gt_form: Ground truth formulation
            seed: Random seed
            N_obs: Number of observational samples
            ode: ODESimulator instance
            use_cache: Whether to use cached result
            verbose: Print cache status
        
        Returns:
            Dictionary with:
                'obs_data': Observational data
                'fitted_mass_action': Fitted mass_action parameters
                'fitted_mm': Fitted mm parameters
                'fitted_hill': Fitted hill parameters
        """
        cache_key = self._get_data_cache_key(pkn, gt_form, seed, N_obs)
        cache_path = self.data_dir / cache_key
        
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                if verbose:
                    print(f"  ✓ Loaded training data from cache (GT={gt_form}, N={N_obs}, seed={seed})")
                return result
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Cache load failed: {e}, regenerating...")
        
        # Generate data
        if verbose:
            print(f"  Generating obs data (GT={gt_form}, N={N_obs})...")
        
        obs_data = ode.generate_obs_data(gt_form, base_params[gt_form], 
                                         N=N_obs, noise=0.05, seed=seed)
        
        # Fit all three formulations
        if verbose:
            print(f"  Fitting 3 ODE models...")
        
        fitted_models = {}
        for form in ['mass_action', 'mm', 'hill']:
            fitted_models[f'fitted_{form}'] = ode.fit_to_obs(
                form, obs_data, seed=seed, verbose=False
            )
        
        result = {
            'obs_data': obs_data,
            **fitted_models
        }
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            if verbose:
                print(f"  ✓ Saved training data to cache")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Cache save failed: {e}")
        
        return result
    
    def get_interventional_data(self, pkn, gt_form, model_form, fitted_params, 
                               N_per_combo, I_grid, seed, ode,
                               use_cache=True, verbose=True):
        """
        Get interventional data from cache or generate
        
        CRITICAL: Cache key includes I_grid to ensure different intervention
        strengths generate separate cache entries
        
        Args:
            pkn: Prior knowledge network
            gt_form: Ground truth formulation (for cache key)
            model_form: Model formulation fitted
            fitted_params: Fitted parameters dict
            N_per_combo: Samples per (k_in, target, I) combination
            I_grid: List of intervention strengths used [0.15, 0.3, 0.5, ...]
            seed: Random seed
            ode: ODESimulator instance
            use_cache: Whether to use cached result
            verbose: Print cache status
        
        Returns:
            List of interventional samples
        """
        cache_key = self._get_interv_cache_key(
            pkn, gt_form, model_form, N_per_combo, I_grid, seed
        )
        cache_path = self.interv_dir / cache_key
        
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                if verbose:
                    print(f"  ✓ Loaded interventional data from cache ({model_form}, I_grid hash={self._hash_list(I_grid)})")
                return result
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Cache load failed: {e}, regenerating...")
        
        # Generate interventional data (uses standard I_grid from ode module)
        if verbose:
            print(f"  Generating interventional data ({model_form})...")
        
        interv_data = ode.generate_interv_data(
            model_form, fitted_params, N_per_combo=N_per_combo, seed=seed
        )
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(interv_data, f)
            if verbose:
                print(f"  ✓ Saved interventional data to cache")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Cache save failed: {e}")
        
        return interv_data
    
    def clear_cache(self, cache_type='all'):
        """
        Clear cached data
        
        Args:
            cache_type: 'all', 'pcoa', 'data', or 'interv'
        """
        if cache_type in ['all', 'pcoa']:
            count = 0
            for f in self.pcoa_dir.glob('*.pkl'):
                f.unlink()
                count += 1
            if count > 0:
                print(f"✓ Cleared PCOA cache ({count} files)")
        
        if cache_type in ['all', 'data']:
            count = 0
            for f in self.data_dir.glob('*.pkl'):
                f.unlink()
                count += 1
            if count > 0:
                print(f"✓ Cleared training data cache ({count} files)")
        
        if cache_type in ['all', 'interv']:
            count = 0
            for f in self.interv_dir.glob('*.pkl'):
                f.unlink()
                count += 1
            if count > 0:
                print(f"✓ Cleared interventional data cache ({count} files)")
    
    def cache_info(self):
        """Print cache statistics"""
        pcoa_files = list(self.pcoa_dir.glob('*.pkl'))
        data_files = list(self.data_dir.glob('*.pkl'))
        interv_files = list(self.interv_dir.glob('*.pkl'))
        
        pcoa_size = sum(f.stat().st_size for f in pcoa_files) / 1024 / 1024
        data_size = sum(f.stat().st_size for f in data_files) / 1024 / 1024
        interv_size = sum(f.stat().st_size for f in interv_files) / 1024 / 1024
        
        print("="*60)
        print("CACHE STATISTICS")
        print("="*60)
        print(f"PCOA structures:     {len(pcoa_files)} files, {pcoa_size:.2f} MB")
        print(f"Training data:       {len(data_files)} files, {data_size:.2f} MB")
        print(f"Interventional data: {len(interv_files)} files, {interv_size:.2f} MB")
        print(f"Total:               {pcoa_size + data_size + interv_size:.2f} MB")
        print(f"Location:            {self.cache_dir.absolute()}")
        print("="*60)

