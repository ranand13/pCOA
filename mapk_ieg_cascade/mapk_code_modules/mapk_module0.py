"""
Module 0: PCOA Structure Discovery and Caching

"""

import pickle
import hashlib
from pathlib import Path
from pcoa_structure_discovery import PCOADiscovery


class PCOAStructureCache:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, system_name, variables, exogenous):
        """Generate cache key from system specification"""
        key_str = f"{system_name}_{sorted(variables)}_{exogenous}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def discover_and_cache(self, ode_module, system_name, force_refresh=False, verbose=True):
        """
        Discover PCOA structures for all formulations and cache
        
        Returns:
            structures: {formulation: (structure, topo, var_to_eq, equations)}
        """
        config = ode_module.get_cascade_config()
        
        # Generate cache key
        cache_key = self._get_cache_key(
            system_name, 
            config['variables'], 
            config['exogenous']
        )
        cache_file = self.cache_dir / f'pcoa_structures_{cache_key}.pkl'
        
        # Check cache
        if not force_refresh and cache_file.exists():
            if verbose:
                print(f"[Module 0] Loading cached PCOA structures from {cache_file.name}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        if verbose:
            print(f"[Module 0] Discovering PCOA structures...")
            print(f"  System: {system_name}")
            print(f"  Variables: {config['variables']}")
        
        # Initialize PCOA
        pkn = {
            'variables': config['variables'],
            'exogenous': config['exogenous'],
            'inhibitions': config['inhibitions']
        }
        pcoa = PCOADiscovery(pkn)
        
        # Discover for each formulation
        structures = {}
        formulations = ['mass_action', 'mm', 'hill']
        
        for form in formulations:
            if verbose:
                print(f"\n  Formulation: {form.upper()}")
            
            eqs = ode_module.get_equilibrium_equations(form, config)
            
            if verbose:
                print(f"    Equilibrium equations: {len(eqs)}")
            
            # NEW: Returns 4 items instead of 2
            structure, topo, var_to_eq, selected_eqs = pcoa.discover_structure(eqs, debug=False)
            
            structures[form] = (structure, topo, var_to_eq, selected_eqs)
            
            if verbose:
                n_edges = sum(len(parents) for parents in structure.values())
                print(f"    Discovered: {n_edges} edges")
                
                # Show which vars are deterministic
                deterministic_vars = [v for v, eq in var_to_eq.items() if eq.startswith('f_R_')]
                if deterministic_vars:
                    print(f"    Deterministic vars: {deterministic_vars}")
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(structures, f)
        
        if verbose:
            print(f"\n  ✓ Cached to {cache_file.name}")
        
        return structures
    
    def load(self, system_name, variables, exogenous):
        """Load cached structures"""
        cache_key = self._get_cache_key(system_name, variables, exogenous)
        cache_file = self.cache_dir / f'pcoa_structures_{cache_key}.pkl'
        
        if not cache_file.exists():
            raise FileNotFoundError(f"No cached structures for {system_name}")
        
        with open(cache_file, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    import mapk_ieg_cascade_clean as ode_system
    
    cache = PCOAStructureCache(cache_dir='investigation_b_cache')
    structures = cache.discover_and_cache(
        ode_system, 
        system_name='mapk_ieg_clean',
        force_refresh=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("PCOA STRUCTURES WITH MATCHING INFO")
    print("="*80)
    
    for form, (struct, topo, var_to_eq, selected_eqs) in structures.items():
        print(f"\n{form.upper()}:")
        print(f"  Structure:")
        for var in struct.keys():
            parents = struct[var]
            eq_name = var_to_eq.get(var, 'UNKNOWN')
            is_R = eq_name.startswith('f_R_')
            marker = ' [DETERMINISTIC via R]' if is_R else ''
            print(f"    {var} ← [{', '.join(parents) if parents else '∅'}] via {eq_name}{marker}")