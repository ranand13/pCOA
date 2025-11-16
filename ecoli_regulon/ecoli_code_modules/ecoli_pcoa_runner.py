"""
PCOA Structure Discovery for E. coli CRP Regulon

Runs PCOA on ODE equilibrium equations to discover causal structure.
The discovered structure is then used to train SCM on synthetic data.

Workflow:
1. Generate equilibrium equations from PKN
2. Run PCOA (Algorithm 1: parameter cancellation, Algorithm 2: variable transformation)
3. Discover causal structure {var: [parents]}
4. Cache structure for SCM training
"""
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import sys
import os

sys.path.append('.')
sys.path.append('../ecoli_code_modules/')
sys.path.append('../../code_modules/')

from equilibrium_generator import EquilibriumGenerator
from pcoa_structure_discovery import PCOADiscovery


class EcoliPCOARunner:
    """
    Run PCOA and manage discovered structures for E. coli
    
    Caches structures to avoid recomputation
    """
    
    def __init__(self, cache_dir='ecoli_crp_data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def discover_and_save(self, pkn, formulation='hill', name='crp_regulon', 
                         verbose=True, debug=False):
        """
        Discover PCOA structure and save to file
        
        Args:
            pkn: PKN dict in framework format
            formulation: 'mass_action', 'mm', or 'hill'
            name: Network identifier (e.g., 'crp_regulon')
            verbose: Print progress
            debug: Show detailed PCOA algorithm output
        
        Returns:
            structure: {var: [parents]}
            topo_order: Topological ordering
        """
        
        cache_file = self.cache_dir / f'{name}_{formulation}_structure.pkl'
        
        if cache_file.exists():
            if verbose:
                print(f"✓ Loading cached structure: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                structure, topo = pickle.load(f)
            
            if verbose:
                self._print_structure_summary(structure, pkn)
            
            return structure, topo
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"PCOA STRUCTURE DISCOVERY: {name} ({formulation})")
            print(f"{'='*80}")
            print(f"  Variables: {len(pkn['variables'])}")
            print(f"  Edges in PKN: {len(pkn['edges'])}")
        
        # Initialize PCOA
        eq_gen = EquilibriumGenerator(pkn)
        pcoa_disc = PCOADiscovery(pkn)
        
        # Generate equilibrium equations (without inhibitions)
        if verbose:
            print(f"\n  Step 1: Generating equilibrium equations...")
        eqs = eq_gen.generate_equations(formulation, with_inhibitions=False)
        
        if verbose:
            print(f"    ✓ Generated {len(eqs)} equilibrium equations")
        
        # Run PCOA discovery
        if verbose:
            print(f"\n  Step 2: Running PCOA...")
            if debug:
                print(f"    (Debug mode - showing Algorithm 1 & 2 details)")
        
        structure, topo = pcoa_disc.discover_structure(eqs, debug=debug)
        
        # Save structure
        with open(cache_file, 'wb') as f:
            pickle.dump((structure, topo), f)
        
        if verbose:
            print(f"\n  ✓ Saved structure to: {cache_file.name}")
            self._print_structure_summary(structure, pkn)
        
        return structure, topo
    
    def _print_structure_summary(self, structure, pkn):
        """Print summary of discovered structure"""
        print(f"\n  Discovered Causal Structure:")
        print(f"  {'-'*76}")
        
        # Show first 10 variables
        for var in pkn['variables'][:10]:
            parents = structure.get(var, [])
            if parents:
                parent_str = ', '.join(parents)
                print(f"    {var:15s} ← [{parent_str}]")
            else:
                print(f"    {var:15s} ← [exogenous]")
        
        if len(pkn['variables']) > 10:
            print(f"    ... ({len(pkn['variables'])-10} more variables)")
        
        # Statistics
        n_exog = sum(1 for v in pkn['variables'] if not structure.get(v, []))
        n_with_parents = len(pkn['variables']) - n_exog
        avg_parents = np.mean([len(structure.get(v, [])) for v in pkn['variables']])
        
        print(f"\n  Structure Statistics:")
        print(f"    Exogenous variables: {n_exog}")
        print(f"    Variables with parents: {n_with_parents}")
        print(f"    Average parents per variable: {avg_parents:.2f}")
    
    def load_structure(self, name, formulation='hill'):
        """
        Load previously discovered structure
        
        Args:
            name: Network identifier
            formulation: 'mass_action', 'mm', or 'hill'
        
        Returns:
            structure, topo_order or (None, None) if not found
        """
        cache_file = self.cache_dir / f'{name}_{formulation}_structure.pkl'
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                structure, topo = pickle.load(f)
            print(f"✓ Loaded structure from: {cache_file.name}")
            return structure, topo
        else:
            print(f"⚠️  Structure not found: {cache_file}")
            return None, None
    
    def discover_all_formulations(self, pkn, name='crp_regulon', verbose=True):
        """
        Discover structures for all three formulations
        
        Args:
            pkn: PKN dict
            name: Network identifier
            verbose: Print progress
        
        Returns:
            dict: {formulation: (structure, topo_order)}
        """
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"PCOA STRUCTURE DISCOVERY: {name} (ALL FORMULATIONS)")
            print(f"{'='*80}")
        
        structures = {}
        
        for form in ['mass_action', 'mm', 'hill']:
            if verbose:
                print(f"\n{'─'*80}")
                print(f"{form.upper()} FORMULATION")
                print(f"{'─'*80}")
            
            structure, topo = self.discover_and_save(pkn, form, name, verbose)
            structures[form] = (structure, topo)
        
        # Check if all formulations yield same structure
        if verbose:
            print(f"\n{'='*80}")
            print("FORMULATION COMPARISON")
            print(f"{'='*80}")
            
            ma_struct = structures['mass_action'][0]
            mm_struct = structures['mm'][0]
            hill_struct = structures['hill'][0]
            
            differs = []
            for var in pkn['variables']:
                ma_parents = set(ma_struct.get(var, []))
                mm_parents = set(mm_struct.get(var, []))
                hill_parents = set(hill_struct.get(var, []))
                
                if not (ma_parents == mm_parents == hill_parents):
                    differs.append({
                        'var': var,
                        'ma': ma_parents,
                        'mm': mm_parents,
                        'hill': hill_parents
                    })
            
            if differs:
                print(f"⚠️  Variables with different structures: {len(differs)}/{len(pkn['variables'])}")
                for diff in differs[:3]:
                    print(f"\n  {diff['var']}:")
                    print(f"    MA:   {diff['ma']}")
                    print(f"    MM:   {diff['mm']}")
                    print(f"    Hill: {diff['hill']}")
                if len(differs) > 3:
                    print(f"    ... and {len(differs)-3} more")
            else:
                print(f"✓ All formulations yield IDENTICAL structure")
                print(f"  This suggests structure is robust to kinetic details")
        
        return structures


def run_pcoa_for_crp_regulon(pkn, cache_dir='ecoli_crp_data/cache', 
                             formulations=['hill'], verbose=True, debug=False):
    """
    Convenience function to run PCOA on CRP regulon
    
    Args:
        pkn: PKN dictionary
        cache_dir: Where to save structures
        formulations: Which formulations to test (default: just 'hill')
        verbose: Print progress
        debug: Show detailed PCOA algorithm output
    
    Returns:
        dict: {formulation: (structure, topo_order)}
    """
    
    runner = EcoliPCOARunner(cache_dir=cache_dir)
    
    if len(formulations) == 1:
        # Single formulation
        structure, topo = runner.discover_and_save(
            pkn, 
            formulation=formulations[0],
            name='crp_regulon',
            verbose=verbose,
            debug=debug
        )
        return {formulations[0]: (structure, topo)}
    else:
        # Multiple formulations - compare them
        return runner.discover_all_formulations(
            pkn, 
            name='crp_regulon',
            verbose=verbose
        )


if __name__ == "__main__":
    # Example usage
    print("Testing PCOA on CRP regulon...")
    
    import sys
    sys.path.append('ecoli_code_modules/')
    from ecoli_crispri_dataloader import EcoliCRISPRiDataLoader
    
    # Load module
    loader = EcoliCRISPRiDataLoader('ecoli_crp_data')
    genes, network = loader.load_functional_module('crp_regulon')
    pkn = loader.build_pkn_dict(genes, network)
    
    # Run PCOA
    structures = run_pcoa_for_crp_regulon(
        pkn,
        cache_dir='ecoli_crp_data/cache',
        formulations=['hill'],  # Just Hill for now
        verbose=True,
        debug=True  # Set to True to see Algorithm 1 & 2 details
    )
    
    print("\n" + "="*80)
    print("PCOA COMPLETE - Structure ready for SCM training")
    print("="*80)