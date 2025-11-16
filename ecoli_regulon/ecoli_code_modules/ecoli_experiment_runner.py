"""
Ecoli Experiment Runner
Runs the full pipeline: data loading, parameter rescaling, SCM training, and testing
"""

import pickle
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ecoli_crispri_dataloader_full import EcoliCRISPRiDataLoader
from stable_parameter_finder import StableParameterFinder
from synthetic_data_generator import SyntheticDataGenerator
from ecoli_pcoa_runner import EcoliPCOARunner
from scm_model import SCMModel


class InvestigationDRunner:
    """Runs Investigation D experiments with parameter rescaling"""
    
    def __init__(self, data_dir: str = 'ecoli_crp_data', cache_dir: str = 'ecoli_crp_data/cache'):
        self.data_dir = data_dir
        self.cache_dir = Path(cache_dir)
        self.loader = EcoliCRISPRiDataLoader(data_dir)
        
    def rescale_parameters(
        self,
        stable_params: Dict,
        real_wt_data: pd.DataFrame,
        genes: List[str],
        verbose: bool = True
    ) -> Dict:
        """
        Rescale ODE parameters to match PPTP-seq data scale
        
        Args:
            stable_params: Original stable parameters
            real_wt_data: Real PPTP-seq wild-type data (log2 scale)
            genes: List of all genes
            verbose: Print progress
            
        Returns:
            Rescaled parameters dictionary
        """
        if verbose:
            print("\n" + "="*80)
            print("RESCALING ODE PARAMETERS")
            print("="*80)
        
        # Calculate baseline shift needed
        ode_baseline_log2 = {g: np.log2(stable_params['baseline_state'][g] + 1e-6) 
                             for g in stable_params['baseline_state']}
        
        measured_genes_in_ode = [g for g in real_wt_data.columns if g in ode_baseline_log2]
        ode_measured_mean = np.mean([ode_baseline_log2[g] for g in measured_genes_in_ode])
        real_measured_mean = real_wt_data[measured_genes_in_ode].mean().mean()
        
        baseline_shift_log2 = real_measured_mean - ode_measured_mean
        scale_factor_linear = 2**baseline_shift_log2
        
        if verbose:
            print(f"\nBaseline Shift:")
            print(f"  ODE baseline (measured genes):  {ode_measured_mean:.2f} log2")
            print(f"  Real PPTP baseline:             {real_measured_mean:.2f} log2")
            print(f"  Shift needed:                   {baseline_shift_log2:+.2f} log2 ({scale_factor_linear:.1f}× linear)")
        
        # Apply shift to baseline state
        for gene in stable_params['baseline_state']:
            stable_params['baseline_state'][gene] *= scale_factor_linear
        
        # Scale production parameters (k_basal and V)
        k_basal_scaled = 0
        v_scaled = 0
        
        for param_name in stable_params['parameters']:
            if param_name.startswith('k_basal_'):
                stable_params['parameters'][param_name] *= scale_factor_linear
                k_basal_scaled += 1
            elif param_name.startswith('V_') and '_basal_' not in param_name:
                stable_params['parameters'][param_name] *= scale_factor_linear
                v_scaled += 1
        
        if verbose:
            print(f"\nScaled production parameters:")
            print(f"  k_basal:        {k_basal_scaled} params by {scale_factor_linear:.1f}×")
            print(f"  V_{{tf}}_{{target}}: {v_scaled} params by {scale_factor_linear:.1f}×")
        
        # Scale V based on gene variability
        gene_std = real_wt_data.std()
        v_params = [k for k in stable_params['parameters'] 
                   if k.startswith('V_') and '_basal_' not in k]
        
        scaled_count = 0
        for param_name in v_params:
            parts = param_name.split('_')
            if len(parts) >= 3:
                target_gene = '_'.join(parts[2:])
                
                if target_gene in gene_std.index:
                    target_var = gene_std[target_gene]
                    var_normalized = np.clip((target_var - 0.2) / (0.8 - 0.2), 0, 1)
                    v_scale = 0.5 + var_normalized * 2.5
                    
                    stable_params['parameters'][param_name] *= v_scale
                    scaled_count += 1
        
        if verbose:
            print(f"\nScaled {scaled_count}/{len(v_params)} V parameters by gene variability")
            
            # Verify
            ode_shifted_mean = np.mean([np.log2(stable_params['baseline_state'][g] + 1e-6) 
                                       for g in measured_genes_in_ode])
            print(f"\nVerification:")
            print(f"  Final baseline: {ode_shifted_mean:.2f} log2 (target: {real_measured_mean:.2f})")
        
        # Add rescaling metadata
        stable_params['rescaling_info'] = {
            'baseline_shift_log2': float(baseline_shift_log2),
            'scale_factor_linear': float(scale_factor_linear),
            'v_params_scaled': scaled_count,
            'original_ode_mean': float(ode_measured_mean),
            'target_pptp_mean': float(real_measured_mean),
        }
        
        return stable_params
    
    def generate_synthetic_data(
        self,
        pkn: Dict,
        stable_params: Dict,
        genes: List[str],
        force_regenerate: bool = True,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic observational and interventional data
        
        Args:
            pkn: Prior knowledge network
            stable_params: Stable ODE parameters
            genes: List of genes
            force_regenerate: Delete cache and regenerate
            verbose: Print progress
            
        Returns:
            (observational_data, interventional_data) both in log2 scale
        """
        if verbose:
            print("\n" + "="*80)
            print("GENERATING SYNTHETIC DATA")
            print("="*80)
        
        # Delete cache if forcing regeneration
        if force_regenerate:
            cache_files = [
                self.cache_dir / 'crp_regulon_observational.csv',
                self.cache_dir / 'crp_regulon_interventional.csv',
                self.cache_dir / 'crp_regulon_metadata.json'
            ]
            for cache_file in cache_files:
                if cache_file.exists():
                    cache_file.unlink()
                    if verbose:
                        print(f"  Deleted cache: {cache_file.name}")
        
        data_gen = SyntheticDataGenerator(pkn, stable_params, cache_dir=str(self.cache_dir))
        
        obs_data, int_data = data_gen.generate_training_dataset(
            n_obs_samples=100,
            intervention_strengths=[0.2, 0.5, 0.8],
            target_genes=None,
            obs_noise_std=0.05,
            seed=42,
            verbose=verbose
        )
        
        # Log-transform
        gene_cols = [c for c in int_data.columns if c in genes]
        int_data_log = int_data.copy()
        for col in gene_cols:
            int_data_log[col] = np.log2(int_data[col] + 1e-6)
        
        # Save
        data_gen.save_dataset(obs_data, int_data, prefix='crp_regulon')
        
        if verbose:
            print(f"\n✓ Generated synthetic data:")
            print(f"  Observational:   {obs_data.shape}")
            print(f"  Interventional:  {int_data_log.shape}")
        
        return obs_data, int_data_log
    
    def prepare_training_data(
        self,
        real_wt_data: pd.DataFrame,
        synthetic_int_data: pd.DataFrame,
        stable_params: Dict,
        genes: List[str],
        verbose: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare combined observational and interventional training data
        
        Args:
            real_wt_data: Real PPTP-seq wild-type (log2)
            synthetic_int_data: Synthetic interventional data (log2)
            stable_params: Rescaled parameters
            genes: All genes
            verbose: Print info
            
        Returns:
            (obs_data_list, int_data_list)
        """
        if verbose:
            print("\n" + "="*80)
            print("PREPARING TRAINING DATA")
            print("="*80)
        
        # Get ODE baseline for missing genes
        ode_baseline_log = {g: np.log2(stable_params['baseline_state'][g] + 1e-6) 
                           for g in genes}
        
        # Fill missing genes in real data
        obs_data_dict = []
        for _, sample in real_wt_data.iterrows():
            filled_sample = dict(ode_baseline_log)
            filled_sample.update(sample.to_dict())
            filled_sample['basal'] = 1.0
            for gene in genes:
                filled_sample[f'I_{gene}'] = 1.0
            obs_data_dict.append(filled_sample)
        
        int_data_dict = synthetic_int_data.to_dict('records')
        
        genes_with_data = list(real_wt_data.columns)
        missing_genes = [g for g in genes if g not in genes_with_data]
        
        if verbose:
            print(f"  Observational: {len(obs_data_dict)} samples")
            print(f"    Real data:   {len(genes_with_data)} genes")
            print(f"    Filled:      {len(missing_genes)} genes (from ODE baseline)")
            print(f"  Interventional: {len(int_data_dict)} samples (synthetic)")
        
        return obs_data_dict, int_data_dict
    
    def train_scm(
        self,
        structure: Dict,
        topo: List,
        obs_data: List[Dict],
        int_data: List[Dict],
        genes: List[str],
        verbose: bool = True
    ) -> SCMModel:
        """
        Train SCM model
        
        Args:
            structure: PCOA structure
            topo: Topological ordering
            obs_data: Observational training data
            int_data: Interventional training data
            genes: All genes
            verbose: Print progress
            
        Returns:
            Trained SCM model
        """
        if verbose:
            print("\n" + "="*80)
            print("TRAINING SCM")
            print("="*80)
        
        scm = SCMModel(
            structure, 
            topo, 
            'basal', 
            {g: f'I_{g}' for g in genes}, 
            genes
        )
        
        scm.train(
            obs_data=obs_data,
            interv_data=int_data,
            weighting_scheme='stratified_trust',
            verbose=verbose
        )
        
        if verbose:
            print(f"✓ SCM trained")
        
        return scm
    
    def test_predictions(
        self,
        scm: SCMModel,
        param_finder: StableParameterFinder,
        stable_params: Dict,
        knockdown_data: Dict,
        genes: List[str],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Test SCM and ODE predictions on knockdown data
        
        Args:
            scm: Trained SCM model
            param_finder: Parameter finder (for ODE solver)
            stable_params: Rescaled parameters
            knockdown_data: Test knockdown data
            genes: All genes
            verbose: Print progress
            
        Returns:
            DataFrame with results
        """
        if verbose:
            print("\n" + "="*80)
            print("TESTING PREDICTIONS")
            print("="*80)
        
        results = []
        ode_solver = param_finder.ode_sim.build_solver('hill', stable_params['parameters'])
        
        for tf_name, test_case in knockdown_data.items():
            if verbose:
                print(f"\n  Testing {tf_name} knockdown...")
            
            baseline_log = test_case['baseline']
            knockdown_log = test_case['knockdown']
            genes_to_test = [g for g in baseline_log.keys() 
                           if g in genes and not np.isnan(knockdown_log.get(g, np.nan))]
            
            # Build baseline
            full_baseline_log = dict(baseline_log)
            full_baseline_log['basal'] = 1.0
            
            # Build intervention
            interventions = {f'I_{g}': 1.0 for g in genes}
            interventions[f'I_{tf_name}'] = test_case['knockdown_strength']
            
            # ODE prediction
            k_val = 1.0
            I_kwargs = {f'I_{g}': interventions.get(f'I_{g}', 1.0) for g in genes}
            ode_pred_linear = ode_solver(k_val, **I_kwargs)
            
            if ode_pred_linear:
                ode_pred_log = {g: np.log2(ode_pred_linear[g] + 1e-6) 
                               for g in ode_pred_linear}
            else:
                ode_pred_log = None
            
            # SCM prediction
            scm_pred_log = scm.predict(full_baseline_log, interventions)
            
            # Calculate errors
            if ode_pred_log and scm_pred_log:
                for gene in genes_to_test:
                    if gene not in ode_pred_log or gene not in scm_pred_log:
                        continue
                    
                    true_val = knockdown_log[gene]
                    base_val = baseline_log[gene]
                    
                    ode_error = abs(ode_pred_log[gene] - true_val)
                    scm_error = abs(scm_pred_log[gene] - true_val)
                    
                    true_fc = true_val - base_val
                    ode_fc = ode_pred_log[gene] - base_val
                    scm_fc = scm_pred_log[gene] - base_val
                    
                    ode_fc_error = abs(ode_fc - true_fc)
                    scm_fc_error = abs(scm_fc - true_fc)
                    
                    results.append({
                        'tf_knockdown': tf_name,
                        'gene': gene,
                        'baseline_log2': base_val,
                        'true_knockdown_log2': true_val,
                        'true_fc': true_fc,
                        'ode_pred_log2': ode_pred_log[gene],
                        'scm_pred_log2': scm_pred_log[gene],
                        'ode_fc': ode_fc,
                        'scm_fc': scm_fc,
                        'ode_abs_error': ode_error,
                        'scm_abs_error': scm_error,
                        'ode_fc_error': ode_fc_error,
                        'scm_fc_error': scm_fc_error,
                        'scm_wins_abs': scm_error < ode_error,
                        'scm_wins_fc': scm_fc_error < ode_fc_error
                    })
        
        results_df = pd.DataFrame(results)
        results_df['ode_fold_error'] = 2**results_df['ode_abs_error']
        results_df['scm_fold_error'] = 2**results_df['scm_abs_error']
        results_df['baseline_error'] = results_df['true_fc'].abs()
        results_df['baseline_fold_error'] = 2**results_df['baseline_error']
        
        if verbose:
            print(f"\n✓ Generated {len(results_df)} predictions across {len(knockdown_data)} TFs")
        
        return results_df
    
    def analyze_results(
        self,
        results_df: pd.DataFrame,
        exclude_tfs: Optional[List[str]] = None,
        save_to_csv: bool = True,
        results_dir: str = 'results',
        verbose: bool = True
    ) -> Dict:
        """
        Analyze and print results with mean statistics
        
        Args:
            results_df: Results DataFrame
            exclude_tfs: TFs to exclude from analysis (e.g., ['crp'])
            save_to_csv: Save results tables to CSV
            results_dir: Directory to save CSV files
            verbose: Print detailed results
            
        Returns:
            Dictionary with summary statistics
        """
        if exclude_tfs is None:
            exclude_tfs = ['crp']
        
        # Filter
        results_filtered = results_df[~results_df['tf_knockdown'].isin(exclude_tfs)].copy()
        
        if verbose:
            print("\n" + "="*80)
            print("INVESTIGATION D RESULTS TABLES")
            print("="*80)
        
        # ===== PER-TF RESULTS =====
        if verbose:
            print("\n" + "="*80)
            print("PER-TF PERFORMANCE")
            print("="*80)
        
        # Calculate per-TF means
        per_tf_data = []
        for tf in sorted(results_filtered['tf_knockdown'].unique()):
            tf_results = results_filtered[results_filtered['tf_knockdown'] == tf]
            per_tf_data.append({
                'TF': tf,
                'N': len(tf_results),
                'Baseline_Log2': tf_results['baseline_error'].mean(),
                'ODE_Log2': tf_results['ode_abs_error'].mean(),
                'SCM_Log2': tf_results['scm_abs_error'].mean(),
                'Baseline_Fold': tf_results['baseline_fold_error'].mean(),
                'ODE_Fold': tf_results['ode_fold_error'].mean(),
                'SCM_Fold': tf_results['scm_fold_error'].mean()
            })
        
        per_tf_df = pd.DataFrame(per_tf_data)
        
        if save_to_csv:
            Path(results_dir).mkdir(exist_ok=True)
            per_tf_df.to_csv(f'{results_dir}/per_tf_results.csv', index=False)
            if verbose:
                print(f"\n✓ Saved: {results_dir}/per_tf_results.csv")
        
        if verbose:
            print("\nLog2 Error (mean):")
            print(f"{'TF':<14} {'Baseline':>12} {'ODE':>12} {'SCM':>12}")
            print("-"*50)
            for _, row in per_tf_df.iterrows():
                print(f"{row['TF']:<14} {row['Baseline_Log2']:>12.3f} "
                      f"{row['ODE_Log2']:>12.3f} {row['SCM_Log2']:>12.3f}")
            
            print("\n\nFold Error (mean):")
            print(f"{'TF':<14} {'Baseline':>12} {'ODE':>12} {'SCM':>12}")
            print("-"*50)
            for _, row in per_tf_df.iterrows():
                print(f"{row['TF']:<14} {row['Baseline_Fold']:>11.2f}× "
                      f"{row['ODE_Fold']:>11.2f}× {row['SCM_Fold']:>11.2f}×")
        
        # ===== STRATIFIED RESULTS =====
        if verbose:
            print("\n" + "="*80)
            print("STRATIFIED BY PERTURBATION MAGNITUDE")
            print("="*80)
        
        small = results_filtered[results_filtered['true_fc'].abs() < 0.5]
        large = results_filtered[results_filtered['true_fc'].abs() >= 0.5]
        
        stratified_data = [
            {
                'Group': 'Small (<0.5)',
                'N': len(small),
                'Baseline_Log2': small['baseline_error'].mean(),
                'ODE_Log2': small['ode_abs_error'].mean(),
                'SCM_Log2': small['scm_abs_error'].mean(),
                'Baseline_Fold': small['baseline_fold_error'].mean(),
                'ODE_Fold': small['ode_fold_error'].mean(),
                'SCM_Fold': small['scm_fold_error'].mean()
            },
            {
                'Group': 'Large (≥0.5)',
                'N': len(large),
                'Baseline_Log2': large['baseline_error'].mean(),
                'ODE_Log2': large['ode_abs_error'].mean(),
                'SCM_Log2': large['scm_abs_error'].mean(),
                'Baseline_Fold': large['baseline_fold_error'].mean(),
                'ODE_Fold': large['ode_fold_error'].mean(),
                'SCM_Fold': large['scm_fold_error'].mean()
            }
        ]
        
        stratified_df = pd.DataFrame(stratified_data)
        
        if save_to_csv:
            stratified_df.to_csv(f'{results_dir}/stratified_results.csv', index=False)
            if verbose:
                print(f"\n✓ Saved: {results_dir}/stratified_results.csv")
        
        if verbose:
            print("\nLog2 Error (mean):")
            print(f"{'Group':<19} {'Baseline':>12} {'ODE':>12} {'SCM':>12}")
            print("-"*55)
            for _, row in stratified_df.iterrows():
                print(f"{row['Group']:<19} {row['Baseline_Log2']:>12.3f} "
                      f"{row['ODE_Log2']:>12.3f} {row['SCM_Log2']:>12.3f}")
            
            print("\n\nFold Error (mean):")
            print(f"{'Group':<19} {'Baseline':>12} {'ODE':>12} {'SCM':>12}")
            print("-"*55)
            for _, row in stratified_df.iterrows():
                print(f"{row['Group']:<19} {row['Baseline_Fold']:>11.2f}× "
                      f"{row['ODE_Fold']:>11.2f}× {row['SCM_Fold']:>11.2f}×")
        
        # ===== OVERALL SUMMARY =====
        baseline_mean_log2 = results_filtered['baseline_error'].mean()
        ode_mean_log2 = results_filtered['ode_abs_error'].mean()
        scm_mean_log2 = results_filtered['scm_abs_error'].mean()
        
        baseline_mean_fold = results_filtered['baseline_fold_error'].mean()
        ode_mean_fold = results_filtered['ode_fold_error'].mean()
        scm_mean_fold = results_filtered['scm_fold_error'].mean()
        
        baseline_median_log2 = results_filtered['baseline_error'].median()
        ode_median_log2 = results_filtered['ode_abs_error'].median()
        scm_median_log2 = results_filtered['scm_abs_error'].median()
        
        baseline_median_fold = results_filtered['baseline_fold_error'].median()
        ode_median_fold = results_filtered['ode_fold_error'].median()
        scm_median_fold = results_filtered['scm_fold_error'].median()
        
        overall_data = [
            {
                'Statistic': 'mean',
                'Baseline_Log2': baseline_mean_log2,
                'ODE_Log2': ode_mean_log2,
                'SCM_Log2': scm_mean_log2,
                'Baseline_Fold': baseline_mean_fold,
                'ODE_Fold': ode_mean_fold,
                'SCM_Fold': scm_mean_fold
            },
            {
                'Statistic': 'median',
                'Baseline_Log2': baseline_median_log2,
                'ODE_Log2': ode_median_log2,
                'SCM_Log2': scm_median_log2,
                'Baseline_Fold': baseline_median_fold,
                'ODE_Fold': ode_median_fold,
                'SCM_Fold': scm_median_fold
            }
        ]
        
        overall_df = pd.DataFrame(overall_data)
        
        if save_to_csv:
            overall_df.to_csv(f'{results_dir}/overall_summary.csv', index=False)
            if verbose:
                print("\n" + "="*80)
                print("OVERALL SUMMARY")
                print("="*80)
                print(f"\n✓ Saved: {results_dir}/overall_summary.csv")
        
        if verbose:
            print("\nLog2 Error:")
            print(f"{'Statistic':<12} {'Baseline':>12} {'ODE':>12} {'SCM':>12}")
            print("-"*50)
            for _, row in overall_df.iterrows():
                print(f"{row['Statistic']:<12} {row['Baseline_Log2']:>12.3f} "
                      f"{row['ODE_Log2']:>12.3f} {row['SCM_Log2']:>12.3f}")
            
            print("\nFold Error:")
            print(f"{'Statistic':<12} {'Baseline':>12} {'ODE':>12} {'SCM':>12}")
            print("-"*50)
            for _, row in overall_df.iterrows():
                print(f"{row['Statistic']:<12} {row['Baseline_Fold']:>11.2f}× "
                      f"{row['ODE_Fold']:>11.2f}× {row['SCM_Fold']:>11.2f}×")
            
            print("\n" + "="*80)
            print(f"ALL RESULTS SAVED TO {results_dir}/")
            print("="*80)
            print(f"  - per_tf_results.csv: Per-TF performance (mean)")
            print(f"  - stratified_results.csv: Stratified by effect size (mean)")
            print(f"  - overall_summary.csv: Overall performance (mean & median)")
            print("="*80)
        
        return {
            'n_predictions': len(results_filtered),
            'n_tfs': len(results_filtered['tf_knockdown'].unique()),
            'baseline_mean_log2': baseline_mean_log2,
            'ode_mean_log2': ode_mean_log2,
            'scm_mean_log2': scm_mean_log2,
            'baseline_mean_fold': baseline_mean_fold,
            'ode_mean_fold': ode_mean_fold,
            'scm_mean_fold': scm_mean_fold,
            'baseline_median_log2': baseline_median_log2,
            'ode_median_log2': ode_median_log2,
            'scm_median_log2': scm_median_log2,
            'baseline_median_fold': baseline_median_fold,
            'ode_median_fold': ode_median_fold,
            'scm_median_fold': scm_median_fold
        }
    
    def run_full_experiment(
        self,
        use_cached_params: bool = True,
        save_results: bool = True,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete Investigation D experiment
        
        Args:
            use_cached_params: Use cached rescaled parameters if available
            save_results: Save results to CSV
            verbose: Print detailed progress
            
        Returns:
            (results_df, summary_stats)
        """
        print("="*80)
        print("INVESTIGATION D: Full Experiment Runner")
        print("="*80)
        
        # Load module
        if verbose:
            print("\n[1/7] Loading CRP regulon module...")
        genes, network = self.loader.load_functional_module('crp_regulon')
        pkn = self.loader.build_pkn_dict(genes, network)
        if verbose:
            print(f"✓ {len(genes)} genes, {len(pkn['edges'])} edges")
        
        # Load/rescale parameters
        if verbose:
            print("\n[2/7] Loading parameters...")
        
        param_finder = StableParameterFinder(pkn, cache_dir=str(self.cache_dir))
        
        rescaled_file = self.cache_dir / 'stable_params_rescaled.pkl'
        if use_cached_params and rescaled_file.exists():
            stable_params = param_finder.load_stable_parameters('stable_params_rescaled.pkl')
            if verbose:
                print(f"✓ Loaded cached rescaled parameters")
        else:
            # Load original and rescale
            stable_params = param_finder.load_stable_parameters('stable_params.pkl')
            if stable_params is None:
                raise FileNotFoundError("Original stable_params.pkl not found")
            
            real_wt_data = self.loader.load_wildtype_expression(genes, media='glucose')
            stable_params = self.rescale_parameters(stable_params, real_wt_data, genes, verbose)
            
            # Save rescaled
            with open(rescaled_file, 'wb') as f:
                pickle.dump(stable_params, f)
            if verbose:
                print(f"✓ Saved rescaled parameters to {rescaled_file.name}")
        
        # Load real data
        if verbose:
            print("\n[3/7] Loading real data...")
        real_wt_data = self.loader.load_wildtype_expression(genes, media='glucose')
        if verbose:
            print(f"✓ Real wild-type: {real_wt_data.shape}")
        
        # Generate synthetic data
        if verbose:
            print("\n[4/7] Generating synthetic data...")
        _, synthetic_int_data = self.generate_synthetic_data(
            pkn, stable_params, genes, force_regenerate=True, verbose=verbose
        )
        
        # Prepare training data
        obs_data, int_data = self.prepare_training_data(
            real_wt_data, synthetic_int_data, stable_params, genes, verbose
        )
        
        # Load PCOA structure
        if verbose:
            print("\n[5/7] Loading PCOA structure...")
        pcoa_runner = EcoliPCOARunner(cache_dir=str(self.cache_dir))
        structure, topo = pcoa_runner.load_structure('crp_regulon', formulation='hill')
        
        if structure is None:
            if verbose:
                print("  Discovering structure...")
            structure, topo = pcoa_runner.discover_and_save(
                pkn, 'hill', 'crp_regulon', verbose=verbose
            )
        else:
            if verbose:
                print(f"✓ Loaded PCOA structure")
        
        # Train SCM
        if verbose:
            print("\n[6/7] Training SCM...")
        scm = self.train_scm(structure, topo, obs_data, int_data, genes, verbose)
        
        # Test predictions
        if verbose:
            print("\n[7/7] Testing predictions...")
        knockdown_data = self.loader.load_crispri_knockdowns(genes, media='glucose')
        if verbose:
            print(f"✓ Loaded {len(knockdown_data)} TF knockdowns")
        
        results_df = self.test_predictions(
            scm, param_finder, stable_params, knockdown_data, genes, verbose
        )
        
        # Analyze results
        summary = self.analyze_results(
            results_df, 
            exclude_tfs=['crp'], 
            save_to_csv=save_results,
            results_dir='results',
            verbose=verbose
        )
        
        # Save results
        if save_results:
            all_file = self.cache_dir / 'investigation_d_results_all.csv'
            filtered_file = self.cache_dir / 'investigation_d_results_filtered.csv'
            
            results_df.to_csv(all_file, index=False)
            results_df[results_df['tf_knockdown'] != 'crp'].to_csv(filtered_file, index=False)
            
            if verbose:
                print(f"\n✓ Saved results:")
                print(f"  All TFs:      {all_file.name}")
                print(f"  Filtered:     {filtered_file.name}")
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)
        
        return results_df, summary


def run_ecoli_experiment(
    data_dir: str = 'ecoli_crp_data',
    use_cached_params: bool = True,
    save_results: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to run Investigation D
    
    Args:
        data_dir: Data directory path
        use_cached_params: Use cached rescaled parameters
        save_results: Save results to CSV
        verbose: Print progress
        
    Returns:
        (results_df, summary_stats)
    """
    runner = InvestigationDRunner(data_dir=data_dir)
    return runner.run_full_experiment(
        use_cached_params=use_cached_params,
        save_results=save_results,
        verbose=verbose
    )