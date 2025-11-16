"""
Investigation B Runner - Complete Pipeline

UPDATED: Reflects all module fixes
- Module 1: No noise in test data, interpolation k_in
- Module 2: No noise in synthetic data
- Module 3: Stability filtering, no GBM, fixed CV weights
- Module 4: Fixed affected variable propagation
"""

import numpy as np
import pandas as pd
from pathlib import Path

from mapk_module0 import PCOAStructureCache
from mapk_module1 import GroundTruthDataGenerator
from mapk_module2 import CandidateODEFitter  
from mapk_module3 import SCMTrainer
from mapk_module4 import ModelEvaluator
import mapk_pkn as ode_system

# =============================================================================
# RUNNER CLASS
# =============================================================================

class InvestigationBRunner:
    """
    Complete Investigation B pipeline with all fixes
    
    UPDATES:
    - CandidateODEFitter (new class name)
    - Speedup parameters for fast ODE fitting
    - Ridge + DummyRegressor in Module 3
    - Formulation-specific parameter bounds (tight for MA, wide for MM/Hill)
    - EQUAL OPTIMIZATION EFFORT for fair comparison
    """
    
    def __init__(self, ode_module, system_name, cache_dir='cache'):
        self.ode_module = ode_module
        self.system_name = system_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.formulations = ['mass_action', 'mm', 'hill']
        
        # Initialize modules
        self.pcoa_cache = PCOAStructureCache(cache_dir=cache_dir)
        self.gt_generator = GroundTruthDataGenerator(ode_module, cache_dir=cache_dir)
        self.candidate_fitter = CandidateODEFitter(ode_module, cache_dir=cache_dir)
        self.scm_trainer = None  # Initialize after PCOA
        self.evaluator = ModelEvaluator(ode_module, cache_dir=cache_dir)
    
    def run_full_investigation(self, N_obs=50, 
                              test_intervention_strengths=None,
                              synth_intervention_strengths=None,
                              N_synth_reps=10, N_test_reps=5,
                              noise_cv=0.05, seed=42,
                              include_synthetic_baselines=False,
                              stability_threshold=50.0,
                              train_saturated=False,
                              equilibrium_rescue=False,
                              # Parameter perturbation:
                              perturb_gt_params=False,
                              perturbation_cv=0.10,
                              # Test data settings:
                              test_integration_time=2000,
                              test_rtol=1e-3, test_atol=1e-6,
                              force_refresh=False, verbose=True):
        """
        Run complete 3×3 investigation with speedup options
        
        CRITICAL: All formulations now use EQUAL optimization effort for fair comparison
        
        OPTIMIZATION SETTINGS (ALL FORMULATIONS):
            fit_integration_time: 2000 (full convergence)
            fit_rtol, fit_atol: 1e-3, 1e-6 (tight tolerances)
            max_iter: 200 (adequate iterations)
            n_restarts: 5 (thorough random search)
        
        PARAMETER BOUNDS (FORMULATION-SPECIFIC):
            MA: tight [0.85, 1.15] - required for feedback loop stability
            MM/Hill: wide [0.1, 10.0] - realistic practice for unknown systems
        """
        if test_intervention_strengths is None:
            test_intervention_strengths = [0.2]
        
        if synth_intervention_strengths is None:
            synth_intervention_strengths = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        
        if verbose:
            print(f"\n{'='*80}")
            print("INVESTIGATION B: MAPK-IEG CASCADE")
            print(f"{'='*80}")
            print(f"Intervention strengths:")
            print(f"  Test:      {test_intervention_strengths}")
            print(f"  Synthetic: {synth_intervention_strengths}")
            print(f"\nParameter perturbation:")
            if perturb_gt_params:
                print(f"  ENABLED (CV={perturbation_cv}) - testing robustness across parameter space")
            else:
                print(f"  DISABLED - using default parameters")
            print(f"\nFormulation-specific fitting settings:")
            print(f"  ALL FORMULATIONS: t=2000, rtol=1e-3, atol=1e-6, maxiter=200, restarts=5")
            print(f"  *** EQUAL optimization effort for fair comparison ***")
            print(f"\nParameter bounds (formulation-specific):")
            print(f"  Mass Action: tight [0.85, 1.15] (feedback stability)")
            print(f"  MM:          wide [0.1, 10.0] (realistic practice)")
            print(f"  Hill:        wide [0.1, 10.0] (realistic practice)")
            print(f"\nTest data (all formulations):")
            print(f"  Integration: t={test_integration_time}, rtol={test_rtol}, atol={test_atol}")
        
        # ==== MODULE 0: PCOA Structure Discovery ====
        if verbose:
            print("\n" + "="*80)
            print("MODULE 0: PCOA Structure Discovery")
            print("="*80)
        
        pcoa_structures = self.pcoa_cache.discover_and_cache(
            self.ode_module,
            system_name=self.system_name,
            force_refresh=force_refresh,
            verbose=verbose
        )
        
        # Initialize SCM trainer with PCOA structures
        self.scm_trainer = SCMTrainer(self.ode_module, pcoa_structures, cache_dir=self.cache_dir)
        
        # ==== MODULE 1: Ground Truth Data ====
        if verbose:
            print("\n" + "="*80)
            print("MODULE 1: Ground Truth Data Generation")
            print("="*80)
        
        gt_data_dict = {}
        for gt_form in self.formulations:
            gt_data_dict[gt_form] = self.gt_generator.run(
                formulation=gt_form,
                N_obs=N_obs,
                intervention_strengths=test_intervention_strengths,
                N_test_reps=N_test_reps,
                noise_cv=noise_cv,
                perturb_gt_params=perturb_gt_params,
                perturbation_cv=perturbation_cv,
                seed=seed,
                force_refresh=force_refresh,
                verbose=verbose
            )
        
        # ==== MODULE 2: Candidate Fitting ====
        if verbose:
            print("\n" + "="*80)
            print("MODULE 2: Candidate Fitting (EQUAL Effort, Different Bounds)")
            print("="*80)
        
        # ONLY bounds differ based on structural requirements
        fitting_settings = {
            'mass_action': {
                'bound_width': 'tight',  # MA needs tight bounds for feedback stability
                'fit_integration_time': 2000,
                'fit_rtol': 1e-3, 'fit_atol': 1e-6,
                'max_iter': 500, # Stronger fitting effort for MA due to instability
                'n_restarts': 10,
            },
            'mm': {
                'bound_width': 'wide',  
                'fit_integration_time': 2000,  
                'fit_rtol': 1e-3, 'fit_atol': 1e-6,  
                'max_iter': 200, 
                'n_restarts': 5,  
            },
            'hill': {
                'bound_width': 'wide',  
                'fit_integration_time': 2000,  
                'fit_rtol': 1e-3, 'fit_atol': 1e-6,  
                'max_iter': 200,  
                'n_restarts': 5, 
            },
        }
        
        candidate_data_dict = {}
        for gt_form in self.formulations:
            for cand_form in self.formulations:
                # Use settings appropriate for candidate formulation
                settings = fitting_settings[cand_form]
                
                candidate_data_dict[(gt_form, cand_form)] = self.candidate_fitter.run(
                    gt_formulation=gt_form,
                    candidate_formulation=cand_form,
                    gt_data=gt_data_dict[gt_form],
                    intervention_strengths=synth_intervention_strengths,
                    N_synth_reps=N_synth_reps,
                    # Use formulation-specific settings:
                    bound_width=settings['bound_width'],
                    fit_integration_time=settings['fit_integration_time'],
                    fit_rtol=settings['fit_rtol'],
                    fit_atol=settings['fit_atol'],
                    max_iter=settings['max_iter'],
                    n_restarts=settings['n_restarts'],
                    # Test data always uses tight settings:
                    test_integration_time=test_integration_time,
                    test_rtol=test_rtol,
                    test_atol=test_atol,
                    seed=seed,
                    force_refresh=force_refresh,
                    verbose=verbose
                )
        
        # ==== MODULE 3: SCM Training ====
        if verbose:
            print("\n" + "="*80)
            print("MODULE 3: SCM Training (Ridge + DummyRegressor)")
            print("="*80)
        
        scm_data_dict = {}
        for gt_form in self.formulations:
            for cand_form in self.formulations:
                scm_data_dict[(gt_form, cand_form)] = self.scm_trainer.run(
                    gt_formulation=gt_form,
                    candidate_formulation=cand_form,
                    gt_data=gt_data_dict[gt_form],
                    candidate_data=candidate_data_dict[(gt_form, cand_form)],
                    include_synthetic_baselines=include_synthetic_baselines,
                    stability_threshold=stability_threshold,
                    train_saturated=train_saturated,
                    seed=seed,
                    force_refresh=force_refresh,
                    verbose=verbose
                )
        
        # ==== MODULE 4: Evaluation ====
        if verbose:
            print("\n" + "="*80)
            print("MODULE 4: Model Evaluation")
            print("="*80)
        
        predictions_df, summary_df = self.evaluator.run_all_evaluations(
            gt_data_dict=gt_data_dict,
            candidate_data_dict=candidate_data_dict,
            scm_data_dict=scm_data_dict,
            equilibrium_rescue=equilibrium_rescue,
            force_refresh=force_refresh,
            verbose=verbose
        )
        
        return predictions_df, summary_df
    
    def generate_results_table(self, summary_df, metric='log2', save_latex=True):
        """Generate pivot tables for ODE, SCM, and Baseline errors"""
        
        if metric == 'log2':
            ode_col, scm_col = 'ode_log2_error', 'scm_log2_error'
            baseline_col = 'baseline_log2_error'
            title_suffix = "LOG2 ERRORS"
            filename = 'results_table_log2.tex'
        elif metric == 'fold':
            ode_col, scm_col = 'ode_fold_error', 'scm_fold_error'
            baseline_col = 'baseline_fold_error'
            title_suffix = "FOLD ERRORS"
            filename = 'results_table_fold.tex'
        
        # Create pivot tables
        pivot_ode = summary_df.pivot(index='gt_formulation', columns='candidate_formulation', values=ode_col)
        pivot_scm = summary_df.pivot(index='gt_formulation', columns='candidate_formulation', values=scm_col)
        pivot_baseline = summary_df.pivot(index='gt_formulation', columns='candidate_formulation', values=baseline_col)
        
        # Rename
        rename_map = {'mass_action': 'MA', 'mm': 'MM', 'hill': 'Hill'}
        pivot_ode = pivot_ode.rename(index=rename_map, columns=rename_map)
        pivot_scm = pivot_scm.rename(index=rename_map, columns=rename_map)
        pivot_baseline = pivot_baseline.rename(index=rename_map, columns=rename_map)
        
        # Print
        print(f"\n{title_suffix} - ODE")
        print(pivot_ode.to_string())
        print(f"\n{title_suffix} - SCM")
        print(pivot_scm.to_string())
        print(f"\n{title_suffix} - BASELINE")
        print(pivot_baseline.to_string())
        
        if save_latex:
            latex_path = self.cache_dir / filename
            with open(latex_path, 'w') as f:
                f.write(pivot_ode.to_latex(float_format="%.3f"))
                f.write("\n\n")
                f.write(pivot_scm.to_latex(float_format="%.3f"))
                f.write("\n\n")
                f.write(pivot_baseline.to_latex(float_format="%.3f"))
            print(f"\n✓ Saved LaTeX to {latex_path}")
        
        return pivot_ode, pivot_scm, pivot_baseline


"""
Investigation B Replicate Runner

Run multiple replicates with perturbed ground truth parameters.
Each replicate gets its own cache directory and unique seed.

Handles instability:
- Detects outliers using IQR method
"""

"""
Investigation B Replicate Runner - COMPLETE

Runs multiple replicates with perturbed GT parameters.
Generates summary tables for both affected-only and all-variables metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle


class InvestigationBReplicateRunner:
    """Run Investigation B with multiple replicates"""
    
    def __init__(self, ode_module, system_name, super_cache_dir='investigation_b_replicates'):
        self.ode_module = ode_module
        self.system_name = system_name
        self.super_cache_dir = Path(super_cache_dir)
        self.super_cache_dir.mkdir(exist_ok=True)
        self.formulations = ['mass_action', 'mm', 'hill']
    
    def run_replicates(self, n_replicates=10, 
                      perturbation_cv=0.075,
                      base_seed=42,
                      verbose=True,
                      **investigation_kwargs):
        """Run multiple replicates with perturbed GT parameters"""
        all_results = []
        failed_replicates = []
        
        for rep_id in range(n_replicates):
            if verbose:
                print(f"\n{'='*80}")
                print(f"REPLICATE {rep_id+1}/{n_replicates}")
                print(f"{'='*80}")
                print(f"Seed: {base_seed + rep_id}")
                print(f"Parameter perturbation: ±{perturbation_cv*100:.1f}%")
            
            rep_cache_dir = self.super_cache_dir / f'replicate_{rep_id:02d}'
            rep_cache_dir.mkdir(exist_ok=True)
            
            try:
                # Import here to avoid circular dependency
                
                runner = InvestigationBRunner(
                    ode_module=self.ode_module,
                    system_name=self.system_name,
                    cache_dir=str(rep_cache_dir)
                )
                
                predictions_df, summary_df = runner.run_full_investigation(
                    perturb_gt_params=True,
                    perturbation_cv=perturbation_cv,
                    seed=base_seed + rep_id,
                    force_refresh=True,
                    verbose=verbose,
                    **investigation_kwargs
                )
                
                predictions_df['replicate'] = rep_id
                summary_df['replicate'] = rep_id
                
                all_results.append({
                    'replicate': rep_id,
                    'predictions': predictions_df,
                    'summary': summary_df,
                    'status': 'success'
                })
                
                if verbose:
                    print(f"✓ Replicate {rep_id+1} SUCCESS")
                
            except Exception as e:
                if verbose:
                    print(f"✗ Replicate {rep_id+1} FAILED: {e}")
                failed_replicates.append({'replicate': rep_id, 'error': str(e)})
                all_results.append({
                    'replicate': rep_id,
                    'predictions': None,
                    'summary': None,
                    'status': 'failed'
                })
        
        aggregate_summary, stability_report = self._aggregate_results(
            all_results, failed_replicates, verbose=verbose
        )
        
        self._save_results(all_results, aggregate_summary, stability_report)
        
        return all_results, aggregate_summary, stability_report
    
    def _aggregate_results(self, all_results, failed_replicates, verbose=True):
        """Aggregate with outlier detection"""
        successful = [r['summary'] for r in all_results if r['status'] == 'success']
        
        if len(successful) == 0:
            print("\n✗ ALL REPLICATES FAILED!")
            return None, {'failed_replicates': failed_replicates, 'outliers': []}
        
        combined_df = pd.concat(successful, ignore_index=True)
        cleaned_df, outliers = self._detect_outliers(combined_df)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"AGGREGATION SUMMARY")
            print(f"{'='*80}")
            print(f"Total replicates: {len(all_results)}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed_replicates)}")
            print(f"Outliers dropped: {len(outliers)}")
        
        agg_results = []
        for gt in self.formulations:
            for cand in self.formulations:
                subset = cleaned_df[
                    (cleaned_df['gt_formulation'] == gt) & 
                    (cleaned_df['candidate_formulation'] == cand)
                ]
                
                if len(subset) == 0:
                    continue
                
                agg_results.append({
                    'gt_formulation': gt,
                    'candidate_formulation': cand,
                    'n_valid': len(subset),
                    'ode_log2_mean': subset['ode_log2_error'].mean(),
                    'ode_log2_std': subset['ode_log2_error'].std(),
                    'scm_log2_mean': subset['scm_log2_error'].mean(),
                    'scm_log2_std': subset['scm_log2_error'].std(),
                    'baseline_log2_mean': subset['baseline_log2_error'].mean(),
                    'baseline_log2_std': subset['baseline_log2_error'].std(),
                    'ode_fold_mean': subset['ode_fold_error'].mean(),
                    'ode_fold_std': subset['ode_fold_error'].std(),
                    'scm_fold_mean': subset['scm_fold_error'].mean(),
                    'scm_fold_std': subset['scm_fold_error'].std(),
                    'baseline_fold_mean': subset['baseline_fold_error'].mean(),
                    'baseline_fold_std': subset['baseline_fold_error'].std(),
                })
        
        aggregate_df = pd.DataFrame(agg_results)
        
        stability_report = {
            'failed_replicates': failed_replicates,
            'outliers': outliers,
            'total': len(all_results),
            'successful': len(successful),
        }
        
        return aggregate_df, stability_report
    
    def _detect_outliers(self, combined_df):
        """Smart outlier detection on summary data"""
        outliers = []
        keep_indices = []
        
        for gt in self.formulations:
            for cand in self.formulations:
                mask = (combined_df['gt_formulation'] == gt) & \
                       (combined_df['candidate_formulation'] == cand)
                subset = combined_df[mask]
                
                if len(subset) < 3:
                    keep_indices.extend(subset.index.tolist())
                    continue
                
                ode_out = self._is_outlier(subset['ode_log2_error'])
                scm_out = self._is_outlier(subset['scm_log2_error'])
                base_out = self._is_outlier(subset['baseline_log2_error'])
                
                for idx in subset.index:
                    pos = subset.index.get_loc(idx)
                    
                    ode_is_out = ode_out[pos]
                    scm_is_out = scm_out[pos]
                    base_is_out = base_out[pos]
                    
                    if ode_is_out and (scm_is_out or base_is_out):
                        outliers.append({
                            'gt': gt,
                            'candidate': cand,
                            'replicate': subset.loc[idx, 'replicate'],
                            'reason': 'systemic',
                        })
                    else:
                        keep_indices.append(idx)
        
        cleaned_df = combined_df.loc[keep_indices].copy()
        return cleaned_df, outliers
    
    def _detect_outliers_df(self, df):
        """Apply outlier detection to recomputed dataframe"""
        outliers = []
        keep_indices = []
        
        for gt in self.formulations:
            for cand in self.formulations:
                mask = (df['gt_formulation'] == gt) & (df['candidate_formulation'] == cand)
                subset = df[mask]
                
                if len(subset) < 3:
                    keep_indices.extend(subset.index.tolist())
                    continue
                
                ode_out = self._is_outlier(subset['ode_log2_error'])
                scm_out = self._is_outlier(subset['scm_log2_error'])
                base_out = self._is_outlier(subset['baseline_log2_error'])
                
                for idx in subset.index:
                    pos = subset.index.get_loc(idx)
                    
                    if ode_out[pos] and (scm_out[pos] or base_out[pos]):
                        outliers.append({
                            'gt': gt,
                            'candidate': cand,
                            'replicate': subset.loc[idx, 'replicate'],
                        })
                    else:
                        keep_indices.append(idx)
        
        return df.loc[keep_indices].copy(), outliers
    
    def _is_outlier(self, series):
        """IQR outlier detection"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        return (series > upper).values
    
    def _aggregate_df(self, df):
        """Aggregate recomputed dataframe"""
        agg_results = []
        
        for gt in self.formulations:
            for cand in self.formulations:
                subset = df[
                    (df['gt_formulation'] == gt) &
                    (df['candidate_formulation'] == cand)
                ]
                
                if len(subset) == 0:
                    continue
                
                agg_results.append({
                    'gt_formulation': gt,
                    'candidate_formulation': cand,
                    'n_replicates': len(subset),
                    'ode_log2_mean': subset['ode_log2_error'].mean(),
                    'ode_log2_std': subset['ode_log2_error'].std(),
                    'scm_log2_mean': subset['scm_log2_error'].mean(),
                    'scm_log2_std': subset['scm_log2_error'].std(),
                    'baseline_log2_mean': subset['baseline_log2_error'].mean(),
                    'baseline_log2_std': subset['baseline_log2_error'].std(),
                    'ode_fold_mean': subset['ode_fold_error'].mean(),
                    'ode_fold_std': subset['ode_fold_error'].std(),
                    'scm_fold_mean': subset['scm_fold_error'].mean(),
                    'scm_fold_std': subset['scm_fold_error'].std(),
                    'baseline_fold_mean': subset['baseline_fold_error'].mean(),
                    'baseline_fold_std': subset['baseline_fold_error'].std(),
                })
        
        return pd.DataFrame(agg_results)
    
    def _save_results(self, all_results, aggregate_df, stability_report):
        """Save all results"""
        if aggregate_df is not None:
            aggregate_df.to_csv(self.super_cache_dir / 'aggregate_summary.csv', index=False)
            print(f"\n✓ Saved: {self.super_cache_dir / 'aggregate_summary.csv'}")
        
        report_path = self.super_cache_dir / 'stability_report.txt'
        with open(report_path, 'w') as f:
            f.write("STABILITY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total: {stability_report['total']}\n")
            f.write(f"Successful: {stability_report['successful']}\n")
            f.write(f"Failed: {len(stability_report['failed_replicates'])}\n")
            f.write(f"Outliers dropped: {len(stability_report['outliers'])}\n\n")
            
            if stability_report['failed_replicates']:
                f.write("FAILED REPLICATES:\n")
                for fail in stability_report['failed_replicates']:
                    f.write(f"  Rep {fail['replicate']}: {fail['error']}\n")
                f.write("\n")
            
            if stability_report['outliers']:
                f.write("OUTLIERS:\n")
                for out in stability_report['outliers']:
                    f.write(f"  {out['gt']}→{out['candidate']}, rep={out['replicate']}, {out['reason']}\n")
        
        print(f"✓ Saved: {report_path}")
    
    def generate_summary_tables(self, save_latex=True):
        """
        Generate summary tables from cached evaluation results
        
        Computes BOTH affected-only and all-variables metrics
        Returns all-variables as main tables
        """
        
        print(f"\n{'='*80}")
        print("RECOMPUTING SUMMARY TABLES FROM CACHE")
        print(f"{'='*80}")
        
        rename = {'mass_action': 'MA', 'mm': 'MM', 'hill': 'Hill'}
        
        # Collect data from all successful replicates
        all_summaries_affected = []
        all_summaries_all = []
        
        for rep_dir in sorted(self.super_cache_dir.glob('replicate_*')):
            rep_id = int(rep_dir.name.split('_')[1])
            eval_file = rep_dir / 'evaluation_results.pkl'
            
            if not eval_file.exists():
                continue
            
            print(f"  Loading replicate {rep_id}...", end=' ')
            
            with open(eval_file, 'rb') as f:
                eval_data = pickle.load(f)
            
            predictions_df = eval_data['predictions_df']
            
            # Compute for all 9 cases
            for gt_form in self.formulations:
                for cand_form in self.formulations:
                    # AFFECTED ONLY
                    subset_affected = predictions_df[
                        (predictions_df['gt_formulation'] == gt_form) &
                        (predictions_df['candidate_formulation'] == cand_form) &
                        (predictions_df['is_affected'] == True)
                    ]
                    
                    if len(subset_affected) > 0:
                        gt_vals = subset_affected['gt_final'].values
                        ode_vals = subset_affected['ode_pred'].values
                        scm_vals = subset_affected['scm_pred'].values
                        baseline_vals = subset_affected['baseline_pred'].values if 'baseline_pred' in subset_affected.columns else subset_affected['gt_initial'].values
                        
                        valid = ~np.isnan(ode_vals) & ~np.isnan(scm_vals)
                        
                        if valid.sum() > 0:
                            ode_log2 = np.abs(np.log2((ode_vals[valid] + 1e-10) / (gt_vals[valid] + 1e-10))).mean()
                            scm_log2 = np.abs(np.log2((scm_vals[valid] + 1e-10) / (gt_vals[valid] + 1e-10))).mean()
                            baseline_log2 = np.abs(np.log2((baseline_vals[valid] + 1e-10) / (gt_vals[valid] + 1e-10))).mean()
                            
                            all_summaries_affected.append({
                                'replicate': rep_id,
                                'gt_formulation': gt_form,
                                'candidate_formulation': cand_form,
                                'ode_log2_error': ode_log2,
                                'scm_log2_error': scm_log2,
                                'baseline_log2_error': baseline_log2,
                                'ode_fold_error': 2 ** ode_log2,
                                'scm_fold_error': 2 ** scm_log2,
                                'baseline_fold_error': 2 ** baseline_log2,
                            })
                    
                    # ALL VARIABLES
                    subset_all = predictions_df[
                        (predictions_df['gt_formulation'] == gt_form) &
                        (predictions_df['candidate_formulation'] == cand_form)
                    ]
                    
                    if len(subset_all) > 0:
                        gt_vals = subset_all['gt_final'].values
                        ode_vals = subset_all['ode_pred'].values
                        scm_vals = subset_all['scm_pred'].values
                        baseline_vals = subset_all['baseline_pred'].values if 'baseline_pred' in subset_all.columns else subset_all['gt_initial'].values
                        
                        valid = ~np.isnan(ode_vals) & ~np.isnan(scm_vals)
                        
                        if valid.sum() > 0:
                            ode_log2 = np.abs(np.log2((ode_vals[valid] + 1e-10) / (gt_vals[valid] + 1e-10))).mean()
                            scm_log2 = np.abs(np.log2((scm_vals[valid] + 1e-10) / (gt_vals[valid] + 1e-10))).mean()
                            baseline_log2 = np.abs(np.log2((baseline_vals[valid] + 1e-10) / (gt_vals[valid] + 1e-10))).mean()
                            
                            all_summaries_all.append({
                                'replicate': rep_id,
                                'gt_formulation': gt_form,
                                'candidate_formulation': cand_form,
                                'ode_log2_error': ode_log2,
                                'scm_log2_error': scm_log2,
                                'baseline_log2_error': baseline_log2,
                                'ode_fold_error': 2 ** ode_log2,
                                'scm_fold_error': 2 ** scm_log2,
                                'baseline_fold_error': 2 ** baseline_log2,
                            })
            
            print("✓")
        
        df_affected = pd.DataFrame(all_summaries_affected)
        df_all = pd.DataFrame(all_summaries_all)
        
        # Apply outlier detection to all-variables data
        print(f"\nApplying outlier detection to all-variables data...")
        df_all_cleaned, outliers_all = self._detect_outliers_df(df_all)
        print(f"  Outliers dropped: {len(outliers_all)}")
        
        # Aggregate both
        agg_affected = self._aggregate_df(df_affected)
        agg_all = self._aggregate_df(df_all_cleaned)
        
        # Save both aggregates
        agg_affected.to_csv(self.super_cache_dir / 'aggregate_affected_only.csv', index=False)
        agg_all.to_csv(self.super_cache_dir / 'aggregate_all_variables.csv', index=False)
        
        return agg_all, stability_report  # Return all-variables as main result
    
    def _detect_outliers_df(self, df):
        """Apply outlier detection to recomputed dataframe"""
        outliers = []
        keep_indices = []
        
        for gt in self.formulations:
            for cand in self.formulations:
                mask = (df['gt_formulation'] == gt) & (df['candidate_formulation'] == cand)
                subset = df[mask]
                
                if len(subset) < 3:
                    keep_indices.extend(subset.index.tolist())
                    continue
                
                ode_out = self._is_outlier(subset['ode_log2_error'])
                scm_out = self._is_outlier(subset['scm_log2_error'])
                base_out = self._is_outlier(subset['baseline_log2_error'])
                
                for idx in subset.index:
                    pos = subset.index.get_loc(idx)
                    
                    if ode_out[pos] and (scm_out[pos] or base_out[pos]):
                        outliers.append({
                            'gt': gt,
                            'candidate': cand,
                            'replicate': subset.loc[idx, 'replicate'],
                        })
                    else:
                        keep_indices.append(idx)
        
        return df.loc[keep_indices].copy(), outliers
    
    def _is_outlier(self, series):
        """IQR outlier detection"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        return (series > upper).values
    
    def _aggregate_df(self, df):
        """Aggregate recomputed dataframe"""
        agg_results = []
        
        for gt in self.formulations:
            for cand in self.formulations:
                subset = df[
                    (df['gt_formulation'] == gt) &
                    (df['candidate_formulation'] == cand)
                ]
                
                if len(subset) == 0:
                    continue
                
                agg_results.append({
                    'gt_formulation': gt,
                    'candidate_formulation': cand,
                    'n_replicates': len(subset),
                    'ode_log2_mean': subset['ode_log2_error'].mean(),
                    'ode_log2_std': subset['ode_log2_error'].std(),
                    'scm_log2_mean': subset['scm_log2_error'].mean(),
                    'scm_log2_std': subset['scm_log2_error'].std(),
                    'baseline_log2_mean': subset['baseline_log2_error'].mean(),
                    'baseline_log2_std': subset['baseline_log2_error'].std(),
                    'ode_fold_mean': subset['ode_fold_error'].mean(),
                    'ode_fold_std': subset['ode_fold_error'].std(),
                    'scm_fold_mean': subset['scm_fold_error'].mean(),
                    'scm_fold_std': subset['scm_fold_error'].std(),
                    'baseline_fold_mean': subset['baseline_fold_error'].mean(),
                    'baseline_fold_std': subset['baseline_fold_error'].std(),
                })
        
        return pd.DataFrame(agg_results)
    
    def generate_summary_tables(self, save_latex=True):
        """
        Generate summary tables - ALL VARIABLES VERSION
        
        Automatically recomputes from cache for both affected and all variables
        Returns all-variables tables as primary result
        
        Args:
            save_latex: Whether to save LaTeX files (default True)
        """
        
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY TABLES FROM CACHE")
        print(f"{'='*80}")
        
        rename = {'mass_action': 'MA', 'mm': 'MM', 'hill': 'Hill'}
        
        # Recompute from cache
        all_summaries_affected = []
        all_summaries_all = []
        
        for rep_dir in sorted(self.super_cache_dir.glob('replicate_*')):
            rep_id = int(rep_dir.name.split('_')[1])
            eval_file = rep_dir / 'evaluation_results.pkl'
            
            if not eval_file.exists():
                continue
            
            with open(eval_file, 'rb') as f:
                eval_data = pickle.load(f)
            
            predictions_df = eval_data['predictions_df']
            
            for gt_form in self.formulations:
                for cand_form in self.formulations:
                    # Affected only
                    subset_aff = predictions_df[
                        (predictions_df['gt_formulation'] == gt_form) &
                        (predictions_df['candidate_formulation'] == cand_form) &
                        (predictions_df['is_affected'] == True)
                    ]
                    
                    if len(subset_aff) > 0:
                        valid = ~np.isnan(subset_aff['ode_pred']) & ~np.isnan(subset_aff['scm_pred'])
                        if valid.sum() > 0:
                            gt = subset_aff['gt_final'].values[valid]
                            ode = subset_aff['ode_pred'].values[valid]
                            scm = subset_aff['scm_pred'].values[valid]
                            base = (subset_aff['baseline_pred'].values[valid] if 'baseline_pred' in subset_aff.columns 
                                   else subset_aff['gt_initial'].values[valid])
                            
                            ode_log2 = np.abs(np.log2((ode + 1e-10) / (gt + 1e-10))).mean()
                            scm_log2 = np.abs(np.log2((scm + 1e-10) / (gt + 1e-10))).mean()
                            baseline_log2 = np.abs(np.log2((base + 1e-10) / (gt + 1e-10))).mean()
                            
                            all_summaries_affected.append({
                                'replicate': rep_id,
                                'gt_formulation': gt_form,
                                'candidate_formulation': cand_form,
                                'ode_log2_error': ode_log2,
                                'scm_log2_error': scm_log2,
                                'baseline_log2_error': baseline_log2,
                                'ode_fold_error': 2 ** ode_log2,
                                'scm_fold_error': 2 ** scm_log2,
                                'baseline_fold_error': 2 ** baseline_log2,
                            })
                    
                    # All variables
                    subset_all = predictions_df[
                        (predictions_df['gt_formulation'] == gt_form) &
                        (predictions_df['candidate_formulation'] == cand_form)
                    ]
                    
                    if len(subset_all) > 0:
                        valid = ~np.isnan(subset_all['ode_pred']) & ~np.isnan(subset_all['scm_pred'])
                        if valid.sum() > 0:
                            gt = subset_all['gt_final'].values[valid]
                            ode = subset_all['ode_pred'].values[valid]
                            scm = subset_all['scm_pred'].values[valid]
                            base = (subset_all['baseline_pred'].values[valid] if 'baseline_pred' in subset_all.columns 
                                   else subset_all['gt_initial'].values[valid])
                            
                            ode_log2 = np.abs(np.log2((ode + 1e-10) / (gt + 1e-10))).mean()
                            scm_log2 = np.abs(np.log2((scm + 1e-10) / (gt + 1e-10))).mean()
                            baseline_log2 = np.abs(np.log2((base + 1e-10) / (gt + 1e-10))).mean()
                            
                            all_summaries_all.append({
                                'replicate': rep_id,
                                'gt_formulation': gt_form,
                                'candidate_formulation': cand_form,
                                'ode_log2_error': ode_log2,
                                'scm_log2_error': scm_log2,
                                'baseline_log2_error': baseline_log2,
                                'ode_fold_error': 2 ** ode_log2,
                                'scm_fold_error': 2 ** scm_log2,
                                'baseline_fold_error': 2 ** baseline_log2,
                            })
        
        df_affected = pd.DataFrame(all_summaries_affected)
        df_all = pd.DataFrame(all_summaries_all)
        
        print(f"\n  Collected {len(df_affected)} affected-only datapoints")
        print(f"  Collected {len(df_all)} all-variables datapoints")
        
        # Apply outlier detection
        print(f"\n  Applying outlier detection...")
        df_all_cleaned, outliers_all = self._detect_outliers_df(df_all)
        print(f"    Outliers dropped: {len(outliers_all)}")
        
        # Aggregate
        agg_affected = self._aggregate_df(df_affected)
        agg_all = self._aggregate_df(df_all_cleaned)
        
        # Display ALL VARIABLES (main result)
        print("\n" + "="*80)
        print("LOG2 ERRORS - ALL VARIABLES (mean ± std)")
        print("="*80)
        
        log2_rows = []
        for _, row in agg_all.iterrows():
            gt = rename[row['gt_formulation']]
            cand = rename[row['candidate_formulation']]
            
            log2_rows.append({
                'GT→Candidate': f"{gt}→{cand}",
                'ODE': f"{row['ode_log2_mean']:.3f}±{row['ode_log2_std']:.3f}",
                'SCM': f"{row['scm_log2_mean']:.3f}±{row['scm_log2_std']:.3f}",
                'Baseline': f"{row['baseline_log2_mean']:.3f}±{row['baseline_log2_std']:.3f}",
                'n': int(row['n_replicates']),
            })
        
        log2_table = pd.DataFrame(log2_rows)
        print(log2_table.to_string(index=False))
        
        print("\n" + "="*80)
        print("FOLD ERRORS - ALL VARIABLES (mean ± std)")
        print("="*80)
        
        fold_rows = []
        for _, row in agg_all.iterrows():
            gt = rename[row['gt_formulation']]
            cand = rename[row['candidate_formulation']]
            
            fold_rows.append({
                'GT→Candidate': f"{gt}→{cand}",
                'ODE': f"{row['ode_fold_mean']:.3f}±{row['ode_fold_std']:.3f}",
                'SCM': f"{row['scm_fold_mean']:.3f}±{row['scm_fold_std']:.3f}",
                'Baseline': f"{row['baseline_fold_mean']:.3f}±{row['baseline_fold_std']:.3f}",
                'n': int(row['n_replicates']),
            })
        
        fold_table = pd.DataFrame(fold_rows)
        print(fold_table.to_string(index=False))
        
        # Save LaTeX
        if save_latex:
            latex_path = self.super_cache_dir / 'summary_tables_all_vars.tex'
            with open(latex_path, 'w') as f:
                f.write("% LOG2 ERRORS - ALL VARIABLES\n")
                f.write(log2_table.to_latex(index=False))
                f.write("\n\n% FOLD ERRORS - ALL VARIABLES\n")
                f.write(fold_table.to_latex(index=False))
            print(f"\n✓ Saved LaTeX (all vars): {latex_path}")
            
            # Affected only (supplement)
            log2_aff_rows = []
            for _, row in agg_affected.iterrows():
                gt = rename[row['gt_formulation']]
                cand = rename[row['candidate_formulation']]
                
                log2_aff_rows.append({
                    'GT→Candidate': f"{gt}→{cand}",
                    'ODE': f"{row['ode_log2_mean']:.3f}±{row['ode_log2_std']:.3f}",
                    'SCM': f"{row['scm_log2_mean']:.3f}±{row['scm_log2_std']:.3f}",
                    'n': int(row['n_replicates']),
                })
            
            latex_path_aff = self.super_cache_dir / 'summary_tables_affected.tex'
            with open(latex_path_aff, 'w') as f:
                f.write("% LOG2 ERRORS - AFFECTED ONLY\n")
                f.write(pd.DataFrame(log2_aff_rows).to_latex(index=False))
            print(f"✓ Saved LaTeX (affected): {latex_path_aff}")
        
        return log2_table, fold_table

