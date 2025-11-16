"""
Module 4: Model Evaluation with Smart Clipping

CRITICAL FIX:
- Smart lower bound: max(0.1 × baseline, pred) prevents catastrophic collapse
- Upper bound: min(10.0 × baseline, pred) prevents overshooting
- Eliminates zero predictions from linear extrapolation
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path


class ModelEvaluator:
    def __init__(self, ode_module, cache_dir='cache'):
        self.ode_module = ode_module
        self.cache_dir = Path(cache_dir)
        self.config = ode_module.get_cascade_config()
    
    def _compute_saturated_equilibrium(self, var, candidate_form, candidate_params):
        """Compute equilibrium for saturated variable from FITTED parameters"""
        if candidate_form == 'mm':
            k_shared_deg = candidate_params.get('k_shared_deg', 0.60)
            
            if var == 'Fra1':
                V = candidate_params.get('V_cFos_Fra1')
                return V / k_shared_deg if V else None
            elif var == 'cJun':
                V = candidate_params.get('V_Fra1_cJun')
                return V / k_shared_deg if V else None
        
        elif candidate_form == 'hill':
            k_shared_deg = candidate_params.get('k_shared_deg', 0.60)
            
            if var == 'Fra1':
                V = candidate_params.get('V_cFos_Fra1')
                return V / k_shared_deg if V else None
            elif var == 'cJun':
                V = candidate_params.get('V_Fra1_cJun')
                return V / k_shared_deg if V else None
        
        return None
    
    def predict_baseline(self, test_sample):
        """Baseline predictor: No change model"""
        return dict(test_sample['initial_state'])
    
    def predict_ode(self, candidate_form, candidate_params, test_sample):
        """Predict with candidate ODE from observed baseline"""
        variables = self.config['variables']
        inhibitions = self.config['inhibitions']
        
        k_in = test_sample['k_in']
        baseline_state = test_sample['initial_state']
        I_dict_full = test_sample['interventions']
        
        I_kwargs = {name: val for name, val in I_dict_full.items() if val < 1.0}
        
        # Build solver with candidate parameters
        config_cand = dict(self.config)
        config_cand['parameters'][candidate_form] = candidate_params
        solver = self.ode_module.build_ode_solver(candidate_form, config_cand)
        
        # Start from baseline
        pred_state = solver(k_in, initial_state=baseline_state, **I_kwargs)
        
        if pred_state and all(0.001 < v < 50 for v in pred_state.values()):
            return pred_state
        
        return None
    
    def predict_scm(self, scm_mechanisms, structure, topo_order, test_sample, 
                    deterministic_info, candidate_form, candidate_params,
                    equilibrium_rescue=False):
        """
        Predict with SCM from observed baseline
        
        CRITICAL FIX: Smart clipping using baseline bounds
        """
        variables = self.config['variables']
        inhibitions = self.config['inhibitions']
        
        baseline = test_sample['initial_state']
        interventions = test_sample['interventions']
        
        affected = self._find_affected_variables(structure, interventions, inhibitions)
        
        state = dict(baseline)
        state['k_in'] = test_sample['k_in']
        
        for var in inhibitions.keys():
            I_name = inhibitions[var]
            state[I_name] = interventions[I_name]
        
        # Predict in topological order
        for var in topo_order:
            is_original = var in variables
            is_R = var.startswith('R_')
            
            if not is_original and not is_R:
                continue
            
            # Handle unaffected original variables
            if is_original and var not in affected:
                if var in scm_mechanisms and scm_mechanisms[var] is not None:
                    # Has mechanism (train_saturated=True) - use it
                    pass
                else:
                    # No mechanism
                    if equilibrium_rescue:
                        eq_val = self._compute_saturated_equilibrium(var, candidate_form, candidate_params)
                        if eq_val is not None:
                            state[var] = eq_val
                            continue
                    continue
            
            # R variables
            if is_R:
                if var in scm_mechanisms and scm_mechanisms[var] is not None:
                    mechanism = scm_mechanisms[var]
                    try:
                        feature_values = [state[f] for f in mechanism['features']]
                        X = np.array([feature_values])
                        pred = mechanism['learner'].predict(X)[0]
                        
                        # CRITICAL FIX: Reasonable bounds for R variables (ratios)
                        state[var] = np.clip(pred, 0.01, 100.0)
                    except:
                        # Fallback: compute from current state
                        parts = var.replace('R_', '').split('_')
                        if len(parts) == 2:
                            num, denom = parts[0], parts[1]
                            if num in state and denom in state:
                                state[var] = state[num] / (state[denom] + 1e-10)
                else:
                    # No mechanism - compute from current state
                    parts = var.replace('R_', '').split('_')
                    if len(parts) == 2:
                        num, denom = parts[0], parts[1]
                        if num in state and denom in state:
                            state[var] = state[num] / (state[denom] + 1e-10)
                continue
            
            # Deterministic original variables
            if var in deterministic_info:
                info = deterministic_info[var]
                R_var = info['R_var']
                other_var = info['other_var']
                
                if R_var in state and other_var in state:
                    if info['is_numerator']:
                        pred = state[R_var] * state[other_var]
                    else:
                        pred = state[other_var] / (state[R_var] + 1e-10)
                    
                    # CRITICAL FIX: Bound deterministic calculations too
                    baseline_val = baseline[var]
                    lower_bound = 0.1 * baseline_val
                    upper_bound = 10.0 * baseline_val
                    state[var] = np.clip(pred, lower_bound, upper_bound)
                continue
            
            # Regular affected variable with mechanism
            if var in scm_mechanisms and scm_mechanisms[var] is not None:
                mechanism = scm_mechanisms[var]
                try:
                    feature_values = [state[f] for f in mechanism['features']]
                    X = np.array([feature_values])
                    pred = mechanism['learner'].predict(X)[0]
                    
                    # CRITICAL FIX: Smart bounds using baseline
                    # Prevents catastrophic collapse from linear extrapolation
                    baseline_val = baseline[var]
                    lower_bound = 0.1 * baseline_val
                    upper_bound = 10.0 * baseline_val
                    
                    state[var] = np.clip(pred, lower_bound, upper_bound)
                except Exception as e:
                    # Fallback: use baseline
                    state[var] = baseline[var]
        
        return {v: state[v] for v in variables}
    
    def _find_affected_variables(self, structure, interventions, inhibitions):
        """Find ORIGINAL variables causally affected by interventions"""
        affected = set()
        
        inv_inhib = {I_name: var for var, I_name in inhibitions.items()}
        
        for I_name, I_val in interventions.items():
            if I_val != 1.0 and I_name in inv_inhib:
                affected.add(inv_inhib[I_name])
        
        changed = True
        while changed:
            changed = False
            for var, parents in structure.items():
                if var in affected:
                    continue
                
                for parent in parents:
                    if parent in affected or (parent.startswith('I_') and interventions.get(parent, 1.0) != 1.0):
                        affected.add(var)
                        changed = True
                        break
        
        original_variables = set(self.config['variables'])
        return affected & original_variables
    
    def evaluate_case(self, gt_formulation, candidate_formulation,
                     gt_data, candidate_data, scm_data, 
                     equilibrium_rescue=False, verbose=True):
        """Evaluate one GT×Candidate case"""
        if verbose:
            print(f"\n[Module 4] Evaluating GT={gt_formulation.upper()}, Cand={candidate_formulation.upper()}")
        
        test_data = gt_data['test_data']
        candidate_params = candidate_data['candidate_fit']['fitted_params']
        scm_mechanisms = scm_data['scm_mechanisms']
        structure = scm_data['structure']
        deterministic_info = scm_data.get('deterministic_info', {})
        
        # Get topological order
        import networkx as nx
        G = nx.DiGraph()
        
        for var in structure.keys():
            G.add_node(var)
        
        for var, parents in structure.items():
            for p in parents:
                if p in structure:
                    G.add_edge(p, var)
        
        try:
            topo_order = list(nx.topological_sort(G))
        except:
            topo_order = list(self.config['variables'])
        
        predictions = []
        
        for test_sample in test_data:
            affected = self._find_affected_variables(
                structure, test_sample['interventions'], self.config['inhibitions']
            )
            
            # Get predictions
            ode_pred = self.predict_ode(candidate_formulation, candidate_params, test_sample)
            scm_pred = self.predict_scm(scm_mechanisms, structure, topo_order, test_sample, 
                                       deterministic_info, candidate_formulation, candidate_params,
                                       equilibrium_rescue=equilibrium_rescue)
            baseline_pred = self.predict_baseline(test_sample)
            
            for var in self.config['variables']:
                gt_final = test_sample['final_state'][var]
                gt_initial = test_sample['initial_state'][var]
                
                ode_val = ode_pred[var] if ode_pred else np.nan
                scm_val = scm_pred.get(var, np.nan)
                baseline_val = baseline_pred[var]
                
                # Compute relative errors
                ode_error = abs(gt_final - ode_val) / (gt_final + 1e-10) if ode_pred else np.nan
                scm_error = abs(gt_final - scm_val) / (gt_final + 1e-10) if not np.isnan(scm_val) else np.nan
                baseline_error = abs(gt_final - baseline_val) / (gt_final + 1e-10)
                
                predictions.append({
                    'gt_formulation': gt_formulation,
                    'candidate_formulation': candidate_formulation,
                    'k_in': test_sample['k_in'],
                    'target': test_sample['target'],
                    'I_value': test_sample['I_value'],
                    'variable': var,
                    'is_affected': var in affected,
                    'gt_initial': gt_initial,
                    'gt_final': gt_final,
                    'ode_pred': ode_val,
                    'scm_pred': scm_val,
                    'baseline_pred': baseline_val,
                    'ode_error': ode_error,
                    'scm_error': scm_error,
                    'baseline_error': baseline_error,
                })
        
        # Compute summary
        df = pd.DataFrame(predictions)
        
        summary = {
            'gt_formulation': gt_formulation,
            'candidate_formulation': candidate_formulation,
            'ode_mean_error': df['ode_error'].mean(),
            'ode_log2_error': np.log2(1 + df['ode_error'].mean()),
            'ode_fold_error': 1 + df['ode_error'].mean(),
            'scm_mean_error': df['scm_error'].mean(),
            'scm_log2_error': np.log2(1 + df['scm_error'].mean()),
            'scm_fold_error': 1 + df['scm_error'].mean(),
            'baseline_mean_error': df['baseline_error'].mean(),
            'baseline_log2_error': np.log2(1 + df['baseline_error'].mean()),
            'baseline_fold_error': 1 + df['baseline_error'].mean(),
            'training_loss': candidate_data['candidate_fit']['loss'],
            'scm_training_r2': np.mean(list(scm_data['training_scores'].values())),
        }
        
        if verbose:
            print(f"  Overall: ODE={summary['ode_log2_error']:.3f}, SCM={summary['scm_log2_error']:.3f}, Baseline={summary['baseline_log2_error']:.3f} log2")
        
        return predictions, summary
    
    def run_all_evaluations(self, gt_data_dict, candidate_data_dict, scm_data_dict,
                           equilibrium_rescue=False, force_refresh=False, verbose=True):
        """Evaluate all 9 GT×Candidate combinations"""
        cache_suffix = '_eqrescue' if equilibrium_rescue else ''
        cache_file = self.cache_dir / f'evaluation_results{cache_suffix}.pkl'
        
        if not force_refresh and cache_file.exists():
            if verbose:
                print(f"[Module 4] Loading cached evaluation results")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                return cached['predictions_df'], cached['summary_df']
        
        if verbose:
            print("\n" + "="*80)
            print("[Module 4] Evaluating All Cases")
            print("="*80)
        
        all_predictions = []
        all_summaries = []
        
        formulations = ['mass_action', 'mm', 'hill']
        
        for gt_form in formulations:
            for cand_form in formulations:
                key = (gt_form, cand_form)
                
                predictions, summary = self.evaluate_case(
                    gt_form, cand_form,
                    gt_data_dict[gt_form],
                    candidate_data_dict[key],
                    scm_data_dict[key],
                    equilibrium_rescue=equilibrium_rescue,
                    verbose=verbose
                )
                
                all_predictions.extend(predictions)
                all_summaries.append(summary)
        
        predictions_df = pd.DataFrame(all_predictions)
        summary_df = pd.DataFrame(all_summaries)
        
        with open(cache_file, 'wb') as f:
            pickle.dump({'predictions_df': predictions_df, 'summary_df': summary_df}, f)
        
        if verbose:
            print("\n" + "="*80)
            print("SUMMARY RESULTS")
            print("="*80)
            print(summary_df[['gt_formulation', 'candidate_formulation', 
                            'ode_log2_error', 'scm_log2_error', 'baseline_log2_error']].to_string())
        
        return predictions_df, summary_df


if __name__ == "__main__":
    pass