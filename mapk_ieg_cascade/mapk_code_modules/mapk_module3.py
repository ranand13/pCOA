"""
Module 3: Structural Causal Model Training

CRITICAL FIXES:
1. Added DummyRegressor for constant targets (CV~0)
2. Ridge regularization instead of LinearRegression (prevents overfitting)
3. Removed GBM (wasn't in successful Investigation B)
4. Stability filtering (max value < threshold)
5. Fixed CV predictions with proper sample weights
6. Proper handling of DummyRegressor (no sample_weight support)

Handles three types of variables:
1. Regular (original + R): Learn from parents
2. Constant: Use baseline (saturated with no parents)
3. Deterministic: Compute from R equation (original vars matched to f_R_*)
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import KFold


class CausalSuperLearner:
    """Super-learner ensemble for causal mechanism learning with regularization"""
    
    def __init__(self):
        # CRITICAL: DummyRegressor + Ridge regularization
        # Matches original successful Investigation B architecture
        self.base_learners = [
            ('constant', DummyRegressor(strategy='mean')),  # For constant targets!
            ('linear', Ridge(alpha=0.1)),
            ('poly2', Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('ridge', Ridge(alpha=0.1))
            ])),
            ('poly3', Pipeline([
                ('poly', PolynomialFeatures(degree=3, include_bias=False)),
                ('ridge', Ridge(alpha=1.0))  # Stronger regularization for degree 3
            ])),
            ('rf', RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42
            )),
        ]
        self.weights = None
        self.fitted_learners = []
    
    def fit(self, X, y, sample_weight=None):
        """Fit all base learners with optional sample weights"""
        
        self.fitted_learners = []
        cv_predictions = []
        
        # CRITICAL FIX: Manual CV with proper sample weight handling
        if sample_weight is not None:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for name, learner in self.base_learners:
                preds = np.zeros(len(y))
                
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]
                    w_train = sample_weight[train_idx]
                    
                    learner_copy = clone(learner)
                    
                    # DummyRegressor doesn't support sample_weight
                    if isinstance(learner_copy, DummyRegressor):
                        learner_copy.fit(X_train, y_train)
                    elif isinstance(learner_copy, Pipeline):
                        final_step_name = learner_copy.steps[-1][0]
                        learner_copy.fit(X_train, y_train, **{f'{final_step_name}__sample_weight': w_train})
                    else:
                        # Ridge and RandomForest support sample_weight directly
                        learner_copy.fit(X_train, y_train, sample_weight=w_train)
                    
                    preds[val_idx] = learner_copy.predict(X_val)
                
                cv_predictions.append(preds)
        else:
            # Without weights, use standard cross_val_predict
            from sklearn.model_selection import cross_val_predict
            for name, learner in self.base_learners:
                try:
                    cv_pred = cross_val_predict(learner, X, y, cv=5)
                    cv_predictions.append(cv_pred)
                except:
                    # Fallback: fit on full data and predict
                    learner_copy = clone(learner)
                    learner_copy.fit(X, y)
                    cv_predictions.append(learner_copy.predict(X))
        
        # Now fit on full data for final models
        for name, learner in self.base_learners:
            learner_copy = clone(learner)
            
            if sample_weight is not None:
                if isinstance(learner_copy, DummyRegressor):
                    learner_copy.fit(X, y)
                elif isinstance(learner_copy, Pipeline):
                    final_step_name = learner_copy.steps[-1][0]
                    learner_copy.fit(X, y, **{f'{final_step_name}__sample_weight': sample_weight})
                else:
                    learner_copy.fit(X, y, sample_weight=sample_weight)
            else:
                learner_copy.fit(X, y)
            
            self.fitted_learners.append((name, learner_copy))
        
        # Meta-learning with sample weights
        Z = np.column_stack(cv_predictions)
        meta_learner = Ridge(alpha=0.01, fit_intercept=False, positive=True)
        
        if sample_weight is not None:
            meta_learner.fit(Z, y, sample_weight=sample_weight)
        else:
            meta_learner.fit(Z, y)
        
        self.weights = meta_learner.coef_
        self.weights /= self.weights.sum()
        
        return self
    
    def predict(self, X):
        """Weighted ensemble prediction"""
        predictions = []
        for name, learner in self.fitted_learners:
            predictions.append(learner.predict(X))
        
        Z = np.column_stack(predictions)
        return Z @ self.weights


class SCMTrainer:
    def __init__(self, ode_module, pcoa_structures, cache_dir='cache'):
        self.ode_module = ode_module
        self.pcoa_structures = pcoa_structures
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.config = ode_module.get_cascade_config()
        self.original_variables = set(self.config['variables'])
    
    def _parse_R_equation(self, eq_name, equation_expr, var_name):
        """
        Parse R equation to extract deterministic formula
        
        Example: f_R_cJun_Fra1: R_cJun_Fra1 - cJun/Fra1 = 0
        Means: R = cJun/Fra1
        
        If var_name = 'cJun' (numerator):
            → cJun = R_cJun_Fra1 * Fra1
            → {'R_var': 'R_cJun_Fra1', 'other_var': 'Fra1', 'is_numerator': True}
        
        If var_name = 'Fra1' (denominator):
            → Fra1 = cJun / R_cJun_Fra1
            → {'R_var': 'R_cJun_Fra1', 'other_var': 'cJun', 'is_numerator': False}
        """
        # Extract R variable name from equation name
        R_var_name = eq_name.replace('f_', '')  # 'R_cJun_Fra1'
        
        # Parse R_X_Y to get X and Y
        parts = R_var_name.replace('R_', '').split('_')
        if len(parts) != 2:
            return None
        
        var1, var2 = parts[0], parts[1]  # X, Y from R_X_Y
        
        # R_X_Y means R = X/Y
        # So: X = R*Y (numerator) and Y = X/R (denominator)
        
        if var_name == var1:
            # var is numerator: var = R * var2
            return {
                'R_var': R_var_name,
                'other_var': var2,
                'is_numerator': True
            }
        elif var_name == var2:
            # var is denominator: var = var1 / R
            return {
                'R_var': R_var_name,
                'other_var': var1,
                'is_numerator': False
            }
        else:
            return None
    
    def train_scm(self, candidate_form, obs_data, synthetic_data, 
                  include_synthetic_baselines=False,
                  stability_threshold=50.0,
                  train_saturated=False,
                  verbose=True):
        """
        Train SCM on BOTH observational + synthetic data
        
        CRITICAL FIXES:
        1. Filters unstable synthetic data
        2. Optionally includes initial_state from synthetic data
        3. Uses 80/20 weighting (obs is ground truth)
        4. CV predictions now properly weighted
        5. DummyRegressor + Ridge regularization
        
        Args:
            candidate_form: 'mass_action', 'mm', or 'hill'
            obs_data: Ground truth observational data (numpy array)
            synthetic_data: Interventional data from candidate ODE (list of dicts)
            include_synthetic_baselines: If True, add initial_state as separate samples
            stability_threshold: Maximum value for stability check
            train_saturated: If True, train models for saturated/constant variables
                            (only k_in as feature). Default False (skip training).
                            Use True to test if learning average behavior helps.
            verbose: Print detailed info
        """
        if verbose:
            print(f"    Training SCM with {candidate_form} structure...")
        
        # Unpack PCOA results (4 items)
        structure, topo_order, var_to_equation, selected_equations = self.pcoa_structures[candidate_form]
        
        variables = self.config['variables']
        inhibitions = self.config['inhibitions']
        
        # Determine all variables (original + R variables)
        all_variables = set(variables)
        for var in structure.keys():
            if var.startswith('R_'):
                all_variables.add(var)
        
        # ==== Convert observational data (numpy array → dict) ====
        obs_samples = []
        for row in obs_data:
            k_in = row[0]
            sample = {'k_in': k_in}
            
            # Add variable values
            for i, var in enumerate(variables):
                sample[var] = row[i+1]
            
            # Add wild-type interventions (all I=1.0)
            for I_name in inhibitions.values():
                sample[I_name] = 1.0
            
            # Compute R variables from observational data
            for var in all_variables:
                if var.startswith('R_'):
                    parts = var.replace('R_', '').split('_')
                    if len(parts) == 2:
                        numerator, denominator = parts[0], parts[1]
                        if numerator in sample and denominator in sample:
                            sample[var] = sample[numerator] / (sample[denominator] + 1e-10)
            
            obs_samples.append(sample)
        
        # ==== Filter and prepare synthetic interventional data ====
        n_synthetic_total = len(synthetic_data)
        
        # CRITICAL FIX: Filter unstable samples
        stable_synthetic = []
        for sample in synthetic_data:
            max_val = max(sample['final_state'][v] for v in variables)
            if max_val < stability_threshold:
                stable_synthetic.append(sample)
        
        n_filtered = n_synthetic_total - len(stable_synthetic)
        
        if verbose:
            print(f"      Synthetic data: {n_synthetic_total} total, {len(stable_synthetic)} stable, {n_filtered} filtered (>{stability_threshold})")
            if n_filtered > n_synthetic_total * 0.3:
                print(f"      ⚠️  WARNING: {n_filtered/n_synthetic_total*100:.1f}% of synthetic data was unstable!")
        
        # Build synthetic sample list
        synth_samples = []
        
        # Optional: Add initial states (baselines) as observational-like data
        if include_synthetic_baselines:
            for sample in stable_synthetic:
                baseline_row = {
                    'k_in': sample['k_in'],
                    **{inhibitions[v]: 1.0 for v in inhibitions.keys()},  # All I=1.0
                    **sample['initial_state']
                }
                
                # Compute R variables for baseline
                for var in all_variables:
                    if var.startswith('R_'):
                        parts = var.replace('R_', '').split('_')
                        if len(parts) == 2:
                            numerator, denominator = parts[0], parts[1]
                            if numerator in sample['initial_state'] and denominator in sample['initial_state']:
                                baseline_row[var] = sample['initial_state'][numerator] / (sample['initial_state'][denominator] + 1e-10)
                
                synth_samples.append(baseline_row)
        
        # Add final states (interventional)
        for sample in stable_synthetic:
            perturbed_row = {
                'k_in': sample['k_in'],
                **{inhibitions[v]: sample['interventions'][inhibitions[v]] for v in inhibitions.keys()},
                **sample['final_state']
            }
            
            # Compute R variables for perturbed
            for var in all_variables:
                if var.startswith('R_'):
                    parts = var.replace('R_', '').split('_')
                    if len(parts) == 2:
                        numerator, denominator = parts[0], parts[1]
                        if numerator in sample['final_state'] and denominator in sample['final_state']:
                            perturbed_row[var] = sample['final_state'][numerator] / (sample['final_state'][denominator] + 1e-10)
            
            synth_samples.append(perturbed_row)
        
        # ==== Combine with 80-20 weighting (obs is more reliable!) ====
        combined = obs_samples + synth_samples
        n_obs = len(obs_samples)
        n_synth = len(synth_samples)
        
        # 80/20 weighting: obs data is ground truth (high quality)
        # Synthetic shows intervention effects but may be contaminated by misspecification
        weights = np.array([0.8/n_obs]*n_obs + [0.2/n_synth]*n_synth)
        weights *= len(combined)  # Normalize to sum to N
        
        if verbose:
            print(f"      Training samples: {n_obs} obs + {n_synth} synth = {len(combined)} total")
            if include_synthetic_baselines:
                print(f"        (includes {len(stable_synthetic)} synthetic baselines at I=1.0)")
            print(f"      Weighting: 80% obs / 20% synth (obs is ground truth)")
        
        # Train mechanisms for ALL variables (original + R)
        scm_mechanisms = {}
        training_scores = {}
        deterministic_info = {}
        
        for var in all_variables:
            is_original = var in variables
            
            # Check if ORIGINAL variable is matched to R equation (→ deterministic)
            matched_eq = var_to_equation.get(var) if is_original else None
            is_deterministic = is_original and matched_eq and matched_eq.startswith('f_R_')
            
            if is_deterministic:
                # Parse R equation to get deterministic formula
                R_info = self._parse_R_equation(matched_eq, selected_equations[matched_eq], var)
                
                if R_info:
                    deterministic_info[var] = R_info
                    scm_mechanisms[var] = None  # No model to train
                    training_scores[var] = 1.0
                    
                    if verbose:
                        formula = f"{R_info['R_var']} * {R_info['other_var']}" if R_info['is_numerator'] else f"{R_info['other_var']} / {R_info['R_var']}"
                        print(f"      {var}: deterministic = {formula}")
                    continue
            
            # Get parents for this variable (original or R)
            parents_causal = structure.get(var, [])
            
            # Build feature set
            features = ['k_in']
            
            for parent in parents_causal:
                if parent in all_variables:  # Original or R variable
                    features.append(parent)
                elif parent.startswith('I_'):
                    features.append(parent)
                elif parent == self.config['exogenous']:
                    pass  # k_in already added
            
            # Add own intervention if ORIGINAL variable and not in parents
            if is_original and var in inhibitions:
                I_var = inhibitions[var]
                if I_var not in features:
                    features.append(I_var)
            
            # Handle constant variables (no features except k_in, only for original)
            if is_original and len(features) == 1 and features[0] == 'k_in':
                if not train_saturated:
                    # DEFAULT: Skip training, keep baseline
                    scm_mechanisms[var] = None
                    training_scores[var] = 1.0
                    if verbose:
                        print(f"      {var}: constant (saturated, skipped)")
                    continue
                else:
                    # EXPERIMENTAL: Train from k_in only
                    # DummyRegressor will likely dominate, predicting mean
                    if verbose:
                        print(f"      {var}: constant but training from k_in (experimental)")
                    # Fall through to training code below
            
            # Train model for this variable (R or original)
            try:
                X_train = np.array([[s[f] for f in features] for s in combined])
                y_train = np.array([s[var] for s in combined])
                
                # Train super-learner WITH SAMPLE WEIGHTS (now CV is fixed too!)
                learner = CausalSuperLearner()
                learner.fit(X_train, y_train, sample_weight=weights)
                
                # Compute training R²
                y_pred = learner.predict(X_train)
                ss_tot = np.sum((y_train - y_train.mean())**2)
                ss_res = np.sum((y_train - y_pred)**2)
                r2 = 1 - ss_res / ss_tot
                
                scm_mechanisms[var] = {
                    'learner': learner,
                    'features': features,
                    'parents': parents_causal,
                }
                training_scores[var] = r2
                
            except KeyError as e:
                if verbose:
                    print(f"      {var}: FAILED - missing feature {e}")
                scm_mechanisms[var] = None
                training_scores[var] = 0.0
        
        if verbose:
            # Only compute mean R² for successfully trained models
            successful_scores = [r2 for var, r2 in training_scores.items() 
                               if var not in deterministic_info and scm_mechanisms.get(var) is not None]
            mean_r2 = np.mean(successful_scores) if successful_scores else 0.0
            n_trained = len([m for v, m in scm_mechanisms.items() if m is not None])
            print(f"      Mean R²: {mean_r2:.4f}")
            print(f"      Trained {n_trained} mechanisms")
        
        return scm_mechanisms, training_scores, structure, deterministic_info
    
    def run(self, gt_formulation, candidate_formulation, gt_data, candidate_data,
            include_synthetic_baselines=False,
            stability_threshold=50.0,
            train_saturated=False,
            seed=42, force_refresh=False, verbose=True):
        """
        Complete Module 3 pipeline
        
        NEW PARAMETERS:
            include_synthetic_baselines: Add initial_state from synthetic as obs-like samples
            stability_threshold: Max value for stability filtering
            train_saturated: If True, train models for saturated/constant variables.
                           Use to test if learning average behavior helps with
                           structural mismatches (e.g., MA→MM where MM saturates
                           variables that have feedback in MA).
        """
        cache_file = self.cache_dir / f'scm_gt{gt_formulation}_cand{candidate_formulation}.pkl'
        
        if not force_refresh and cache_file.exists():
            if verbose:
                print(f"[Module 3] Loading cached SCM (GT={gt_formulation}, Cand={candidate_formulation})")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        if verbose:
            print(f"\n[Module 3] Training SCM: GT={gt_formulation.upper()}, Cand={candidate_formulation.upper()}")
        
        scm_mechanisms, training_scores, structure, deterministic_info = self.train_scm(
            candidate_formulation,
            gt_data['obs_data'],
            candidate_data['synthetic_data'],
            include_synthetic_baselines=include_synthetic_baselines,
            stability_threshold=stability_threshold,
            train_saturated=train_saturated,
            verbose=verbose
        )
        
        result = {
            'gt_formulation': gt_formulation,
            'candidate_formulation': candidate_formulation,
            'scm_mechanisms': scm_mechanisms,
            'training_scores': training_scores,
            'structure': structure,
            'deterministic_info': deterministic_info,
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        if verbose:
            print(f"    ✓ Cached to {cache_file.name}")
        
        return result


if __name__ == "__main__":
    pass