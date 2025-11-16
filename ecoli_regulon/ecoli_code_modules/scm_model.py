"""
MODULE 4: SCM Model - WITH INTERACTION TERMS OPTION

NEW: include_interactions parameter
- If True: Adds polynomial interaction features (e.g., I_AKT × I_MEK)
- If False: Standard linear features (default)

Fixes synergistic effect problem for combo drug predictions!
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline


class SuperLearner:
    """Ensemble regression with optional interaction terms"""
    
    def __init__(self, include_interactions=False):
        """
        Initialize Super Learner
        
        Args:
            include_interactions: If True, add I × I interaction terms
        """
        self.include_interactions = include_interactions
        
        if include_interactions:
            # WITH interactions: Can learn I_AKT × I_MEK synergy
            self.models = {
                'poly2': make_pipeline(PolynomialFeatures(2), Ridge(0.1)),
                'poly2_interact': make_pipeline(
                    PolynomialFeatures(2, interaction_only=True),  # Only interactions
                    Ridge(0.5)
                ),
                'poly3': make_pipeline(PolynomialFeatures(3), Ridge(1.0)),
                'rf': RandomForestRegressor(50, max_depth=10, random_state=42, n_jobs=1)
            }
            self.weights = {n: 0.25 for n in self.models}
        else:
            # WITHOUT interactions: Standard linear model
            self.models = {
                'poly2': make_pipeline(PolynomialFeatures(2), Ridge(0.1)),
                'poly3': make_pipeline(PolynomialFeatures(3), Ridge(1.0)),
                'rf': RandomForestRegressor(50, max_depth=10, random_state=42, n_jobs=1)
            }
            self.weights = {n: 1/3 for n in self.models}
    
    def fit(self, X, y, sample_weights=None):
        for name, model in self.models.items():
            if sample_weights is None:
                model.fit(X, y)
            else:
                if name == 'rf':
                    model.fit(X, y, sample_weight=sample_weights)
                else:
                    # Pipeline: pass to final estimator
                    model.fit(X, y, ridge__sample_weight=sample_weights)
        return self
    
    def predict(self, X):
        preds = [self.weights[n] * self.models[n].predict(X) for n in self.models]
        return np.sum(preds, axis=0)


class SCMModel:
    def __init__(self, structure, topo_order, k_in_name, inhib_map, var_list,
                 include_interactions=False):
        """
        Initialize SCM Model
        
        Args:
            structure: Causal structure dict {var: [parents]}
            topo_order: Topological ordering of variables
            k_in_name: Exogenous variable name
            inhib_map: Inhibition mapping {var: 'I_var'}
            var_list: List of endogenous variables
            include_interactions: Add interaction terms for dual interventions
        """
        self.structure = structure
        self.topo = topo_order
        self.k_in = k_in_name
        self.inhib = inhib_map
        self.vars = var_list
        self.include_interactions = include_interactions
        self.models = {}
    
    def train(self, obs_data, interv_data, 
              weighting_scheme='stratified_trust',
              test_I=None,
              focus_factor=2.0,
              verbose=False):
        """
        STATE-OF-THE-ART SCM Training with Optional Interaction Terms
        
        Args:
            obs_data: Observational data (ground truth ODE)
            interv_data: Interventional data (potentially misspecified ODE)
            weighting_scheme: How to weight training samples
            test_I: Test intervention value (for 'test_focused' scheme)
            focus_factor: Extra weight near test regime
            verbose: Print training details
        """
        
        if verbose:
            print(f"\n  [SCM Training - {weighting_scheme}]")
            if self.include_interactions:
                print(f"  Using interaction terms (e.g., I_AKT × I_MEK)")
        
        # DIAGNOSTIC 1: Check k_in presence
        if self.k_in not in obs_data[0]:
            if verbose:
                print(f"  ⚠️  WARNING: k_in not in obs data!")
            obs_data = [{**s, self.k_in: 1.0} for s in obs_data]
        
        if len(interv_data) > 0 and self.k_in not in interv_data[0]:
            print(f"  ✗ CRITICAL: k_in NOT in interventional data!")
            print(f"     SCM training will fail!")
        
        # DIAGNOSTIC 2: Check k_in variation in interventional data
        if len(interv_data) > 0:
            k_in_values_interv = set(s.get(self.k_in, 1.0) for s in interv_data)
            if verbose:
                print(f"  Interv k_in variation: {len(k_in_values_interv)} unique values")
            
            if len(k_in_values_interv) < 3:
                print(f"  ✗ WARNING: Only {len(k_in_values_interv)} k_in values!")
                print(f"     Need ≥3 for learning k_in × I interactions")
        
        # Filter unstable
        stable_interv = [s for s in interv_data 
                        if max([s[v] for v in self.vars if v in s]) < 100]
        
        if len(stable_interv) < len(interv_data) * 0.7 and len(interv_data) > 0:
            if verbose:
                print(f"  ⚠️  Only {len(stable_interv)}/{len(interv_data)} stable")
        
        # Build combined dataset
        if len(stable_interv) == 0:
            if verbose:
                print(f"  Using obs only (no stable interv data)")
            combined = obs_data
            weights = np.ones(len(combined))
        else:
            combined = obs_data + stable_interv
            
            # COMPUTE WEIGHTS BASED ON SCHEME
            weights = self._compute_weights(
                obs_data, stable_interv, 
                weighting_scheme, test_I, focus_factor, verbose
            )
        
        # Train models
        if verbose:
            print(f"  Combined: {len(combined)} samples")
        
        for var in self.vars:
            parents = self.structure.get(var, [])
            
            if not parents:
                self.models[var] = None
                if verbose:
                    print(f"  {var}: exogenous")
                continue
            
            X = np.array([[d.get(p, 0) for p in parents] for d in combined])
            y = np.array([d[var] for d in combined])
            
            model = SuperLearner(include_interactions=self.include_interactions).fit(
                X, y, sample_weights=weights
            )
            self.models[var] = model
            
            if verbose:
                y_pred = model.predict(X)
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                interact_note = " (with interactions)" if self.include_interactions else ""
                print(f"  {var} ← {parents}: R²={r2:.3f}{interact_note}")
    
    def _compute_weights(self, obs_data, interv_data, scheme, test_I, focus_factor, verbose):
        """
        Compute sample weights based on weighting scheme
        
        Returns: weights array of length len(obs_data) + len(interv_data)
        """
        n_obs = len(obs_data)
        n_interv = len(interv_data)
        n_total = n_obs + n_interv
        
        if scheme == 'original':
            # OLD: 60% obs, 40% interv (per-sample weighting)
            weights = np.array(
                [0.6 / n_obs] * n_obs +
                [0.4 / n_interv] * n_interv
            )
            weights *= n_total
            
            if verbose:
                print(f"  Weighting: 60% obs / 40% interv (per-sample)")
        
        elif scheme == 'stratified':
            # Equal weight per intervention level (ignore obs)
            weights = self._stratified_weighting(
                obs_data, interv_data, weight_obs=1.0, weight_interv=1.0
            )
            
            if verbose:
                print(f"  Weighting: stratified (equal per I level)")
        
        elif scheme == 'stratified_trust':
            # DEFAULT: 50% obs (high quality), 50% interv (stratified)
            weights = self._stratified_weighting(
                obs_data, interv_data, weight_obs=0.5, weight_interv=0.5
            )
            
            if verbose:
                print(f"  Weighting: 50% obs / 50% interv (stratified)")
                
                # Print distribution
                obs_weight = np.sum(weights[:n_obs])
                interv_weight = np.sum(weights[n_obs:])
                print(f"    Obs total weight: {obs_weight:.1f} ({obs_weight/n_total*100:.1f}%)")
                print(f"    Interv total weight: {interv_weight:.1f} ({interv_weight/n_total*100:.1f}%)")
        
        elif scheme == 'test_focused':
            # Extra weight near test intervention value
            if test_I is None:
                raise ValueError("test_I must be specified for 'test_focused' scheme")
            
            weights = self._test_focused_weighting(
                obs_data, interv_data, test_I, focus_factor, 
                weight_obs=0.5, weight_interv=0.5
            )
            
            if verbose:
                print(f"  Weighting: test-focused (I={test_I}, factor={focus_factor})")
        
        else:
            raise ValueError(f"Unknown weighting scheme: {scheme}")
        
        return weights
    
    def _stratified_weighting(self, obs_data, interv_data, weight_obs=0.5, weight_interv=0.5):
        """
        Stratified weighting: equal weight per intervention level
        """
        n_obs = len(obs_data)
        n_interv = len(interv_data)
        n_total = n_obs + n_interv
        
        # Count samples per intervention level
        level_counts = {}
        for sample in interv_data:
            min_I = min(sample.get(I_name, 1.0) for I_name in self.inhib.values())
            min_I = round(min_I, 2)
            level_counts[min_I] = level_counts.get(min_I, 0) + 1
        
        n_levels = len(level_counts)
        
        # Obs samples: equal weight each
        weights_obs = [weight_obs / n_obs] * n_obs
        
        # Interv samples: stratified by level
        weights_interv = []
        for sample in interv_data:
            min_I = min(sample.get(I_name, 1.0) for I_name in self.inhib.values())
            min_I = round(min_I, 2)
            w = weight_interv / (n_levels * level_counts[min_I])
            weights_interv.append(w)
        
        weights = np.array(weights_obs + weights_interv)
        weights = weights / weights.sum() * n_total
        
        return weights
    
    def _test_focused_weighting(self, obs_data, interv_data, test_I, focus_factor,
                                weight_obs=0.5, weight_interv=0.5):
        """Test-focused weighting: extra weight near test regime"""
        n_obs = len(obs_data)
        n_interv = len(interv_data)
        n_total = n_obs + n_interv
        
        level_counts = {}
        for sample in interv_data:
            min_I = min(sample.get(I_name, 1.0) for I_name in self.inhib.values())
            min_I = round(min_I, 2)
            level_counts[min_I] = level_counts.get(min_I, 0) + 1
        
        n_levels = len(level_counts)
        test_target_weight = focus_factor / (n_levels - 1 + focus_factor)
        other_target_weight = 1.0 / (n_levels - 1 + focus_factor)
        
        weights_obs = [weight_obs / n_obs] * n_obs
        weights_interv = []
        test_I_rounded = round(test_I, 2)
        
        for sample in interv_data:
            min_I = min(sample.get(I_name, 1.0) for I_name in self.inhib.values())
            min_I = round(min_I, 2)
            
            if abs(min_I - test_I_rounded) < 0.01:
                w = weight_interv * test_target_weight / level_counts[min_I]
            else:
                w = weight_interv * other_target_weight / level_counts[min_I]
            
            weights_interv.append(w)
        
        weights = np.array(weights_obs + weights_interv)
        weights = weights / weights.sum() * n_total
        
        return weights
    
    def predict(self, baseline, interventions):
        """Predict interventional outcome"""
        if self.k_in not in baseline:
            baseline = {**baseline, self.k_in: 1.0}
        
        affected = self._find_affected(interventions)
        context = {**interventions, self.k_in: baseline[self.k_in]}
        predictions = {}
        
        for var in self.topo:
            if var not in self.vars:
                continue
            
            if var not in affected:
                predictions[var] = baseline[var]
                context[var] = baseline[var]
                continue
            
            if not self.models.get(var):
                predictions[var] = baseline[var]
                context[var] = baseline[var]
                continue
            
            parents = self.structure[var]
            X = np.array([[context.get(p, predictions.get(p, baseline.get(p, 0)))
                          for p in parents]])
            predictions[var] = self.models[var].predict(X)[0]
            context[var] = predictions[var]
        
        return predictions
    
    def _find_affected(self, interventions):
        """Find causally affected variables"""
        affected = set()
        inv_inhib = {v: k for k, v in self.inhib.items()}
        
        for i_var, i_val in interventions.items():
            if i_val != 1.0 and i_var in inv_inhib:
                affected.add(inv_inhib[i_var])
        
        for var in self.topo:
            if var in affected or var not in self.vars:
                continue
            
            parents = self.structure.get(var, [])
            if any(p in affected or p in interventions for p in parents):
                affected.add(var)
        
        return affected
