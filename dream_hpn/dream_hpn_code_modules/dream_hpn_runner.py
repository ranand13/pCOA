"""HPN-DREAM MCF7 breast cancer signaling - Investigation C

UPDATES:
- Generates equations with inhibitions (with_inhibitions=True)
- Fixed inhibition detection via symbol checking
- Full PKN with feedback loops (8 edges)
- Optimized I_AKT = 0.95, centered synthetic data
- Tests on good ligands only (EGF, Serum, Insulin)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

from dream_data_loader import DreamDataLoader
from equilibrium_generator import EquilibriumGenerator
from pcoa_structure_discovery import PCOADiscovery
from scm_model import SCMModel


class DreamHPNRunner:
    
    def __init__(self, 
                 data_dir='dream_hpn_data',
                 cache_dir='dream_hpn_cache',
                 aggregate_sites=True,
                 joint_iter_ma=30,
                 joint_iter_nonlinear=20,
                 verbose=True):
        
        self.data_dir = data_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.aggregate_sites = aggregate_sites
        self.joint_iter_ma = joint_iter_ma
        self.joint_iter_nonlinear = joint_iter_nonlinear
        self.verbose = verbose
        
        self.proteins = None
        self.graph = None
        self.pkn_minimal = None
        self.train_obs = None
        self.test_cases = None
        self.fitted_odes = {}
        self.synthetic_data = {}
        self.pcoa_structures = {}
        self.scms = {}
        self.results_df = None
        
    def _log(self, msg, header=False):
        if self.verbose:
            if header:
                print(f"\n{'='*80}\n{msg}\n{'='*80}")
            else:
                print(msg)
    
    def _get_cache_path(self, name):
        return self.cache_dir / f"{name}.pkl"
    
    def _save_cache(self, name, obj):
        with open(self._get_cache_path(name), 'wb') as f:
            pickle.dump(obj, f)
    
    def _load_cache(self, name):
        path = self._get_cache_path(name)
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_data(self):
        self._log("LOADING DATA", header=True)
        
        loader = DreamDataLoader(self.data_dir, aggregate_sites=self.aggregate_sites)
        self.train_obs, self.test_cases, self.proteins = loader.load_data()
        gene_pkn = loader.build_literature_pkn()
        
        self.graph, ordered = loader.map_to_protein_network(gene_pkn, self.proteins)
        inhibitions = loader.get_inhibition_targets(self.proteins)
        
        self.pkn_minimal = {
            'variables': self.proteins, 
            'exogenous': 'input_signal',
            'inhibitions': inhibitions,
            'inhibition_type': 'activity', 
            'edges': [(p, c) for c, ps in self.graph.items() for p in ps if p in self.proteins],
            'no_degradation': []
        }
        
        self._log(f"Training: {len(self.train_obs)} DMSO samples")
        self._log(f"Testing: {len(self.test_cases)} perturbations")
        self._log(f"Proteins: {len(self.proteins)} {self.proteins}")
        self._log(f"Edges: {len(self.pkn_minimal['edges'])}")
    
    def fit_odes(self, force_recompute=False):
        self._log("FITTING ODE MODELS", header=True)
        
        cache_key = f"fitted_odes_agg{self.aggregate_sites}"
        if not force_recompute:
            cached = self._load_cache(cache_key)
            if cached is not None:
                self.fitted_odes = cached
                self._log("Loaded from cache")
                return
        
        train_subset = self.train_obs[self.proteins + ['input_signal']].dropna()
        
        for form in ['mass_action', 'mm', 'hill']:
            max_iter = self.joint_iter_ma if form == 'mass_action' else self.joint_iter_nonlinear
            
            if form == 'mass_action':
                self.fitted_odes[form] = self._fit_ma_joint(train_subset, max_iter)
            else:
                self.fitted_odes[form] = self._fit_nonlinear_joint(train_subset, form, max_iter)
        
        self._save_cache(cache_key, self.fitted_odes)
    
    def _fit_ma_joint(self, train_data, max_iter):
        n = len(self.proteins)
        p_idx = {p: i for i, p in enumerate(self.proteins)}
        
        edges = [(p_idx[parent], p_idx[child])
                 for child, parents in self.graph.items() if child in p_idx
                 for parent in parents if parent in p_idx]
        
        if self.verbose:
            print(f"  MA (joint, {n} vars, {len(edges)} edges, {len(train_data)} samples)")
        
        total_evals = max_iter * 10
        pbar = tqdm_auto(total=total_evals, desc="    Fitting MA", leave=False, 
                    disable=not self.verbose, unit="eval")
        
        def loss(params):
            pbar.update(1)
            
            k_in, k_out = params[:n], params[n:2*n]
            A = np.zeros((n, n))
            for idx, (j, i) in enumerate(edges):
                A[j, i] = params[2*n + idx]
            
            error, count = 0, 0
            for _, row in train_data.iterrows():
                try:
                    A_scaled = A / k_out[:, np.newaxis]
                    system = np.eye(n) - A_scaled.T + np.eye(n)*1e-6
                    X_ss = np.linalg.solve(system, k_in/k_out * row['input_signal'])
                    
                    if np.all(X_ss > 0) and np.all(X_ss < 50):
                        for i, p in enumerate(self.proteins):
                            if pd.notna(row[p]) and row[p] > 0:
                                error += ((X_ss[i] - row[p]) / row[p])**2
                                count += 1
                    else:
                        error += 100
                except:
                    error += 100
            
            return error / max(count, 1) if count > 0 else 1e6
        
        bounds = [(0.1, 2.0)]*n + [(0.05, 0.5)]*n + [(0.0, 2.0)]*len(edges)
        result = differential_evolution(loss, bounds, maxiter=max_iter,
                                        popsize=10, seed=42, workers=1, atol=0.01)
        
        pbar.close()
        
        A = np.zeros((n, n))
        for idx, (j, i) in enumerate(edges):
            A[j, i] = result.x[2*n + idx]
        
        if self.verbose:
            print(f"    → Loss: {result.fun:.3f}")
        
        return {'k_in': result.x[:n], 'k_out': result.x[n:2*n], 'k_react': A}
    
    def _fit_nonlinear_joint(self, train_data, form, max_iter):
        n = len(self.proteins)
        p_idx = {p: i for i, p in enumerate(self.proteins)}
        
        edges = [(p_idx[parent], p_idx[child])
                 for child, parents in self.graph.items() if child in p_idx
                 for parent in parents if parent in p_idx]
        
        n_samples = min(50, len(train_data))
        train_subset = train_data.sample(n_samples, random_state=42)
        
        if self.verbose:
            print(f"  {form.upper()} (joint, {len(train_subset)}/{len(train_data)} samples)")
        
        total_evals = max_iter * 8
        pbar = tqdm_auto(total=total_evals, desc=f"    Fitting {form.upper()}", leave=False,
                    disable=not self.verbose, unit="eval")
        
        def loss(params):
            pbar.update(1)
            
            k_in, k_out = params[:n], params[n:2*n]
            A = np.zeros((n, n))
            for idx, (j, i) in enumerate(edges):
                A[j, i] = params[2*n + idx]
            
            error, count = 0, 0
            
            for _, row in train_subset.iterrows():
                y0 = np.ones(n) * 0.5
                
                def ode(t, y):
                    dy = np.zeros(n)
                    for i, prot in enumerate(self.proteins):
                        prod = k_in[i] * row['input_signal']
                        
                        for parent in self.graph.get(prot, []):
                            if parent in p_idx:
                                j = p_idx[parent]
                                
                                act = 1.0
                                if 'MEK' in parent:
                                    act = row.get('I_MEK', 1.0)
                                elif 'AKT' in parent:
                                    act = row.get('I_AKT', 1.0)
                                
                                if form == 'mm':
                                    prod += A[j,i] * act * y[j] / (0.5 + y[j])
                                else:
                                    prod += A[j,i] * act * (y[j]**2) / (0.5**2 + y[j]**2)
                        
                        dy[i] = prod - k_out[i] * y[i]
                    return dy
                
                try:
                    sol = solve_ivp(ode, [0, 100], y0, method='BDF',
                                    rtol=1e-3, atol=1e-6)
                    
                    if sol.success and np.all(sol.y[:,-1] > 1e-5) and np.all(sol.y[:,-1] < 50):
                        for i, p in enumerate(self.proteins):
                            if pd.notna(row[p]) and row[p] > 0:
                                error += ((sol.y[i,-1] - row[p]) / row[p])**2
                                count += 1
                    else:
                        error += 50
                except:
                    error += 50
            
            return error / max(count, 1) if count > 0 else 1e6
        
        bounds = [(0.1, 1.5)]*n + [(0.1, 0.4)]*n + [(0.1, 1.5)]*len(edges)
        result = differential_evolution(loss, bounds, maxiter=max_iter,
                                        popsize=8, seed=42, workers=1, atol=0.02)
        
        pbar.close()
        
        A = np.zeros((n, n))
        for idx, (j, i) in enumerate(edges):
            A[j, i] = result.x[2*n + idx]
        
        if self.verbose:
            print(f"    → Loss: {result.fun:.3f}")
        
        return {'k_in': result.x[:n], 'k_out': result.x[n:2*n], 'k_react': A}
    
    def generate_synthetic(self, force_recompute=False):
        self._log("GENERATING SYNTHETIC DATA", header=True)
        
        cache_key = f"synthetic_data_agg{self.aggregate_sites}"
        if not force_recompute:
            cached = self._load_cache(cache_key)
            if cached is not None:
                self.synthetic_data = cached
                self._log("Loaded from cache")
                return
        
        for form in ['mass_action', 'mm', 'hill']:
            self._log(f"Generating {form}...")
            self.synthetic_data[form] = self._generate_synth(form)
        
        self._save_cache(cache_key, self.synthetic_data)
    
    def _generate_synth(self, form):
        n = len(self.proteins)
        p_idx = {p: i for i, p in enumerate(self.proteins)}
        
        samples = []
        inputs = [0.0, 0.25, 0.5, 0.75, 1.0]
        I_AKT_vals = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        I_MEK_vals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
        
        for inp in inputs:
            for I_A in I_AKT_vals:
                for I_M in I_MEK_vals:
                    ss = self._solve_steady_state(p_idx, form, inp, {'I_AKT': I_A, 'I_MEK': I_M})
                    if ss:
                        samples.append({**ss, 'input_signal': inp, 'I_AKT': I_A, 'I_MEK': I_M})
        
        if self.verbose:
            coverage = len(samples) / (len(inputs) * len(I_AKT_vals) * len(I_MEK_vals))
            print(f"  {form.upper()}: {len(samples)} samples ({coverage*100:.0f}% coverage)")
        
        return pd.DataFrame(samples)
    
    def _solve_steady_state(self, p_idx, form, input_sig, interventions):
        n = len(self.proteins)
        y0 = np.ones(n) * 0.5
        params = self.fitted_odes[form]
        k_in, k_out, A = params['k_in'], params['k_out'], params['k_react']
        
        def ode(t, y):
            dy = np.zeros(n)
            for i, prot in enumerate(self.proteins):
                prod = k_in[i] * input_sig
                
                for parent in self.graph.get(prot, []):
                    if parent in p_idx:
                        j = p_idx[parent]
                        
                        act = 1.0
                        if 'MEK' in parent:
                            act = interventions.get('I_MEK', 1.0)
                        elif 'AKT' in parent:
                            act = interventions.get('I_AKT', 1.0)
                        
                        if form == 'mass_action':
                            prod += A[j,i] * act * y[j]
                        elif form == 'mm':
                            prod += A[j,i] * act * y[j] / (0.5 + y[j])
                        else:
                            prod += A[j,i] * act * (y[j]**2) / (0.5**2 + y[j]**2)
                
                dy[i] = prod - k_out[i] * y[i]
            return dy
        
        try:
            sol = solve_ivp(ode, [0, 150], y0, method='BDF', rtol=1e-4, atol=1e-7)
            if sol.success and np.all(sol.y[:,-1] > 1e-5) and np.all(sol.y[:,-1] < 50):
                return dict(zip(self.proteins, sol.y[:,-1]))
        except:
            pass
        return None
    
    def _predict_ode(self, baseline, input_sig, interventions, form):
        n = len(self.proteins)
        p_idx = {p: i for i, p in enumerate(self.proteins)}
        y0 = np.array([baseline.get(p, 0.5) for p in self.proteins])
        k_in, k_out, A = self.fitted_odes[form]['k_in'], self.fitted_odes[form]['k_out'], self.fitted_odes[form]['k_react']
        
        def ode(t, y):
            dy = np.zeros(n)
            for i, prot in enumerate(self.proteins):
                prod = k_in[i] * input_sig
                for parent in self.graph.get(prot, []):
                    if parent in p_idx:
                        j = p_idx[parent]
                        act = 1.0
                        if 'MEK' in parent:
                            act = interventions.get('I_MEK', 1.0)
                        elif 'AKT' in parent:
                            act = interventions.get('I_AKT', 1.0)
                        
                        if form == 'mass_action':
                            prod += A[j,i] * act * y[j]
                        elif form == 'mm':
                            prod += A[j,i] * act * y[j] / (0.5 + y[j])
                        else:
                            prod += A[j,i] * act * (y[j]**2) / (0.5**2 + y[j]**2)
                dy[i] = prod - k_out[i] * y[i]
            return dy
        
        try:
            sol = solve_ivp(ode, [0, 150], y0, method='BDF', rtol=1e-4, atol=1e-7)
            if sol.success and np.all(sol.y[:,-1] > 1e-5) and np.all(sol.y[:,-1] < 50):
                return dict(zip(self.proteins, sol.y[:,-1]))
        except:
            pass
        return None
    
    def discover_structures(self, force_recompute=False):
        self._log("PCOA STRUCTURE DISCOVERY", header=True)
        
        cache_key = f"pcoa_structures_agg{self.aggregate_sites}"
        if not force_recompute:
            cached = self._load_cache(cache_key)
            if cached is not None:
                self.pcoa_structures = cached
                self._log("Loaded from cache")
                return
        
        eq_gen = EquilibriumGenerator(self.pkn_minimal)
        pcoa_disc = PCOADiscovery(self.pkn_minimal)
        
        for form in ['mass_action', 'mm', 'hill']:
            # NOTE: Using False for now until inhibition logic is fixed
            eqs = eq_gen.generate_equations(form, with_inhibitions=True)
            structure, topo = pcoa_disc.discover_structure(eqs, debug=True)
            self.pcoa_structures[form] = (structure, topo)
            
            if self.verbose:
                self._log(f"\n{form}:")
                for var in self.proteins:
                    parents = structure.get(var, [])
                    parent_str = ', '.join(parents) if parents else 'exogenous'
                    self._log(f"  {var} ← [{parent_str}]")
        
        self._save_cache(cache_key, self.pcoa_structures)
    
    def train_scms(self, force_recompute=False):
        self._log("TRAINING SCM MODELS", header=True)
        
        cache_key = f"trained_scms_agg{self.aggregate_sites}"
        if not force_recompute:
            cached = self._load_cache(cache_key)
            if cached is not None:
                self.scms = cached
                self._log("Loaded from cache")
                return
        
        inhibitions = self.pkn_minimal['inhibitions']
        obs_with_I = self.train_obs[self.proteins + ['input_signal', 'I_AKT', 'I_MEK']].copy()
        
        for form in ['mass_action', 'mm', 'hill']:
            self._log(f"Training {form}...")
            structure, topo = self.pcoa_structures[form]
            
            scm = SCMModel(structure, topo, 'input_signal', inhibitions, self.proteins,
                          include_interactions=False)
            scm.train(obs_with_I.to_dict('records'), 
                     self.synthetic_data[form].to_dict('records'),
                     weighting_scheme='stratified_trust', 
                     verbose=self.verbose)
            self.scms[form] = scm
        
        self._save_cache(cache_key, self.scms)
    
    def test_predictions(self, save_protein_errors=False):
        self._log(f"TESTING PREDICTIONS ({len(self.test_cases)} cases)", header=True)
        
        all_predictions = []
        
        for test_idx, tc in enumerate(tqdm(self.test_cases, disable=not self.verbose, desc="Testing")):
            baseline = tc['baseline']
            true = tc['true_perturbed']
            baseline_with_signal = {**baseline, 'input_signal': tc['input_signal']}
            interventions_opt = {'I_AKT': tc['I_AKT'], 'I_MEK': tc['I_MEK']}
            
            measured = [p for p in self.proteins if p in true]
            if len(measured) < 3:
                continue
            
            for form in ['mass_action', 'mm', 'hill']:
                ode_pred = self._predict_ode(baseline, tc['input_signal'], interventions_opt, form)
                scm_pred = self.scms[form].predict(baseline_with_signal, interventions_opt)
                
                if ode_pred:
                    for p in measured:
                        if p in ode_pred and p in scm_pred and true[p] > 0 and p in baseline:
                            baseline_val = baseline[p]
                            true_val = true[p]
                            ode_val = ode_pred[p]
                            scm_val = scm_pred[p]
                            
                            baseline_log2 = abs(np.log2(baseline_val + 1e-10) - np.log2(true_val + 1e-10))
                            ode_log2 = abs(np.log2(ode_val + 1e-10) - np.log2(true_val + 1e-10))
                            scm_log2 = abs(np.log2(scm_val + 1e-10) - np.log2(true_val + 1e-10))
                            
                            all_predictions.append({
                                'formulation': form,
                                'test_case_id': test_idx,
                                'ligand': tc['ligand'],
                                'target': tc.get('target'),
                                'protein': p,
                                'baseline_log2': baseline_log2,
                                'ode_log2': ode_log2,
                                'scm_log2': scm_log2,
                                'baseline_fold': 2**baseline_log2,
                                'ode_fold': 2**ode_log2,
                                'scm_fold': 2**scm_log2
                            })
        
        self.results_df = pd.DataFrame(all_predictions)
        
        if save_protein_errors:
            results_path = self.cache_dir / 'protein_errors.csv'
            self.results_df.to_csv(results_path, index=False)
            self._log(f"\nSaved: {results_path}")
    
    def print_results(self):
        if self.results_df is None or len(self.results_df) == 0:
            self._log("No results")
            return
        
        self._log("AKT INHIBITOR PREDICTION ERRORS", header=True)
        
        n_per_form = len(self.results_df) // 3
        self._log(f"N = {n_per_form} measurements per formulation")
        
        self._log(f"\nLog₂ Errors (mean ± SE):")
        self._log(f"{'Form':<12} {'Baseline':>18} {'ODE':>18} {'SCM':>18}")
        self._log("-" * 68)
        
        for form in ['mass_action', 'mm', 'hill']:
            form_df = self.results_df[self.results_df['formulation'] == form]
            n = len(form_df)
            
            b = form_df['baseline_log2'].mean()
            b_se = form_df['baseline_log2'].std() / np.sqrt(n)
            o = form_df['ode_log2'].mean()
            o_se = form_df['ode_log2'].std() / np.sqrt(n)
            s = form_df['scm_log2'].mean()
            s_se = form_df['scm_log2'].std() / np.sqrt(n)
            
            self._log(f"{form.upper():<12} {b:>7.3f} ± {b_se:<7.3f} {o:>7.3f} ± {o_se:<7.3f} {s:>7.3f} ± {s_se:<7.3f}")
        
        self._log(f"\nFold Errors (mean ± SE):")
        self._log(f"{'Form':<12} {'Baseline':>18} {'ODE':>18} {'SCM':>18}")
        self._log("-" * 68)
        
        for form in ['mass_action', 'mm', 'hill']:
            form_df = self.results_df[self.results_df['formulation'] == form]
            n = len(form_df)
            
            b = form_df['baseline_fold'].mean()
            b_se = form_df['baseline_fold'].std() / np.sqrt(n)
            o = form_df['ode_fold'].mean()
            o_se = form_df['ode_fold'].std() / np.sqrt(n)
            s = form_df['scm_fold'].mean()
            s_se = form_df['scm_fold'].std() / np.sqrt(n)
            
            self._log(f"{form.upper():<12} {b:>7.2f} ± {b_se:<7.2f}× {o:>7.2f} ± {o_se:<7.2f}× {s:>7.2f} ± {s_se:<7.2f}×")
    
    def run(self, force_recompute=False, save_protein_errors=False):
        self._log("HPN-DREAM MCF7 BREAST CANCER SIGNALING", header=True)
        self._log(f"Cache: {self.cache_dir}")
        
        self.load_data()
        self.fit_odes(force_recompute=force_recompute)
        self.generate_synthetic(force_recompute=force_recompute)
        self.discover_structures(force_recompute=force_recompute)
        self.train_scms(force_recompute=force_recompute)
        self.test_predictions(save_protein_errors=save_protein_errors)
        self.print_results()
        
        self._log("\nComplete!", header=True)
        
        return self.results_df