"""
MODULE 1: ODE Simulator - CORRECTED & COMPLETE

UPDATED: Added inhibition_type parameter
- 'production' (default): I_X on terms INTO X
- 'activity': I_X on terms FROM X
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm


class ODESimulator:
    def __init__(self, pkn):
        self.pkn = pkn
        self.vars = pkn['variables']
        self.k_in = pkn['exogenous']
        self.inhib = pkn['inhibitions']
        self.no_degradation = set(pkn.get('no_degradation', []))
        self.saturation = pkn.get('saturation', {'mm': [], 'hill': []})
        self.inhibition_type = pkn.get('inhibition_type', 'production')  # NEW
    
    def build_solver(self, form, params):
        """Build ODE solver for given formulation"""
        def solver(k_val, **I):
            I_vals = {self.inhib[v]: I.get(self.inhib[v], 1.0) for v in self.inhib}
            y0 = [0.1] * len(self.vars)
            
            def ode(t, y):
                state = dict(zip(self.vars, [max(x, 1e-10) for x in y]))
                dy = []
                
                for var in self.vars:
                    prod, cons = 0, 0
                    
                    for edge in self.pkn['edges']:
                        src, tgt = edge[0], edge[1] if len(edge) >= 2 else None
                        etype = edge[2] if len(edge) >= 3 else 'activate'
                        
                        if tgt != var:
                            continue
                        
                        # UPDATED: Handle both inhibition types
                        if self.inhibition_type == 'production':
                            I_var = I_vals.get(self.inhib.get(tgt), 1.0)
                        else:
                            I_var = I_vals.get(self.inhib.get(src), 1.0) if src in self.vars else 1.0
                        
                        if src == self.k_in:
                            key = f'k_{src}_{tgt}' if form == 'mass_action' else f'V_{src}_{tgt}'
                            prod += I_var * k_val * params.get(key, 0.5)
                        elif src in self.vars:
                            if form == 'mass_action':
                                prod += I_var * params.get(f'k_{src}_{tgt}', 0.5) * state[src]
                            elif form == 'mm':
                                V = params.get(f'V_{src}_{tgt}', 0.5)
                                Km = params.get('Km', 1.0)
                                prod += I_var * V * state[src] / (Km + state[src])
                            elif form == 'hill':
                                V = params.get(f'V_{src}_{tgt}', 0.5)
                                K, n = params.get('K', 1.0), params.get('n', 2.0)
                                prod += I_var * V * (state[src]**n) / (K**n + state[src]**n)
                        
                        if etype == 'convert' and src == var and tgt in self.vars:
                            if self.inhibition_type == 'production':
                                I_tgt = I_vals.get(self.inhib.get(tgt), 1.0)
                            else:
                                I_tgt = I_vals.get(self.inhib.get(src), 1.0)
                            
                            if form == 'mass_action':
                                cons += I_tgt * params.get(f'k_{var}_{tgt}', 0.5) * state[var]
                            elif form == 'mm':
                                V = params.get(f'V_{var}_{tgt}', 0.5)
                                Km = params.get('Km', 1.0)
                                cons += I_tgt * V * state[var] / (Km + state[var])
                            elif form == 'hill':
                                V = params.get(f'V_{var}_{tgt}', 0.5)
                                K, n = params.get('K', 1.0), params.get('n', 2.0)
                                cons += I_tgt * V * (state[var]**n) / (K**n + state[var]**n)
                    
                    if var not in self.no_degradation:
                        dy.append(prod - cons - params.get(f'k_d{var}', 0.3) * state[var])
                    else:
                        dy.append(prod - cons)
                
                return dy
            
            try:
                sol = solve_ivp(ode, [0, 100], y0, method='LSODA',
                               rtol=1e-4, atol=1e-7, max_step=1.0)
                if sol.success and np.all(sol.y[:, -1] > 1e-3):
                    return dict(zip(self.vars, sol.y[:, -1]))
            except:
                pass
            return None
        
        return solver
    
    def generate_obs_data(self, form, base_params, N=150, noise=0.05, seed=None):
        """Generate observational data - dose-response design"""
        if seed: np.random.seed(seed)
        
        solver = self.build_solver(form, base_params)
        cells = []
        
        doses = np.linspace(0.75, 1.25, 5)
        n_per_dose = N // len(doses)
        
        pbar = tqdm(total=N, desc=f"  Obs data ({form})", leave=True)
        
        for dose in doses:
            for _ in range(n_per_dose):
                ss = solver(dose)
                if ss and max(ss.values()) < 100:
                    cells.append({
                        **{v: ss[v] * (1 + np.random.normal(0, noise)) for v in ss},
                        self.k_in: dose
                    })
                    pbar.update(1)
        
        pbar.close()
        return cells
    
    def generate_interv_data(self, form, fitted_params, N_per_combo=10, seed=None):
        """Generate interventional data - GRID DESIGN"""
        if seed: np.random.seed(seed)
        
        solver = self.build_solver(form, fitted_params)
        data = []
        
        k_in_grid = np.linspace(0.75, 1.25, 5)
        I_grid = [0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
        
        total = len(k_in_grid) * len(self.inhib) * len(I_grid) * N_per_combo
        pbar = tqdm(total=total, desc=f"  Interv ({form})", leave=False)
        
        for k in k_in_grid:
            for inhib_var in self.inhib.keys():
                for I_val in I_grid:
                    for _ in range(N_per_combo):
                        I_base = {self.inhib[v]: 1.0 for v in self.inhib}
                        baseline = solver(k, **I_base)
                        
                        if not baseline or max(baseline.values()) > 100:
                            pbar.update(1)
                            continue
                        
                        I_interv = I_base.copy()
                        I_interv[self.inhib[inhib_var]] = I_val
                        intervened = solver(k, **I_interv)
                        
                        if intervened and max(intervened.values()) < 100:
                            data.append({**intervened, **I_interv, self.k_in: k})
                            pbar.update(1)
        
        pbar.close()
        return data
    
    def fit_to_obs(self, form, obs_data, seed=42, verbose=False, fast_mode=True):
        """Fit using dose information"""
        pnames = self._get_param_names(form)
        fit_cells = self._stratified_sample(obs_data, seed)
        
        def loss(p):
            params = self._params_dict(form, pnames, p)
            solver = self.build_solver(form, params)
            err, count = 0, 0
            
            for cell in fit_cells:
                k_val = cell.get(self.k_in, 1.0)
                pred = solver(k_val=k_val)
                
                if pred and max(pred.values()) < 500:
                    try:
                        cell_err = sum(
                            min(((pred[v] - cell[v]) / (cell[v] + 1e-6))**2, 100)
                            for v in self.vars
                        )
                        err += cell_err
                        count += 1
                    except:
                        return 1e6
                else:
                    return 1e6
            
            return err / max(count, 1) if count > 0 else 1e6
        
        bounds = self._get_bounds(form, pnames)
        
        result = differential_evolution(
            loss, bounds,
            maxiter=15 if fast_mode else 40,
            popsize=8 if fast_mode else 10,
            seed=seed, workers=1,
            atol=0.02, tol=0.02
        )
        
        fitted = self._params_dict(form, pnames, result.x)
        print(f"    {form.upper()} fitted: loss={result.fun:.4f}")
        return fitted
    
    def predict_from_baseline(self, form, params, baseline_state, k_val, **I):
        """Predict from baseline"""
        y0 = [baseline_state[v] for v in self.vars]
        I_vals = {self.inhib[v]: I.get(self.inhib[v], 1.0) for v in self.inhib}
        
        def ode(t, y):
            state = dict(zip(self.vars, [max(x, 1e-10) for x in y]))
            dy = []
            
            for var in self.vars:
                prod, cons = 0, 0
                
                for edge in self.pkn['edges']:
                    src, tgt = edge[0], edge[1] if len(edge) >= 2 else None
                    etype = edge[2] if len(edge) >= 3 else 'activate'
                    
                    if tgt != var:
                        continue
                    
                    # UPDATED: Handle both inhibition types
                    if self.inhibition_type == 'production':
                        I_var = I_vals.get(self.inhib.get(tgt), 1.0)
                    else:
                        I_var = I_vals.get(self.inhib.get(src), 1.0) if src in self.vars else 1.0
                    
                    if src == self.k_in:
                        key = f'k_{src}_{tgt}' if form == 'mass_action' else f'V_{src}_{tgt}'
                        prod += I_var * k_val * params.get(key, 0.5)
                    elif src in self.vars:
                        if form == 'mass_action':
                            prod += I_var * params.get(f'k_{src}_{tgt}', 0.5) * state[src]
                        elif form == 'mm':
                            V = params.get(f'V_{src}_{tgt}', 0.5)
                            Km = params.get('Km', 1.0)
                            prod += I_var * V * state[src] / (Km + state[src])
                        elif form == 'hill':
                            V = params.get(f'V_{src}_{tgt}', 0.5)
                            K, n = params.get('K', 1.0), params.get('n', 2.0)
                            prod += I_var * V * (state[src]**n) / (K**n + state[src]**n)
                    
                    if etype == 'convert' and src == var and tgt in self.vars:
                        if self.inhibition_type == 'production':
                            I_tgt = I_vals.get(self.inhib.get(tgt), 1.0)
                        else:
                            I_tgt = I_vals.get(self.inhib.get(src), 1.0)
                        
                        if form == 'mass_action':
                            cons += I_tgt * params.get(f'k_{var}_{tgt}', 0.5) * state[var]
                        elif form == 'mm':
                            V = params.get(f'V_{var}_{tgt}', 0.5)
                            Km = params.get('Km', 1.0)
                            cons += I_tgt * V * state[var] / (Km + state[var])
                        elif form == 'hill':
                            V = params.get(f'V_{var}_{tgt}', 0.5)
                            K, n = params.get('K', 1.0), params.get('n', 2.0)
                            cons += I_tgt * V * (state[var]**n) / (K**n + state[var]**n)
                
                if var not in self.no_degradation:
                    dy.append(prod - cons - params.get(f'k_d{var}', 0.3) * state[var])
                else:
                    dy.append(prod - cons)
            
            return dy
        
        try:
            sol = solve_ivp(ode, [0, 100], y0, method='LSODA',
                           rtol=1e-4, atol=1e-7, max_step=1.0)
            if sol.success and np.all(sol.y[:, -1] > 1e-3):
                return dict(zip(self.vars, sol.y[:, -1]))
        except:
            pass
        return None
    
        return solver
    
    def generate_obs_data(self, form, base_params, N=150, noise=0.05, seed=None):
        """Generate obs data - dose-response design"""
        if seed: np.random.seed(seed)
        
        solver = self.build_solver(form, base_params)
        cells = []
        
        doses = np.linspace(0.75, 1.25, 5)
        n_per_dose = N // len(doses)
        
        pbar = tqdm(total=N, desc=f"  Obs data ({form})", leave=True)
        
        for dose in doses:
            for _ in range(n_per_dose):
                ss = solver(dose)
                if ss and max(ss.values()) < 100:
                    cells.append({
                        **{v: ss[v] * (1 + np.random.normal(0, noise)) for v in ss},
                        self.k_in: dose
                    })
                    pbar.update(1)
        
        pbar.close()
        return cells
    
    def generate_interv_data(self, form, fitted_params, N_per_combo=10, seed=None):
        """Generate interv data - GRID DESIGN"""
        if seed: np.random.seed(seed)
        
        solver = self.build_solver(form, fitted_params)
        data = []
        
        k_in_grid = np.linspace(0.75, 1.25, 5)
        I_grid = [0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
        
        total = len(k_in_grid) * len(self.inhib) * len(I_grid) * N_per_combo
        pbar = tqdm(total=total, desc=f"  Interv ({form})", leave=False)
        
        for k in k_in_grid:
            for inhib_var in self.inhib.keys():
                for I_val in I_grid:
                    for _ in range(N_per_combo):
                        I_base = {self.inhib[v]: 1.0 for v in self.inhib}
                        baseline = solver(k, **I_base)
                        
                        if not baseline or max(baseline.values()) > 100:
                            pbar.update(1)
                            continue
                        
                        I_interv = I_base.copy()
                        I_interv[self.inhib[inhib_var]] = I_val
                        intervened = solver(k, **I_interv)
                        
                        if intervened and max(intervened.values()) < 100:
                            data.append({**intervened, **I_interv, self.k_in: k})
                            pbar.update(1)
        
        pbar.close()
        return data
    
    def fit_to_obs(self, form, obs_data, seed=42, verbose=False, fast_mode=True):
        """Fit using dose information"""
        pnames = self._get_param_names(form)
        fit_cells = self._stratified_sample(obs_data, seed)
        
        def loss(p):
            params = self._params_dict(form, pnames, p)
            solver = self.build_solver(form, params)
            err, count = 0, 0
            
            for cell in fit_cells:
                k_val = cell.get(self.k_in, 1.0)
                pred = solver(k_val=k_val)
                
                if pred and max(pred.values()) < 500:
                    try:
                        cell_err = sum(
                            min(((pred[v] - cell[v]) / (cell[v] + 1e-6))**2, 100)
                            for v in self.vars
                        )
                        err += cell_err
                        count += 1
                    except:
                        return 1e6
                else:
                    return 1e6
            
            return err / max(count, 1) if count > 0 else 1e6
        
        bounds = self._get_bounds(form, pnames)
        
        result = differential_evolution(
            loss, bounds,
            maxiter=15 if fast_mode else 40,
            popsize=8 if fast_mode else 10,
            seed=seed, workers=1,
            atol=0.02, tol=0.02
        )
        
        fitted = self._params_dict(form, pnames, result.x)
        print(f"    {form.upper()} fitted: loss={result.fun:.4f}")
        return fitted
    
    def predict_from_baseline(self, form, params, baseline_state, k_val, **I):
        """Predict from baseline"""
        y0 = [baseline_state[v] for v in self.vars]
        I_vals = {self.inhib[v]: I.get(self.inhib[v], 1.0) for v in self.inhib}
        
        def ode(t, y):
            state = dict(zip(self.vars, [max(x, 1e-10) for x in y]))
            dy = []
            
            for var in self.vars:
                prod, cons = 0, 0
                
                for edge in self.pkn['edges']:
                    src, tgt = edge[0], edge[1] if len(edge) >= 2 else None
                    etype = edge[2] if len(edge) >= 3 else 'activate'
                    
                    if tgt != var:
                        continue
                    
                    # UPDATED: Handle both inhibition types
                    if self.inhibition_type == 'production':
                        I_var = I_vals.get(self.inhib.get(tgt), 1.0)
                    else:
                        I_var = I_vals.get(self.inhib.get(src), 1.0) if src in self.vars else 1.0
                    
                    if src == self.k_in:
                        key = f'k_{src}_{tgt}' if form == 'mass_action' else f'V_{src}_{tgt}'
                        prod += I_var * k_val * params.get(key, 0.5)
                    elif src in self.vars:
                        if form == 'mass_action':
                            prod += I_var * params.get(f'k_{src}_{tgt}', 0.5) * state[src]
                        elif form == 'mm':
                            V = params.get(f'V_{src}_{tgt}', 0.5)
                            Km = params.get('Km', 1.0)
                            prod += I_var * V * state[src] / (Km + state[src])
                        elif form == 'hill':
                            V = params.get(f'V_{src}_{tgt}', 0.5)
                            K, n = params.get('K', 1.0), params.get('n', 2.0)
                            prod += I_var * V * (state[src]**n) / (K**n + state[src]**n)
                    
                    if etype == 'convert' and src == var and tgt in self.vars:
                        if self.inhibition_type == 'production':
                            I_tgt = I_vals.get(self.inhib.get(tgt), 1.0)
                        else:
                            I_tgt = I_vals.get(self.inhib.get(src), 1.0)
                        
                        if form == 'mass_action':
                            cons += I_tgt * params.get(f'k_{var}_{tgt}', 0.5) * state[var]
                        elif form == 'mm':
                            V = params.get(f'V_{var}_{tgt}', 0.5)
                            Km = params.get('Km', 1.0)
                            cons += I_tgt * V * state[var] / (Km + state[var])
                        elif form == 'hill':
                            V = params.get(f'V_{var}_{tgt}', 0.5)
                            K, n = params.get('K', 1.0), params.get('n', 2.0)
                            cons += I_tgt * V * (state[var]**n) / (K**n + state[var]**n)
                
                if var not in self.no_degradation:
                    dy.append(prod - cons - params.get(f'k_d{var}', 0.3) * state[var])
                else:
                    dy.append(prod - cons)
            
            return dy
        
        try:
            sol = solve_ivp(ode, [0, 100], y0, method='LSODA',
                           rtol=1e-4, atol=1e-7, max_step=1.0)
            if sol.success and np.all(sol.y[:, -1] > 1e-3):
                return dict(zip(self.vars, sol.y[:, -1]))
        except:
            pass
        return None
    
    def _get_param_names(self, form):
        pnames = []
        for edge in self.pkn['edges']:
            src, tgt = edge[0], edge[1]
            pnames.append(f'k_{src}_{tgt}' if form == 'mass_action' else f'V_{src}_{tgt}')
        for v in self.vars:
            if v not in self.no_degradation:
                pnames.append(f'k_d{v}')
        return list(dict.fromkeys(pnames))
    
    def _params_dict(self, form, pnames, p):
        params = dict(zip(pnames, p))
        if form in ['mm', 'hill']:
            params['Km'] = 1.0
            if form == 'hill':
                params['K'] = 1.0
                params['n'] = 2.0
        return params
    
    def _get_bounds(self, form, pnames):
        if form == 'mass_action':
            return [(0.005, 0.15) if not pn.startswith('k_d') else (0.005, 0.05)
                    for pn in pnames]
        else:
            return [(0.1, 3.0) if not pn.startswith('k_d') else (0.01, 0.8)
                    for pn in pnames]
    
    def _stratified_sample(self, obs_data, seed):
        sorted_obs = sorted(obs_data, key=lambda x: x[self.vars[0]])
        np.random.seed(seed)
        
        n_strata = 3
        strata_size = len(sorted_obs) // n_strata
        sampled_idx = []
        
        for i in range(n_strata):
            start = i * strata_size
            end = start + strata_size if i < n_strata - 1 else len(sorted_obs)
            if end > start:
                n_sample = min(5, end - start)
                sampled_idx.extend(np.random.choice(range(start, end), n_sample, replace=False))
        
        return [sorted_obs[i] for i in sampled_idx]