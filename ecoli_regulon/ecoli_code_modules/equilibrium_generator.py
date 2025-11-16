"""
MODULE 2: Equilibrium Equation Generator

Converts PKN to symbolic equilibrium equations.
Supports inhibition_type parameter for different intervention models.

Responsibility: Convert PKN → sympy equilibrium equations
NO knowledge of: ODE simulation, SCM, testing
"""

import sympy as sp


class EquilibriumGenerator:
    def __init__(self, pkn):
        """
        Initialize equilibrium equation generator
        
        Args:
            pkn: PKN dictionary containing:
                - variables: list of variable names
                - exogenous: name of exogenous input
                - inhibitions: dict mapping variables to inhibition names
                - edges: list of (source, target, type) tuples
                - no_degradation: list of variables without degradation
                - inhibition_type: 'production' (default) or 'activity'
        """
        self.pkn = pkn
        self.vars = pkn['variables']
        self.k_in = pkn['exogenous']
        self.inhib = pkn['inhibitions']
        self.no_degradation = set(pkn.get('no_degradation', []))
        self.saturation = pkn.get('saturation', {'mm': [], 'hill': []})
        self.inhibition_type = pkn.get('inhibition_type', 'production')
    
    def generate_equations(self, form, with_inhibitions=False):
        """
        Generate equilibrium equations
        
        Args:
            form: Kinetic form - 'mass_action', 'mm', or 'hill'
            with_inhibitions: Include I_X intervention terms
        
        Returns:
            dict: {eq_name: sympy_expression}
        """
        k = sp.Symbol(self.k_in, real=True, positive=True)
        
        if with_inhibitions:
            I = {v: sp.Symbol(self.inhib[v], real=True, positive=True) 
                 for v in self.inhib}
        else:
            I = {v: 1 for v in self.inhib}
        
        sat = set(self.saturation.get(form, []))
        
        # Build production/consumption for each variable
        production = {v: 0 for v in self.vars}
        consumption = {v: 0 for v in self.vars}
        
        for edge in self.pkn['edges']:
            src, tgt = edge[0], edge[1] if len(edge) >= 2 else None
            etype = edge[2] if len(edge) >= 3 else 'activate'
            
            if tgt not in self.vars:
                continue
            
            # Inhibition logic depends on type
            if self.inhibition_type == 'production':
                # Production inhibition: I_target multiplies terms INTO target
                I_factor = I.get(tgt, 1)
            else:  # 'activity'
                # Activity inhibition: I_source multiplies terms FROM source
                I_factor = I.get(src, 1) if src in self.vars else 1
            
            # Build rate term
            if src == self.k_in:
                param_name = f'k_{src}_{tgt}' if form == 'mass_action' else f'V_{src}_{tgt}'
                rate = I_factor * k * sp.Symbol(param_name, real=True, positive=True)
                
            elif src in self.vars:
                src_sym = sp.Symbol(src, real=True, positive=True)
                
                if form == 'mass_action':
                    param = sp.Symbol(f'k_{src}_{tgt}', real=True, positive=True)
                    rate = I_factor * param * src_sym
                
                elif form == 'mm':
                    V = sp.Symbol(f'V_{src}_{tgt}', real=True, positive=True)
                    if src in sat:
                        rate = I_factor * V
                    else:
                        Km = sp.Symbol('Km', real=True, positive=True)
                        rate = I_factor * V * src_sym / (Km + src_sym)
                
                elif form == 'hill':
                    V = sp.Symbol(f'V_{src}_{tgt}', real=True, positive=True)
                    if src in sat:
                        rate = I_factor * V
                    else:
                        K = sp.Symbol('K', real=True, positive=True)
                        n = sp.Symbol('n', real=True, positive=True)
                        rate = I_factor * V * (src_sym**n) / (K**n + src_sym**n)
            else:
                continue
            
            # Apply based on edge type
            if etype == 'activate':
                production[tgt] += rate
            elif etype == 'convert':
                production[tgt] += rate
                if src in self.vars:
                    consumption[src] += rate
        
        # Build equilibrium equations
        equations = {}
        for var in self.vars:
            var_sym = sp.Symbol(var, real=True, positive=True)
            
            if var in self.no_degradation:
                equations[f'f_{var}'] = production[var] - consumption[var]
            else:
                deg_param = sp.Symbol(f'k_d{var}', real=True, positive=True)
                equations[f'f_{var}'] = production[var] - consumption[var] - deg_param * var_sym
        
        return equations