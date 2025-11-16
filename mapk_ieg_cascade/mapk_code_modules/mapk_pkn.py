"""
MAPK→Immediate Early Gene Cascade

CRITICAL FIXES:
1. Stable feedback loop (k_shared_deg = 0.030 for MA, 0.60 for MM/Hill)
2. Configurable tolerances (rtol, atol, integration_time)

Loop gain = 0.58 < 1.0 (stable)
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def get_cascade_config():
    """Configuration for MAPK→IEG cascade"""
    return {
        'variables': ['ERK', 'MEK', 'cFos', 'Fra1', 'cJun', 'CyclinD', 'Bcl2'],
        'exogenous': 'GF',
        'inhibitions': {
            'ERK': 'I_ERK',
            'MEK': 'I_MEK',
            'CyclinD': 'I_CyclinD',
        },
        
        'initial_conditions': {
            'mass_action': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            'mm': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            'hill': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        },
        
        'parameters': {
            'mass_action': {
                'k_GF_MEK': 0.030,
                'k_deg_MEK': 0.022,
                'k_GF_ERK': 0.025,
                'k_MEK_ERK': 0.024,
                'k_deg_ERK': 0.022,
                'k_GF_cFos': 0.020,
                'k_MEKERK_cFos': 0.028,
                'k_cJun_cFos': 0.024,
                'k_cFos_Fra1': 0.026,
                'k_Fra1_cJun': 0.025,
                'k_shared_deg': 0.030,
                'k_cJun_CyclinD': 0.024,
                'k_deg_CyclinD': 0.020,
                'k_cFos_Bcl2': 0.026,
                'k_deg_Bcl2': 0.020,
            },
            
            'mm': {
                'V_GF_MEK': 0.60,
                'k_deg_MEK': 0.45,
                'V_GF_ERK': 0.50,
                'V_MEK_ERK': 0.40,
                'k_deg_ERK': 0.45,
                'V_GF_cFos': 0.40,
                'V_MEKERK_cFos': 0.55,
                'V_cJun_cFos': 0.48,
                'V_cFos_Fra1': 0.52,
                'V_Fra1_cJun': 0.50,
                'k_shared_deg': 0.60,
                'V_cJun_CyclinD': 0.48,
                'k_deg_CyclinD': 0.40,
                'V_cFos_Bcl2': 0.52,
                'k_deg_Bcl2': 0.40,
                'Km': 0.2,
            },
            
            'hill': {
                'V_GF_MEK': 0.60,
                'k_deg_MEK': 0.45,
                'V_GF_ERK': 0.50,
                'V_MEK_ERK': 0.40,
                'k_deg_ERK': 0.45,
                'V_GF_cFos': 0.40,
                'V_MEKERK_cFos': 0.55,
                'V_cJun_cFos': 0.48,
                'V_cFos_Fra1': 0.52,
                'V_Fra1_cJun': 0.50,
                'k_shared_deg': 0.60,
                'V_cJun_CyclinD': 0.48,
                'k_deg_CyclinD': 0.40,
                'V_cFos_Bcl2': 0.52,
                'k_deg_Bcl2': 0.40,
                'K': 0.2,
                'n': 2.0,
            }
        },
        
        'saturation': {
            'mm': ['Fra1', 'cJun'],
            'hill': ['Fra1', 'cJun'],
        }
    }


def build_ode_solver(form, config, rtol=1e-3, atol=1e-6):
    """Build ODE solver with configurable tolerances"""
    params = config['parameters'][form]
    variables = config['variables']
    y0 = config['initial_conditions'][form]
    inhibitions = config['inhibitions']
    
    def solver(gf_val, integration_time=2000, **I):
        """Solve ODE to steady state"""
        I_vals = {inhibitions[v]: I.get(inhibitions[v], 1.0) for v in inhibitions}
        
        if form == 'mass_action':
            def ode_system(t, y):
                ERK, MEK, cFos, Fra1, cJun, CyclinD, Bcl2 = [max(x, 1e-10) for x in y]
                
                I_ERK = I_vals.get('I_ERK', 1.0)
                I_MEK = I_vals.get('I_MEK', 1.0)
                I_CyclinD = I_vals.get('I_CyclinD', 1.0)
                
                dERK = (I_ERK * gf_val * params['k_GF_ERK']
                       + I_ERK * params['k_MEK_ERK'] * MEK
                       - params['k_deg_ERK'] * ERK)
                
                dMEK = (I_MEK * gf_val * params['k_GF_MEK']
                       - params['k_deg_MEK'] * MEK)
                
                dcFos = (gf_val * params['k_GF_cFos']
                        + params['k_MEKERK_cFos'] * MEK * ERK
                        + params['k_cJun_cFos'] * cJun
                        - params['k_shared_deg'] * cFos)
                
                dFra1 = (params['k_cFos_Fra1'] * cFos
                        - params['k_shared_deg'] * Fra1)
                
                dcJun = (params['k_Fra1_cJun'] * Fra1
                        - params['k_shared_deg'] * cJun)
                
                dCyclinD = (I_CyclinD * params['k_cJun_CyclinD'] * cJun
                           - params['k_deg_CyclinD'] * CyclinD)
                
                dBcl2 = (params['k_cFos_Bcl2'] * cFos
                        - params['k_deg_Bcl2'] * Bcl2)
                
                return [dERK, dMEK, dcFos, dFra1, dcJun, dCyclinD, dBcl2]
        
        elif form == 'mm':
            def ode_system(t, y):
                ERK, MEK, cFos, Fra1, cJun, CyclinD, Bcl2 = [max(x, 1e-10) for x in y]
                Km = params['Km']
                sat = set(config['saturation'].get('mm', []))
                
                I_ERK = I_vals.get('I_ERK', 1.0)
                I_MEK = I_vals.get('I_MEK', 1.0)
                I_CyclinD = I_vals.get('I_CyclinD', 1.0)
                
                dERK = (I_ERK * gf_val * params['V_GF_ERK']
                       + I_ERK * params['V_MEK_ERK'] * MEK / (Km + MEK)
                       - params['k_deg_ERK'] * ERK)
                
                dMEK = (I_MEK * gf_val * params['V_GF_MEK']
                       - params['k_deg_MEK'] * MEK)
                
                dcFos = (gf_val * params['V_GF_cFos']
                        + (params['V_MEKERK_cFos'] if 'cFos' in sat else 
                           params['V_MEKERK_cFos'] * MEK / (Km + MEK) * ERK / (Km + ERK))
                        + (params['V_cJun_cFos'] if 'cFos' in sat else params['V_cJun_cFos'] * cJun / (Km + cJun))
                        - params['k_shared_deg'] * cFos)
                
                dFra1 = ((params['V_cFos_Fra1'] if 'Fra1' in sat else params['V_cFos_Fra1'] * cFos / (Km + cFos))
                        - params['k_shared_deg'] * Fra1)
                
                dcJun = ((params['V_Fra1_cJun'] if 'cJun' in sat else params['V_Fra1_cJun'] * Fra1 / (Km + Fra1))
                        - params['k_shared_deg'] * cJun)
                
                dCyclinD = (I_CyclinD * (params['V_cJun_CyclinD'] if 'CyclinD' in sat else params['V_cJun_CyclinD'] * cJun / (Km + cJun))
                           - params['k_deg_CyclinD'] * CyclinD)
                
                dBcl2 = ((params['V_cFos_Bcl2'] if 'Bcl2' in sat else params['V_cFos_Bcl2'] * cFos / (Km + cFos))
                        - params['k_deg_Bcl2'] * Bcl2)
                
                return [dERK, dMEK, dcFos, dFra1, dcJun, dCyclinD, dBcl2]
        
        elif form == 'hill':
            def ode_system(t, y):
                ERK, MEK, cFos, Fra1, cJun, CyclinD, Bcl2 = [max(x, 1e-10) for x in y]
                K, n = params['K'], params['n']
                sat = set(config['saturation'].get('hill', []))
                
                I_ERK = I_vals.get('I_ERK', 1.0)
                I_MEK = I_vals.get('I_MEK', 1.0)
                I_CyclinD = I_vals.get('I_CyclinD', 1.0)
                
                dERK = (I_ERK * gf_val * params['V_GF_ERK']
                       + I_ERK * params['V_MEK_ERK'] * (MEK**n) / (K**n + MEK**n)
                       - params['k_deg_ERK'] * ERK)
                
                dMEK = (I_MEK * gf_val * params['V_GF_MEK']
                       - params['k_deg_MEK'] * MEK)
                
                dcFos = (gf_val * params['V_GF_cFos']
                        + (params['V_MEKERK_cFos'] if 'cFos' in sat else 
                           params['V_MEKERK_cFos'] * (MEK**n) / (K**n + MEK**n) * (ERK**n) / (K**n + ERK**n))
                        + (params['V_cJun_cFos'] if 'cFos' in sat else params['V_cJun_cFos'] * (cJun**n) / (K**n + cJun**n))
                        - params['k_shared_deg'] * cFos)
                
                dFra1 = ((params['V_cFos_Fra1'] if 'Fra1' in sat else params['V_cFos_Fra1'] * (cFos**n) / (K**n + cFos**n))
                        - params['k_shared_deg'] * Fra1)
                
                dcJun = ((params['V_Fra1_cJun'] if 'cJun' in sat else params['V_Fra1_cJun'] * (Fra1**n) / (K**n + Fra1**n))
                        - params['k_shared_deg'] * cJun)
                
                dCyclinD = (I_CyclinD * (params['V_cJun_CyclinD'] if 'CyclinD' in sat else params['V_cJun_CyclinD'] * (cJun**n) / (K**n + cJun**n))
                           - params['k_deg_CyclinD'] * CyclinD)
                
                dBcl2 = ((params['V_cFos_Bcl2'] if 'Bcl2' in sat else params['V_cFos_Bcl2'] * (cFos**n) / (K**n + cFos**n))
                      - params['k_deg_Bcl2'] * Bcl2)
                
                return [dERK, dMEK, dcFos, dFra1, dcJun, dCyclinD, dBcl2]
        
        else:
            raise ValueError(f"Unknown formulation: {form}")
        
        try:
            sol = solve_ivp(
                ode_system, 
                [0, integration_time], 
                y0,
                method='LSODA', 
                rtol=rtol,
                atol=atol,
                max_step=5.0
            )
            
            if sol.success and np.all(sol.y[:, -1] > 1e-3) and np.all(sol.y[:, -1] < 50):
                dydt_final = ode_system(sol.t[-1], sol.y[:, -1])
                max_deriv = max(abs(d) for d in dydt_final)
                
                if max_deriv < 1e-3:
                    return dict(zip(variables, sol.y[:, -1]))
        except:
            pass
        
        return None
    
    return solver


def get_parameter_info(formulation):
    """Get parameter names and ABSOLUTE bounds for optimization"""
    config = get_cascade_config()
    default_params = config['parameters'][formulation]
    
    if formulation == 'mass_action':
        mult_low, mult_high = 0.85, 1.15
    else:
        mult_low, mult_high = 0.7, 1.3
    
    param_info = []
    for name in default_params.keys():
        default_val = default_params[name]
        param_info.append({
            'name': name,
            'bounds': (mult_low * default_val, mult_high * default_val)
        })
    
    return param_info


def get_equilibrium_equations(form, config):
    """Equilibrium equations - NO INHIBITIONS"""
    ERK = sp.Symbol('ERK', real=True, positive=True)
    MEK = sp.Symbol('MEK', real=True, positive=True)
    cFos = sp.Symbol('cFos', real=True, positive=True)
    Fra1 = sp.Symbol('Fra1', real=True, positive=True)
    cJun = sp.Symbol('cJun', real=True, positive=True)
    CyclinD = sp.Symbol('CyclinD', real=True, positive=True)
    Bcl2 = sp.Symbol('Bcl2', real=True, positive=True)
    GF = sp.Symbol('GF', real=True, positive=True)
    
    params_dict = config['parameters'][form]
    params = {name: sp.Symbol(name, real=True, positive=True)
             for name in params_dict.keys()}
    
    if form == 'mass_action':
        equations = {
            'f_ERK': (GF * params['k_GF_ERK']
                     + MEK * params['k_MEK_ERK']
                     - ERK * params['k_deg_ERK']),
            
            'f_MEK': (GF * params['k_GF_MEK']
                     - MEK * params['k_deg_MEK']),
            
            'f_cFos': (GF * params['k_GF_cFos']
                      + MEK * ERK * params['k_MEKERK_cFos']
                      + cJun * params['k_cJun_cFos']
                      - cFos * params['k_shared_deg']),
            
            'f_Fra1': (cFos * params['k_cFos_Fra1']
                      - Fra1 * params['k_shared_deg']),
            
            'f_cJun': (Fra1 * params['k_Fra1_cJun']
                      - cJun * params['k_shared_deg']),
            
            'f_CyclinD': (cJun * params['k_cJun_CyclinD']
                         - CyclinD * params['k_deg_CyclinD']),
            
            'f_Bcl2': (cFos * params['k_cFos_Bcl2']
                      - Bcl2 * params['k_deg_Bcl2']),
        }
    
    elif form == 'mm':
        Km = params['Km']
        sat = set(config['saturation'].get('mm', []))
        
        equations = {
            'f_ERK': (GF * params['V_GF_ERK']
                     + params['V_MEK_ERK'] * MEK / (Km + MEK)
                     - ERK * params['k_deg_ERK']),
            
            'f_MEK': (GF * params['V_GF_MEK']
                     - MEK * params['k_deg_MEK']),
            
            'f_cFos': (GF * params['V_GF_cFos']
                      + (params['V_MEKERK_cFos'] if 'cFos' in sat else 
                        params['V_MEKERK_cFos'] * MEK / (Km + MEK) * ERK / (Km + ERK))
                      + (params['V_cJun_cFos'] if 'cFos' in sat else params['V_cJun_cFos'] * cJun / (Km + cJun))
                      - cFos * params['k_shared_deg']),
            
            'f_Fra1': ((params['V_cFos_Fra1'] if 'Fra1' in sat else params['V_cFos_Fra1'] * cFos / (Km + cFos))
                      - Fra1 * params['k_shared_deg']),
            
            'f_cJun': ((params['V_Fra1_cJun'] if 'cJun' in sat else params['V_Fra1_cJun'] * Fra1 / (Km + Fra1))
                      - cJun * params['k_shared_deg']),
            
            'f_CyclinD': ((params['V_cJun_CyclinD'] if 'CyclinD' in sat else params['V_cJun_CyclinD'] * cJun / (Km + cJun))
                         - CyclinD * params['k_deg_CyclinD']),
            
            'f_Bcl2': ((params['V_cFos_Bcl2'] if 'Bcl2' in sat else params['V_cFos_Bcl2'] * cFos / (Km + cFos))
                      - Bcl2 * params['k_deg_Bcl2']),
        }
    
    elif form == 'hill':
        K, n = params['K'], params['n']
        sat = set(config['saturation'].get('hill', []))
        
        equations = {
            'f_ERK': (GF * params['V_GF_ERK']
                     + params['V_MEK_ERK'] * (MEK**n) / (K**n + MEK**n)
                     - ERK * params['k_deg_ERK']),
            
            'f_MEK': (GF * params['V_GF_MEK']
                     - MEK * params['k_deg_MEK']),
            
            'f_cFos': (GF * params['V_GF_cFos']
                      + (params['V_MEKERK_cFos'] if 'cFos' in sat else 
                        params['V_MEKERK_cFos'] * (MEK**n) / (K**n + MEK**n) * (ERK**n) / (K**n + ERK**n))
                      + (params['V_cJun_cFos'] if 'cFos' in sat else params['V_cJun_cFos'] * (cJun**n) / (K**n + cJun**n))
                      - cFos * params['k_shared_deg']),
            
            'f_Fra1': ((params['V_cFos_Fra1'] if 'Fra1' in sat else params['V_cFos_Fra1'] * (cFos**n) / (K**n + cFos**n))
                      - Fra1 * params['k_shared_deg']),
            
            'f_cJun': ((params['V_Fra1_cJun'] if 'cJun' in sat else params['V_Fra1_cJun'] * (Fra1**n) / (K**n + Fra1**n))
                      - cJun * params['k_shared_deg']),
            
            'f_CyclinD': ((params['V_cJun_CyclinD'] if 'CyclinD' in sat else params['V_cJun_CyclinD'] * (cJun**n) / (K**n + cJun**n))
                         - CyclinD * params['k_deg_CyclinD']),
            
            'f_Bcl2': ((params['V_cFos_Bcl2'] if 'Bcl2' in sat else params['V_cFos_Bcl2'] * (cFos**n) / (K**n + cFos**n))
                      - Bcl2 * params['k_deg_Bcl2']),
        }
    
    else:
        raise ValueError(f"Unknown formulation: {form}")
    
    return equations