"""
Causal Ordering Analysis - Complete Implementation with Smart Selection

Features:
- Algorithm 1: Cluster-focused THEN full sweep
- Algorithm 2: Cluster-focused THEN full sweep with upstream parameter benefit
- Smart selection: lineage-based dedup, cluster-count tie-breaking
- R equations: automatically included in final set
- Detailed debugging output
"""

import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations

class CausalOrderingAnalyzer:
    def __init__(self, equilibrium_equations, variables, parameters, exogenous=None, name="System"):
        self.name = name
        processed_eqs, conservation_params = self._generate_conservations(equilibrium_equations, variables)
        self.equations = processed_eqs
        self.variables = {v.name: v for v in variables}
        self.parameters = {p.name: p for p in parameters}
        for cp_name in conservation_params:
            if cp_name not in self.parameters:
                self.parameters[cp_name] = sp.Symbol(cp_name, real=True, positive=True)
        self.conservation_params = conservation_params
        self.exogenous = set(exogenous) if exogenous else set()
    
    def _generate_conservations(self, equations, variables):
        eqs = dict(equations)
        var_dict = {v.name: v for v in variables}
        conservation_params = set()
        removed_eqs = set()
        
        eq_list = list(eqs.items())
        for i, (name1, expr1) in enumerate(eq_list):
            if name1 in removed_eqs:
                continue
            for j, (name2, expr2) in enumerate(eq_list):
                if i >= j or name2 in removed_eqs:
                    continue
                if expr1 == 0 or expr2 == 0:
                    continue
                try:
                    ratio = sp.simplify(expr2 / expr1)
                    if ratio.is_Number and ratio != 0:
                        if name1.startswith('f_cons') or name2.startswith('f_cons'):
                            continue
                        
                        var1_name = name1.replace('f_', '')
                        var2_name = name2.replace('f_', '')
                        
                        if var1_name in var_dict and var2_name in var_dict:
                            var1, var2 = var_dict[var1_name], var_dict[var2_name]
                            init1 = sp.Symbol(f"{var1_name}_0", real=True, positive=True)
                            init2 = sp.Symbol(f"{var2_name}_0", real=True, positive=True)
                            conservation_params.add(init1.name)
                            conservation_params.add(init2.name)
                            cons_eq = var2 - ratio*var1 - init2 + ratio*init1
                            cons_name = f"f_cons_{var1_name}_{var2_name}"
                            
                            print(f"  CONSERVATION: {cons_name}")
                            
                            del eqs[name1]
                            removed_eqs.add(name1)
                            eqs[cons_name] = cons_eq
                            break
                except:
                    continue
        return eqs, conservation_params
    
    def _build_bipartite(self, equations, variables):
        G = nx.Graph()
        for var_name in variables:
            G.add_node(f"v_{var_name}", bipartite=0)
        for eq_name in equations:
            G.add_node(eq_name, bipartite=1)
        for eq_name, expr in equations.items():
            for var_name, var_sym in variables.items():
                if var_sym in expr.free_symbols:
                    G.add_edge(f"v_{var_name}", eq_name)
        return G
    
    def causal_ordering(self, equations, variables, debug=False):
        """Causal ordering with detailed matching and dependency output"""
        B = self._build_bipartite(equations, variables)
        left = {n for n, d in B.nodes(data=True) if d.get('bipartite') == 0}
        M = nx.bipartite.maximum_matching(B, top_nodes=left)
        
        if debug:
            print(f"\n[DEBUG] Causal Ordering Details:")
            print(f"  Variable ← Equation (dependencies):")
            for var_name in sorted(variables.keys()):
                var_node = f"v_{var_name}"
                if var_node in M:
                    eq_name = M[var_node]
                    vars_in_eq = sorted([v.name for v in variables.values() 
                                        if v in equations[eq_name].free_symbols])
                    deps = [v for v in vars_in_eq if v != var_name]
                    print(f"    {var_name} ← {eq_name}: depends on {deps}")
                else:
                    print(f"    {var_name} ← UNMATCHED!")
        
        G = nx.DiGraph()
        for node in B.nodes():
            G.add_node(node)
        for u, v in B.edges():
            var = u if u.startswith('v_') else v
            eq = v if u.startswith('v_') else u
            if M.get(var) == eq:
                G.add_edge(eq, var)
            else:
                G.add_edge(var, eq)
        
        sccs = list(nx.strongly_connected_components(G))
        sccs.sort(key=lambda x: (-len(x), sorted(list(x))))
        
        if debug:
            print(f"\n  Strongly Connected Components:")
            for i, scc in enumerate(sccs):
                vars_in = sorted([n.replace('v_', '') for n in scc if n.startswith('v_')])
                if len(vars_in) > 1:
                    print(f"    SCC {i}: {vars_in} (coupled)")
                elif len(vars_in) == 1:
                    print(f"    SCC {i}: {vars_in} (singleton)")
        
        cluster_graph = nx.DiGraph()
        node_to_cluster, seen, cc = {}, set(), 0
        for scc in sccs:
            cluster = set(scc)
            for node in scc:
                if node in M:
                    cluster.add(M[node])
            if frozenset(cluster) in seen:
                continue
            seen.add(frozenset(cluster))
            cname = f"C{cc}"
            cc += 1
            vi = sorted([n.replace('v_', '') for n in cluster if n.startswith('v_')])
            ei = sorted([n for n in cluster if not n.startswith('v_')])
            cluster_graph.add_node(cname, nodes=cluster, variables=vi, equations=ei, exogenous=[])
            for node in cluster:
                node_to_cluster[node] = cname
        
        for u, v in G.edges():
            if u in node_to_cluster and v in node_to_cluster:
                cu, cv = node_to_cluster[u], node_to_cluster[v]
                if cu != cv:
                    cluster_graph.add_edge(cu, cv)
        
        for exo in self.exogenous:
            ec = f"C{cc}"
            cc += 1
            cluster_graph.add_node(ec, nodes={f"exo_{exo.name}"}, variables=[], equations=[], exogenous=[exo.name])
            for eq_name, expr in equations.items():
                if exo in expr.free_symbols and eq_name in node_to_cluster:
                    cluster_graph.add_edge(ec, node_to_cluster[eq_name])
        
        return cluster_graph
    
    def _compute_upstream_params(self, equations, variables, debug=False):
        """Compute upstream parameters for ORIGINAL variables only"""
        graph = self.causal_ordering(equations, variables, debug=debug)
        upstream_params = {}
        
        # Only compute for original variables
        original_var_names = set(self.variables.keys())
        
        for var_name in original_var_names:
            var_cluster = None
            for cname, data in graph.nodes(data=True):
                if var_name in data.get('variables', []):
                    var_cluster = cname
                    break
            
            if var_cluster:
                upstream = nx.ancestors(graph, var_cluster)
                upstream.add(var_cluster)
                params = set()
                for u_cluster in upstream:
                    for eq_name in graph.nodes[u_cluster].get('equations', []):
                        if eq_name in equations:
                            for p in self.parameters.values():
                                # Only count ORIGINAL parameters (not conservation)
                                if p in equations[eq_name].free_symbols and p.name not in self.conservation_params:
                                    params.add(p.name)
                upstream_params[var_name] = len(params)
            else:
                upstream_params[var_name] = 0
        
        return upstream_params
    
    def _compute_upstream_with_forced_R_matching(self, equations, variables, R_eq_name, forced_var_name):
        """Compute upstream params with forced variable ← R equation matching"""
        # Build bipartite graph
        B = self._build_bipartite(equations, variables)
        left = {n for n, d in B.nodes(data=True) if d.get('bipartite') == 0}
        
        # Check if forced matching is valid
        forced_var_node = f"v_{forced_var_name}"
        if forced_var_node not in B.nodes() or R_eq_name not in B.nodes():
            raise ValueError(f"Cannot force {forced_var_name} ← {R_eq_name}")
        
        if not B.has_edge(forced_var_node, R_eq_name):
            raise ValueError(f"No edge between {forced_var_name} and {R_eq_name}")
        
        # Remove the forced pair from graph temporarily
        B_reduced = B.copy()
        B_reduced.remove_node(forced_var_node)
        B_reduced.remove_node(R_eq_name)
        
        # Find matching on remaining graph
        left_reduced = left - {forced_var_node}
        M_reduced = nx.bipartite.maximum_matching(B_reduced, top_nodes=left_reduced)
        
        # Add forced matching back
        M = dict(M_reduced)
        M[forced_var_node] = R_eq_name
        M[R_eq_name] = forced_var_node
        
        # Now build causal graph using this specific matching
        G = nx.DiGraph()
        for node in B.nodes():
            G.add_node(node)
        for u, v in B.edges():
            var = u if u.startswith('v_') else v
            eq = v if u.startswith('v_') else u
            if M.get(var) == eq:
                G.add_edge(eq, var)
            else:
                G.add_edge(var, eq)
        
        # Build cluster graph (same as causal_ordering)
        sccs = list(nx.strongly_connected_components(G))
        sccs.sort(key=lambda x: (-len(x), sorted(list(x))))
        
        cluster_graph = nx.DiGraph()
        node_to_cluster, seen, cc = {}, set(), 0
        for scc in sccs:
            cluster = set(scc)
            for node in scc:
                if node in M:
                    cluster.add(M[node])
            if frozenset(cluster) in seen:
                continue
            seen.add(frozenset(cluster))
            cname = f"C{cc}"
            cc += 1
            vi = sorted([n.replace('v_', '') for n in cluster if n.startswith('v_')])
            ei = sorted([n for n in cluster if not n.startswith('v_')])
            cluster_graph.add_node(cname, nodes=cluster, variables=vi, equations=ei, exogenous=[])
            for node in cluster:
                node_to_cluster[node] = cname
        
        for u, v in G.edges():
            if u in node_to_cluster and v in node_to_cluster:
                cu, cv = node_to_cluster[u], node_to_cluster[v]
                if cu != cv:
                    cluster_graph.add_edge(cu, cv)
        
        for exo in self.exogenous:
            ec = f"C{cc}"
            cc += 1
            cluster_graph.add_node(ec, nodes={f"exo_{exo.name}"}, variables=[], equations=[], exogenous=[exo.name])
            for eq_name, expr in equations.items():
                if exo in expr.free_symbols and eq_name in node_to_cluster:
                    cluster_graph.add_edge(ec, node_to_cluster[eq_name])
        
        # Compute upstream params using this cluster graph
        upstream_params = {}
        original_var_names = set(self.variables.keys())
        
        for var_name in original_var_names:
            var_cluster = None
            for cname, data in cluster_graph.nodes(data=True):
                if var_name in data.get('variables', []):
                    var_cluster = cname
                    break
            
            if var_cluster:
                upstream = nx.ancestors(cluster_graph, var_cluster)
                upstream.add(var_cluster)
                params = set()
                for u_cluster in upstream:
                    for eq_name in cluster_graph.nodes[u_cluster].get('equations', []):
                        if eq_name in equations:
                            for p in self.parameters.values():
                                if p in equations[eq_name].free_symbols and p.name not in self.conservation_params:
                                    params.add(p.name)
                upstream_params[var_name] = len(params)
            else:
                upstream_params[var_name] = 0
        
        return sum(upstream_params.values())
    
    def _normalize_equation(self, expr):
        if expr == 0:
            return 0
        coeffs = expr.as_coefficients_dict()
        for term in sorted(coeffs.keys(), key=str):
            if term != 1 and coeffs[term] != 0:
                return expr / coeffs[term]
        return expr
    
    def _is_scalar_multiple(self, expr1, expr2, norm_cache):
        if expr1 == 0 or expr2 == 0:
            return expr1 == expr2
        if expr1 not in norm_cache:
            norm_cache[expr1] = self._normalize_equation(expr1)
        if expr2 not in norm_cache:
            norm_cache[expr2] = self._normalize_equation(expr2)
        return sp.expand(norm_cache[expr1] - norm_cache[expr2]) == 0
    
    def _detect_problematic_clusters(self, equations, variables, debug=False):
        """Find clusters with >1 variable that need breaking"""
        cluster_graph = self.causal_ordering(equations, variables, debug=False)
        
        problematic = []
        for cluster_name, data in cluster_graph.nodes(data=True):
            vars_in_cluster = data.get('variables', [])
            eqs_in_cluster = data.get('equations', [])
            
            if len(vars_in_cluster) > 1:
                problematic.append({
                    'name': cluster_name,
                    'variables': set(vars_in_cluster),
                    'equations': set(eqs_in_cluster)
                })
        
        if debug and problematic:
            print(f"  Problematic clusters: {len(problematic)}")
            for c in problematic:
                print(f"    {c['name']}: vars={sorted(c['variables'])}")
        
        return problematic
    
    def algorithm1_generate_equations(self, debug=False, max_iterations=None):
        """Algorithm 1: Cluster-focused THEN full sweep"""
        import time
        eqs = dict(self.equations)
        lineage = {e: [e] for e in eqs.keys()}
        tried = set()
        orig = set(eqs.keys())
        norm_cache = {}
        max_iter = max_iterations if max_iterations is not None else 20
        
        # Pre-compute parameter sets
        eq_params = {}
        for eq_name, expr in eqs.items():
            eq_params[eq_name] = {p for p in self.parameters.values() 
                                 if p in expr.free_symbols}
        
        # ===== PHASE 1: CLUSTER-FOCUSED =====
        if debug:
            print(f"\n{'='*60}")
            print(f"PHASE 1: CLUSTER-FOCUSED")
            print(f"{'='*60}")
        
        for it in range(max_iter):
            if debug:
                print(f"\n=== Iteration {it} ===")
            
            problematic = self._detect_problematic_clusters(eqs, self.variables, debug=debug)
            
            if not problematic:
                if debug:
                    print(f"  ✓ All singleton. Phase 1 done.")
                break
            
            cluster_equations = set()
            for cluster in problematic:
                cluster_equations.update(cluster['equations'])
            
            if debug:
                print(f"  Cluster equations: {len(cluster_equations)}")
            
            # Only try pairs within clusters
            pairs_to_try = []
            for e1 in sorted(eqs.keys()):
                if e1 not in cluster_equations:
                    continue
                for e2 in sorted(eqs.keys()):
                    if e1 >= e2 or e2 not in cluster_equations:
                        continue
                    pk = frozenset([e1, e2])
                    if pk in tried:
                        continue
                    if it > 0 and e1 in orig and e2 in orig:
                        continue
                    shared_params = eq_params.get(e1, set()) & eq_params.get(e2, set())
                    if len(shared_params) == 0:
                        continue
                    pairs_to_try.append((e1, e2))
                    tried.add(pk)
            
            if debug:
                print(f"  Pairs to try: {len(pairs_to_try)}")
            
            cands = []
            for e1, e2 in pairs_to_try:
                is_ABC_pair = ('A' in e1 or 'B' in e1 or 'C' in e1) and ('A' in e2 or 'B' in e2 or 'C' in e2)
                
                vars1 = {v for v in self.variables.values() if v in eqs[e1].free_symbols}
                vars2 = {v for v in self.variables.values() if v in eqs[e2].free_symbols}
                shared_vars = vars1 & vars2
                
                combos = {(1, 1)}
                for var in shared_vars:
                    try:
                        c1 = eqs[e1].coeff(var, 1)
                        c2 = eqs[e2].coeff(var, 1)
                        if c1 and c2 and c1 != 0 and c2 != 0:
                            combos.add((c2, -c1))
                            if debug and is_ABC_pair and ('Kinase' in str(c1) or 'Kinase' in str(c2)):
                                print(f"  [ABC DEBUG] {e1} + {e2}, var={var.name}")
                                print(f"    c1={str(c1)[:80]}, c2={str(c2)[:80]}")
                                print(f"    Combo: ({str(c2)[:40]}, {str(-c1)[:40]})")
                    except:
                        pass
                
                be, bc, bco = 0, None, None
                best_var_elim = 0
                
                for a1, a2 in combos:
                    cb = sp.simplify(a1*eqs[e1] + a2*eqs[e2])
                    if cb == 0:
                        continue
                    dup = any(self._is_scalar_multiple(cb, ex, norm_cache) 
                             for ex in eqs.values() if ex != 0)
                    if dup:
                        continue
                    
                    p1 = eq_params.get(e1, set())
                    p2 = eq_params.get(e2, set())
                    pc = {p for p in self.parameters.values() if p in cb.free_symbols}
                    param_elim = len(p1 | p2) - len(pc)
                    
                    v1 = {v for v in self.variables.values() if v in eqs[e1].free_symbols}
                    v2 = {v for v in self.variables.values() if v in eqs[e2].free_symbols}
                    vc = {v for v in self.variables.values() if v in cb.free_symbols}
                    var_elim = len(v1 | v2) - len(vc)
                    
                    total_benefit = param_elim + var_elim
                    if total_benefit > be or (total_benefit == be and var_elim > best_var_elim):
                        be = total_benefit
                        bc = cb
                        bco = (a1, a2)
                        best_var_elim = var_elim
                
                if be > 0:
                    nl = sorted(set(lineage[e1] + lineage[e2]))
                    cands.append((be, best_var_elim, -len(nl), e1, e2, bc, nl, bco))
            
            if debug:
                print(f"  Found {len(cands)} candidates")
            
            if not cands:
                break
            
            cands.sort(key=lambda x: (-x[0], -x[1], x[2]))
            
            added = 0
            max_to_add = min(5, len(cands))
            
            for benefit, var_elim, _, e1, e2, cb, nl, co in cands[:max_to_add]:
                if benefit < 1:
                    break
                base_name = '_'.join([e.replace('f_', '') for e in nl]) + "'"
                nn, ver = base_name, 1
                while nn in eqs:
                    nn = f"{base_name}_v{ver}"
                    ver += 1
                
                param_elim = benefit - var_elim
                if debug:
                    co_str = f"({str(co[0])[:15]}, {str(co[1])[:15]})" if co != (1, 1) else "(1, 1)"
                    print(f"  {nn} ← {e1} + {e2} {co_str}: {param_elim}p + {var_elim}v")
                
                eqs[nn] = cb
                lineage[nn] = nl
                eq_params[nn] = {p for p in self.parameters.values() if p in cb.free_symbols}
                added += 1
            
            if added == 0:
                break
        
        # ===== PHASE 2: FULL SWEEP =====
        if debug:
            print(f"\n{'='*60}")
            print(f"PHASE 2: FULL SWEEP")
            print(f"{'='*60}")
        
        phase2_start = len(eqs)
        
        for it in range(3):
            if debug:
                print(f"\n=== Full Sweep Iteration {it} ===")
            
            # Try ALL pairs
            pairs_to_try = []
            for e1 in sorted(eqs.keys()):
                for e2 in sorted(eqs.keys()):
                    if e1 >= e2:
                        continue
                    pk = frozenset([e1, e2])
                    if pk in tried:
                        continue
                    if it > 0 and e1 in orig and e2 in orig:
                        continue
                    shared_params = eq_params.get(e1, set()) & eq_params.get(e2, set())
                    if len(shared_params) == 0:
                        continue
                    pairs_to_try.append((e1, e2))
                    tried.add(pk)
            
            if debug:
                print(f"  Pairs to try: {len(pairs_to_try)}")
            
            if not pairs_to_try:
                if debug:
                    print(f"  No new pairs to try")
                break
            
            cands = []
            for e1, e2 in pairs_to_try:
                vars1 = {v for v in self.variables.values() if v in eqs[e1].free_symbols}
                vars2 = {v for v in self.variables.values() if v in eqs[e2].free_symbols}
                shared_vars = vars1 & vars2
                
                combos = {(1, 1)}
                for var in shared_vars:
                    try:
                        c1 = eqs[e1].coeff(var, 1)
                        c2 = eqs[e2].coeff(var, 1)
                        if c1 and c2 and c1 != 0 and c2 != 0:
                            combos.add((c2, -c1))
                    except:
                        pass
                
                be, bc, bco = 0, None, None
                best_var_elim = 0
                
                for a1, a2 in combos:
                    cb = sp.simplify(a1*eqs[e1] + a2*eqs[e2])
                    if cb == 0:
                        continue
                    dup = any(self._is_scalar_multiple(cb, ex, norm_cache) 
                             for ex in eqs.values() if ex != 0)
                    if dup:
                        continue
                    
                    p1 = eq_params.get(e1, set())
                    p2 = eq_params.get(e2, set())
                    pc = {p for p in self.parameters.values() if p in cb.free_symbols}
                    param_elim = len(p1 | p2) - len(pc)
                    
                    v1 = {v for v in self.variables.values() if v in eqs[e1].free_symbols}
                    v2 = {v for v in self.variables.values() if v in eqs[e2].free_symbols}
                    vc = {v for v in self.variables.values() if v in cb.free_symbols}
                    var_elim = len(v1 | v2) - len(vc)
                    
                    total_benefit = param_elim + var_elim
                    if total_benefit > be or (total_benefit == be and var_elim > best_var_elim):
                        be = total_benefit
                        bc = cb
                        bco = (a1, a2)
                        best_var_elim = var_elim
                
                if be > 0:
                    nl = sorted(set(lineage[e1] + lineage[e2]))
                    cands.append((be, best_var_elim, -len(nl), e1, e2, bc, nl, bco))
            
            if debug:
                print(f"  Found {len(cands)} candidates")
            
            if not cands:
                break
            
            cands.sort(key=lambda x: (-x[0], -x[1], x[2]))
            
            added = 0
            max_to_add = min(3, len(cands))
            
            for benefit, var_elim, _, e1, e2, cb, nl, co in cands[:max_to_add]:
                if benefit < 1:
                    break
                base_name = '_'.join([e.replace('f_', '') for e in nl]) + "'"
                nn, ver = base_name, 1
                while nn in eqs:
                    nn = f"{base_name}_v{ver}"
                    ver += 1
                
                param_elim = benefit - var_elim
                if debug:
                    co_str = f"({str(co[0])[:15]}, {str(co[1])[:15]})" if co != (1, 1) else "(1, 1)"
                    print(f"  {nn} ← {e1} + {e2} {co_str}: {param_elim}p + {var_elim}v")
                    vars_in = sorted([v.name for v in self.variables.values() if v in cb.free_symbols])
                    print(f"    Variables: {vars_in}")
                    
                    # Extra debug for A_B_C equations
                    if 'A' in nn and 'B' in nn and 'C' in nn:
                        print(f"    [DEBUG A_B_C] Full expression: {str(cb)[:200]}")
                
                eqs[nn] = cb
                lineage[nn] = nl
                eq_params[nn] = {p for p in self.parameters.values() if p in cb.free_symbols}
                added += 1
            
            if added == 0:
                break
        
        if debug and len(eqs) > phase2_start:
            print(f"\n  Phase 2 added {len(eqs) - phase2_start} equations")
                    
        return eqs, lineage
    
    def algorithm2_variable_transformation(self, debug=True, base_equations=None, base_lineage=None, return_lineage=False):
        """Algorithm 2: Cluster-focused THEN full sweep with upstream parameter benefit"""
        from itertools import combinations
        
        eqs = dict(base_equations if base_equations is not None else self.equations)
        variables = dict(self.variables)
        lineage = dict(base_lineage) if base_lineage is not None else {e: [e] for e in eqs.keys()}
        
        # ===== PHASE 1: CLUSTER-FOCUSED =====
        if debug:
            print(f"\n{'='*60}")
            print(f"PHASE 1: CLUSTER-FOCUSED TRANSFORMATIONS")
            print(f"{'='*60}")
        
        problematic = self._detect_problematic_clusters(eqs, variables, debug=debug)
        
        if problematic:
            cluster_vars = set()
            for cluster in problematic:
                cluster_vars.update(cluster['variables'])
            
            multiplicative_vars = set()
            for eq_name, expr in eqs.items():
                for term in sp.preorder_traversal(expr):
                    if isinstance(term, sp.Mul):
                        vars_in = {v for v in self.variables.values() if v in term.free_symbols}
                        if len(vars_in) >= 2:
                            multiplicative_vars.update(vars_in)
            
            multiplicative_vars = {v for v in multiplicative_vars if v.name in cluster_vars}
            
            if debug:
                print(f"  Cluster mult vars: {sorted([v.name for v in multiplicative_vars])}")
            
            if multiplicative_vars:
                eqs, variables, lineage = self._try_transformations(
                    eqs, variables, multiplicative_vars, lineage, debug, phase_name="Phase 1"
                )
                if debug:
                    R_after_p1 = [eq for eq in eqs.keys() if eq.startswith('f_R_')]
                    print(f"  [PHASE 1 COMPLETE] R equations in eqs: {R_after_p1}")
            else:
                if debug:
                    print(f"  No transformations to test")
        
        # ===== PHASE 2: FULL SWEEP =====
        if debug:
            print(f"\n{'='*60}")
            print(f"PHASE 2: FULL SWEEP TRANSFORMATIONS")
            print(f"{'='*60}")
        
        # Find ALL multiplicative vars
        all_mult_vars = set()
        for expr in eqs.values():
            for term in sp.preorder_traversal(expr):
                if isinstance(term, sp.Mul):
                    vars_in = {v for v in self.variables.values() if v in term.free_symbols}
                    if len(vars_in) >= 2:
                        all_mult_vars.update(vars_in)
        
        if debug:
            print(f"  All mult vars: {sorted([v.name for v in all_mult_vars])}")
        
        if all_mult_vars:
            eqs, variables, lineage = self._try_transformations(
                eqs, variables, all_mult_vars, lineage, debug, phase_name="Phase 2"
            )
            if debug:
                R_after_p2 = [eq for eq in eqs.keys() if eq.startswith('f_R_')]
                print(f"  [PHASE 2 COMPLETE] R equations in eqs: {R_after_p2}")
        
        if debug:
            R_final = [eq for eq in eqs.keys() if eq.startswith('f_R_')]
            print(f"\n  [RETURNING FROM ALG2] R equations: {R_final}")
            print(f"  [RETURNING FROM ALG2] Total equations: {len(eqs)}")
        
        if return_lineage:
            return eqs, variables, lineage
        else:
            return eqs, variables
    
    def _try_transformations(self, eqs, variables, mult_vars, lineage, debug, phase_name=""):
        """Helper: try ratio transformations using upstream parameter benefit"""
        from itertools import combinations
        
        candidates = {}
        for v1, v2 in combinations(mult_vars, 2):
            for num, den in [(v1, v2), (v2, v1)]:
                name = f"R_{num.name}_{den.name}"
                if name not in variables:
                    candidates[name] = {
                        'new_var': sp.Symbol(name, real=True, positive=True),
                        'num': num, 
                        'den': den
                    }
        
        if not candidates:
            if debug:
                print(f"  No transformations to test")
            return eqs, variables, lineage
        
        if debug:
            print(f"  Testing {len(candidates)} transformations")
        
        best_benefit = 0
        best_result = None
        
        for trans_name, info in candidates.items():
            # Debug output for key transformations
            is_key = ('D' in trans_name and 'Kinase' in trans_name)
            is_KK = 'Kinase_Kinase_new' in trans_name or 'Kinase_new_Kinase' in trans_name
            
            # Add R and rewrite
            test_vars = dict(variables)
            test_vars[trans_name] = info['new_var']
            
            test_eqs = dict(eqs)
            test_eqs[f"f_{trans_name}"] = info['new_var'] - info['num'] / info['den']
            
            # Track lineage for R equation
            test_lineage = dict(lineage)
            test_lineage[f"f_{trans_name}"] = [f"f_{trans_name}"]
            
            rewritten = 0
            for eq_name in list(test_eqs.keys()):
                if eq_name == f"f_{trans_name}" or eq_name.startswith('f_R_'):
                    continue
                if info['num'] in test_eqs[eq_name].free_symbols:
                    test_eqs[eq_name] = sp.expand(
                        test_eqs[eq_name].subs(info['num'], info['new_var'] * info['den'])
                    )
                    rewritten += 1
            
            if debug and (is_key or is_KK):
                print(f"\n  [{'KEY' if is_key else 'KK'}] {trans_name}: rewrote {rewritten} eqs")
            
            # Check benefit using upstream parameter count with FORCED R matchings
            try:
                baseline_score = sum(self._compute_upstream_params(eqs, variables, debug=False).values())
                
                # Try TWO forced matchings for R
                best_new_score = float('inf')
                
                # Option 1: Force Kinase ← f_R
                try:
                    score1 = self._compute_upstream_with_forced_R_matching(
                        test_eqs, test_vars, f"f_{trans_name}", info['den'].name
                    )
                    best_new_score = min(best_new_score, score1)
                    if debug and (is_key or is_KK):
                        print(f"    Force {info['den'].name} ← f_{trans_name}: score={score1}")
                except:
                    pass
                
                # Option 2: Force Kinase_new ← f_R  
                try:
                    score2 = self._compute_upstream_with_forced_R_matching(
                        test_eqs, test_vars, f"f_{trans_name}", info['num'].name
                    )
                    best_new_score = min(best_new_score, score2)
                    if debug and (is_key or is_KK):
                        print(f"    Force {info['num'].name} ← f_{trans_name}: score={score2}")
                except:
                    pass
                
                if best_new_score == float('inf'):
                    # Fallback: compute normally
                    best_new_score = sum(self._compute_upstream_params(test_eqs, test_vars, debug=False).values())
                
                benefit = baseline_score - best_new_score
                
                if debug and (is_key or is_KK):
                    print(f"    Baseline: {baseline_score}, Best new: {best_new_score}")
                    print(f"    Benefit: {benefit}")
                
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_result = (trans_name, info, test_eqs, test_vars, test_lineage)
                    if debug and (is_key or is_KK):
                        print(f"    ⭐ NEW BEST")
            except Exception as e:
                if debug and (is_key or is_KK):
                    print(f"    Error computing benefit: {e}")
        
        if best_result:
            trans_name, info, new_eqs, new_vars, new_lineage = best_result
            if debug:
                print(f"\n  ✓ {phase_name} SELECTED: {trans_name}")
                R_eq_name = f"f_{trans_name}"
                if R_eq_name in new_eqs:
                    print(f"    Confirmed: {R_eq_name} in equations")
                    # Extra debug for unexpected Rs
                    if 'Kinase_Kinase_new' not in trans_name and 'Kinase_new_Kinase' not in trans_name:
                        print(f"    [UNEXPECTED R] Why was this selected over R_Kinase_Kinase_new?")
                        print(f"      Benefit: {best_benefit}")
                else:
                    print(f"    ERROR: {R_eq_name} NOT in equations!")
            return new_eqs, new_vars, new_lineage
        else:
            if debug:
                print(f"  {phase_name}: No beneficial transformations")
            return eqs, variables, lineage
    
    def algorithm1_optimal_selection(self, all_equations, lineage, debug=False):
        """Smart optimal selection via lineage-aware deduplication"""
        if debug:
            print(f"\n=== Smart Optimal Selection ===")
            print(f"All equations: {len(all_equations)}")
        
        n = len(self.variables)
        original_names = sorted(self.equations.keys())
        
        # Build lineage vectors
        def eq_to_vector(eq_name):
            vec = [0] * len(original_names)
            for base_eq in lineage.get(eq_name, [eq_name]):
                if base_eq in original_names:
                    vec[original_names.index(base_eq)] = 1
            return vec
        
        vectors = {eq_name: eq_to_vector(eq_name) for eq_name in all_equations.keys()}
        
        # Deduplicate by lineage vector
        unique_by_lineage = {}
        for eq_name, vec in vectors.items():
            vec_tuple = tuple(vec)
            if vec_tuple not in unique_by_lineage:
                unique_by_lineage[vec_tuple] = eq_name
        
        if debug:
            print(f"After lineage dedup: {len(unique_by_lineage)}")
        
        # Enumerate all combinations and check rank
        valid_bases = []
        for eq_combo in combinations(unique_by_lineage.values(), n):
            mat = sp.Matrix([vectors[eq] for eq in eq_combo])
            if mat.rank() == n:
                valid_bases.append(list(eq_combo))
        
        if debug:
            print(f"Valid bases: {len(valid_bases)}")
        
        if not valid_bases:
            if debug:
                print("  No valid bases found, falling back to greedy")
            return self.algorithm1_greedy_selection(all_equations, lineage, debug)
        
        # Score each basis with cluster tie-breaking
        scores_with_clusters = []
        for eq_names in valid_bases:
            eqs_used = {eq: all_equations[eq] for eq in eq_names}
            try:
                upstream_params = self._compute_upstream_params(eqs_used, self.variables, debug=False)
                score = sum(upstream_params.values())
                
                cluster_graph = self.causal_ordering(eqs_used, self.variables, debug=False)
                multi_var_clusters = 0
                total_cluster_size = 0
                for cname, data in cluster_graph.nodes(data=True):
                    vars_in = data.get('variables', [])
                    if len(vars_in) > 1:
                        multi_var_clusters += 1
                        total_cluster_size += len(vars_in)
                
                scores_with_clusters.append((score, multi_var_clusters, total_cluster_size, eq_names, upstream_params))
            except:
                pass
        
        scores_with_clusters.sort(key=lambda x: (x[0], x[1], x[2]))
        
        if debug and scores_with_clusters:
            print(f"\n  Top 5 bases:")
            for i, (score, n_clusters, cluster_size, eq_names, _) in enumerate(scores_with_clusters[:5]):
                print(f"    {i+1}. Score={score}, Clusters={n_clusters}x{cluster_size}: {list(eq_names)}")
            
            unique_scores = len(set(s[0] for s in scores_with_clusters))
            if unique_scores == 1:
                print(f"\n  ⚠️  ALL {len(scores_with_clusters)} bases have identical score={scores_with_clusters[0][0]}")
                unique_clusters = len(set((s[1], s[2]) for s in scores_with_clusters))
                print(f"  But {unique_clusters} unique cluster configurations")
            else:
                print(f"\n  Unique scores: {unique_scores}")
        
        best = scores_with_clusters[0]
        return {eq: all_equations[eq] for eq in best[3]}
    
    def algorithm1_greedy_selection(self, all_equations, lineage, debug=False):
        """Greedy selection fallback"""
        if debug:
            print(f"\n=== Greedy Selection ===")
        
        B = self._build_bipartite(all_equations, self.variables)
        degrees = {eq: B.degree(eq) for eq in all_equations.keys()}
        
        original_names = sorted(self.equations.keys())
        def eq_to_vector(eq_name):
            vec = [0] * len(original_names)
            for base_eq in lineage.get(eq_name, [eq_name]):
                if base_eq in original_names:
                    vec[original_names.index(base_eq)] = 1
            return vec
        
        vectors = {eq_name: eq_to_vector(eq_name) for eq_name in all_equations.keys()}
        sorted_eqs = sorted(all_equations.keys(), key=lambda eq: degrees[eq])
        
        selected = {}
        n = len(self.variables)
        max_matched = 0
        
        for eq_name in sorted_eqs:
            test_eqs = dict(selected)
            test_eqs[eq_name] = all_equations[eq_name]
            
            eq_names_test = list(test_eqs.keys())
            mat = sp.Matrix([vectors[eq] for eq in eq_names_test])
            if mat.rank() != len(eq_names_test):
                continue
            
            B_test = self._build_bipartite(test_eqs, self.variables)
            left = {n for n, d in B_test.nodes(data=True) if d.get('bipartite') == 0}
            M = nx.bipartite.maximum_matching(B_test, top_nodes=left)
            matched = len([v for v in self.variables.keys() if f"v_{v}" in M])
            
            if matched > max_matched:
                selected[eq_name] = all_equations[eq_name]
                max_matched = matched
                if matched == n:
                    break
        
        return selected
    
    def compare(self, debug=False, use_optimal=False, max_iterations=None):
        """Full comparison with Algorithm 1→2 workflow"""
        self.print_equations()
        
        print("\n[1] Standard")
        g1 = self.causal_ordering(self.equations, self.variables)
        
        print("\n[2] Algorithm 1")
        all_eqs, lineage = self.algorithm1_generate_equations(debug, max_iterations)
        
        print("\n[3] Algorithm 2")
        transformed_eqs, transformed_vars, updated_lineage = self.algorithm2_variable_transformation(
            debug=debug, base_equations=all_eqs, base_lineage=lineage, return_lineage=True
        )
        
        print(f"\nAfter transformation:")
        print(f"  Variables: {len(transformed_vars)} (original: {len(self.variables)})")
        print(f"  Equations: {len(transformed_eqs)}")
        
        if debug:
            R_eqs_list = [eq for eq in transformed_eqs.keys() if eq.startswith('f_R_')]
            print(f"  R equations in transformed_eqs: {len(R_eqs_list)}")
            if R_eqs_list:
                print(f"    {R_eqs_list}")
            else:
                print(f"    WARNING: No R equations found despite variables existing!")
                print(f"    All equation names: {list(transformed_eqs.keys())}")
        
        # Separate R equations (must always be included)
        R_equations = {eq: expr for eq, expr in transformed_eqs.items() if eq.startswith('f_R_')}
        non_R_eqs = {eq: expr for eq, expr in transformed_eqs.items() if not eq.startswith('f_R_')}
        
        if R_equations:
            print(f"\n[Final Selection] R variables detected: {list(R_equations.keys())}")
            print(f"  Selecting from {len(non_R_eqs)} non-R equations for {len(self.variables)} original variables")
            if use_optimal:
                selected = self.algorithm1_optimal_selection(non_R_eqs, updated_lineage, debug)
            else:
                selected = self.algorithm1_greedy_selection(non_R_eqs, updated_lineage, debug)
            # Add R equations back
            for R_eq, R_expr in R_equations.items():
                selected[R_eq] = R_expr
            print(f"  Final: {len(selected)} equations ({len(selected) - len(R_equations)} selected + {len(R_equations)} R)")
        else:
            print(f"\n[Final Selection] No R variables")
            if use_optimal:
                selected = self.algorithm1_optimal_selection(transformed_eqs, updated_lineage, debug)
            else:
                selected = self.algorithm1_greedy_selection(transformed_eqs, updated_lineage, debug)
        
        print(f"Final equations: {list(selected.keys())}")
        
        if debug:
            print(f"\n[DEBUG] Final Matching Details:")
            final_graph = self.causal_ordering(selected, transformed_vars, debug=False)
            B_temp = self._build_bipartite(selected, transformed_vars)
            left_temp = {n for n, d in B_temp.nodes(data=True) if d.get('bipartite') == 0}
            M_temp = nx.bipartite.maximum_matching(B_temp, top_nodes=left_temp)
            for var_name in sorted(self.variables.keys()):
                var_node = f"v_{var_name}"
                if var_node in M_temp:
                    eq_matched = M_temp[var_node]
                    vars_in_eq = sorted([v for v in transformed_vars.keys() if transformed_vars[v] in selected[eq_matched].free_symbols])
                    print(f"  {var_name} ← {eq_matched}: vars={vars_in_eq}")
                    
                    # Extra detail for B
                    if var_name == 'B':
                        print(f"    [B EQUATION] {str(selected[eq_matched])[:300]}")
        
        g2 = self.causal_ordering(all_eqs, self.variables)
        g3 = self.causal_ordering(selected, transformed_vars)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        self.visualize(g1, axes[0], "Standard")
        self.visualize(g2, axes[1], "After Algorithm 1")  
        self.visualize(g3, axes[2], "After Algorithm 2")
        plt.tight_layout()
        plt.show()
    
    def test_algorithm1_only(self, debug=True):
        """Test only Algorithm 1"""
        print(f"\n{'='*70}")
        print(f"ALGORITHM 1 TEST")
        print(f"{'='*70}")
        
        self.print_equations()
        all_eqs, lineage = self.algorithm1_generate_equations(debug=True, max_iterations=5)
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Total equations: {len(all_eqs)}")
        
        return all_eqs, lineage
    
    def test_algorithm2_only(self, base_equations=None, base_lineage=None, debug=True):
        """Test only Algorithm 2"""
        print(f"\n{'='*70}")
        print(f"ALGORITHM 2 TEST")
        print(f"{'='*70}")
        
        transformed_eqs, transformed_vars, updated_lineage = self.algorithm2_variable_transformation(
            debug=True, base_equations=base_equations, base_lineage=base_lineage, return_lineage=True
        )
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Variables: {len(transformed_vars)}")
        print(f"Equations: {len(transformed_eqs)}")
        
        return transformed_eqs, transformed_vars, updated_lineage
    
    def visualize(self, graph, ax, title=""):
        nodes = [n for n in graph.nodes() if graph.nodes[n].get('variables') or graph.nodes[n].get('exogenous')]
        if not nodes:
            ax.axis('off')
            return
        sub = graph.subgraph(nodes)
        
        try:
            layers = list(nx.topological_generations(sub))
            pos = {}
            for li, layer in enumerate(layers):
                for ni, node in enumerate(list(layer)):
                    pos[node] = (li * 4, (len(layer)-1)*1.2 - ni*2.4)
        except:
            pos = nx.spring_layout(sub, k=4, seed=42)
        
        for u, v in sub.edges():
            x1, y1, x2, y2 = pos[u][0], pos[u][1], pos[v][0], pos[v][1]
            dx, dy = x2-x1, y2-y1
            L = (dx**2+dy**2)**0.5
            if L > 0:
                ax.annotate('', xy=(x2-dx/L*0.6, y2-dy/L*0.6), xytext=(x1+dx/L*0.6, y1+dy/L*0.6),
                           arrowprops=dict(arrowstyle='->', lw=3, color='#333'))
        
        for node in nodes:
            x, y = pos[node]
            d = graph.nodes[node]
            parts = []
            if d.get('exogenous'):
                parts.append(', '.join(d['exogenous']))
            if d.get('variables'):
                parts.append(', '.join(d['variables']))
            if d.get('equations'):
                parts.append('(' + ', '.join([e.replace('_prime',"'") for e in d['equations']]) + ')')
            label = '\n'.join(parts)
            w = max(1.2, max(len(p) for p in parts)*0.09) if parts else 1.2
            h = max(0.7, len(parts)*0.4+0.2) if parts else 0.7
            box = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.06",
                                         ec='black', fc='#E3F2FD', lw=3, zorder=10)
            ax.add_patch(box)
            ax.text(x, y, label, ha='center', va='center', size=12, weight='bold', zorder=11)
        
        xs, ys = zip(*pos.values()) if pos else ([0], [0])
        ax.set_xlim(min(xs)-2, max(xs)+2)
        ax.set_ylim(min(ys)-2, max(ys)+2)
        ax.axis('off')
        ax.set_aspect('equal')
        if title:
            ax.set_title(title, size=14, weight='bold', pad=25)
    
    def print_equations(self):
        print(f"\n{self.name}")
        print("="*70)
        if self.conservation_params:
            print(f"Conservation params: {sorted(self.conservation_params)}")
        for eq in sorted(self.equations.keys()):
            print(f"{eq}: {self.equations[eq]} = 0")
        print("="*70)