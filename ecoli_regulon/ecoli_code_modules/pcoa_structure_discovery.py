"""
MODULE 3: PCOA Structure Discovery - FIXED

Responsibility: Discover causal structure from equilibrium equations
- Run Algorithm 1 (parameter cancellation)
- Run Algorithm 2 (variable transformation)  
- Add inhibitions via lineage tracking
- Return DAG structure

CRITICAL FIX: Separate R equations before selection!
"""

import sympy as sp
import networkx as nx


class PCOADiscovery:
    def __init__(self, pkn):
        self.pkn = pkn
        self.vars = pkn['variables']
        self.k_in = pkn['exogenous']
        self.inhib = pkn['inhibitions']
    
    def discover_structure(self, equilibrium_equations, debug=False):
        """
        Main entry point: equilibrium equations → causal structure
        
        Workflow:
        1. Algorithm 1: Generate ALL rewritten equations
        2. Algorithm 2: Transform using ALL equations (discovers constants)
        3. Separate R equations (must always be included)
        4. Final Selection: Pick optimal basis for ORIGINAL variables
        5. Add R equations back
        6. Add inhibitions and convert to DAG
        
        Args:
            equilibrium_equations: {eq_name: sympy_expr}
            debug: Print detailed output
        
        Returns:
            structure: {var: [parent_list]}
            topo_order: [var1, var2, ...]
        """
        from causal_ordering_analyzer import CausalOrderingAnalyzer
        
        # Extract symbols
        all_syms = set()
        for expr in equilibrium_equations.values():
            all_syms |= expr.free_symbols
        
        var_syms = [sp.Symbol(v, real=True, positive=True) for v in self.vars]
        exo_syms = [sp.Symbol(self.k_in, real=True, positive=True)]
        
        var_names = {str(s) for s in var_syms}
        exo_names = {str(s) for s in exo_syms}
        param_syms = [s for s in all_syms if str(s) not in var_names | exo_names]
        
        # Initialize analyzer
        analyzer = CausalOrderingAnalyzer(
            equilibrium_equations, var_syms, param_syms, exo_syms
        )
        
        if debug:
            print("\n[Algorithm 1] Equation Rewriting")
            print("-" * 60)
        
        # Algorithm 1: Generate ALL rewritten equations
        all_eqs, lineage = analyzer.algorithm1_generate_equations(
            debug=debug, max_iterations=10
        )
        
        if debug:
            print(f"\nGenerated {len(all_eqs)} total equations")
            print(f"  Original: {len(equilibrium_equations)}")
            print(f"  Combined: {len(all_eqs) - len(equilibrium_equations)}")
        
        if debug:
            print("\n[Algorithm 2] Variable Transformation")
            print("-" * 60)
        
        # Algorithm 2: Transform using ALL equations
        transformed_eqs, transformed_vars = analyzer.algorithm2_variable_transformation(
            debug=debug,
            base_equations=all_eqs,
            base_lineage=lineage,
            return_lineage=False  # Don't need lineage return here
        )
        
        # Update lineage for any equations not already tracked
        final_lineage = dict(lineage)
        for eq_name in transformed_eqs:
            if eq_name not in final_lineage:
                final_lineage[eq_name] = [eq_name]
        
        if debug:
            print(f"\nAfter transformation:")
            print(f"  Variables: {len(transformed_vars)} (original: {len(var_syms)})")
            print(f"  Equations: {len(transformed_eqs)}")
        
        # CRITICAL FIX: Separate R equations before selection
        R_equations = {eq: expr for eq, expr in transformed_eqs.items() if eq.startswith('f_R_')}
        non_R_eqs = {eq: expr for eq, expr in transformed_eqs.items() if not eq.startswith('f_R_')}
        
        if debug and R_equations:
            print(f"  R equations to preserve: {list(R_equations.keys())}")
        
        if debug:
            print("\n[Final Selection] Optimal Equation Basis")
            print("-" * 60)
        
        # Select for ORIGINAL variables only (not R variables)
        if len(non_R_eqs) > len(var_syms):
            if debug:
                print(f"  Selecting {len(var_syms)} from {len(non_R_eqs)} non-R equations")
            selected_eqs = analyzer.algorithm1_optimal_selection(
                non_R_eqs, final_lineage, debug=debug
            )
        else:
            # Already minimal
            selected_eqs = non_R_eqs
            if debug:
                print(f"  Using all {len(non_R_eqs)} non-R equations (already minimal)")
        
        # Add R equations back (always included)
        for R_eq, R_expr in R_equations.items():
            selected_eqs[R_eq] = R_expr
        
        if debug and R_equations:
            print(f"  Added {len(R_equations)} R equations back")
            print(f"  Final total: {len(selected_eqs)} equations")
        
        if debug:
            print(f"\nFinal equation set: {list(selected_eqs.keys())}")
            print(f"Final variables: {list(transformed_vars.keys())}")
        
        # Build causal ordering graph
        cluster_graph = analyzer.causal_ordering(selected_eqs, transformed_vars, debug=False)
        
        # Add inhibitions using original lineage
        cluster_graph = self._add_inhibitions(cluster_graph, selected_eqs, final_lineage)
        
        # Convert to variable DAG (only keep original variables, drop transformed ones)
        dag = self._to_variable_dag_filtered(cluster_graph, transformed_vars)
        
        if not nx.is_directed_acyclic_graph(dag):
            if debug:
                print("WARNING: DAG has cycles, returning trivial structure")
            return {v: [] for v in self.vars}, sorted(self.vars)
        
        topo = list(nx.topological_sort(dag))
        structure = {v: list(dag.predecessors(v)) if v in dag else [] 
                    for v in self.vars}
        
        return structure, topo
    
    
    def _to_variable_dag_filtered(self, cluster_graph, transformed_vars):
        """
        Convert cluster graph to DAG, keeping only ORIGINAL variables
        
        Transformed variables (R_*, P_*) are filtered out since they were
        eliminated during transformation.
        """
        import networkx as nx
        dag = nx.DiGraph()
        cluster_contents = {}
        
        # Identify which variables to keep (original + exogenous)
        original_var_names = set(self.vars)
        exogenous_names = {self.k_in} | set(self.inhib.values())
        
        for cluster, data in cluster_graph.nodes(data=True):
            # Filter to keep only original variables
            vars_in = [v for v in data.get('variables', []) 
                      if v in original_var_names]
            
            exo_in = [e for e in data.get('exogenous', [])
                     if e in exogenous_names]
            
            if vars_in or exo_in:
                cluster_contents[cluster] = vars_in + exo_in
                
                for node in cluster_contents[cluster]:
                    node_type = 'exogenous' if node in exo_in else 'endogenous'
                    dag.add_node(node, type=node_type)
        
        # Expand cluster edges to variable edges
        for u, v in cluster_graph.edges():
            if u in cluster_contents and v in cluster_contents:
                for u_node in cluster_contents[u]:
                    for v_node in cluster_contents[v]:
                        if u_node != v_node:
                            dag.add_edge(u_node, v_node)
        
        return dag
    
    def _add_inhibitions(self, cluster_graph, selected_equations, lineage):
        """Add inhibition nodes using lineage tracking"""
        cg = cluster_graph.copy()
        
        # Map equations to clusters
        eq_to_cluster = {}
        for cluster, data in cg.nodes(data=True):
            for eq_name in data.get('equations', []):
                eq_to_cluster[eq_name] = cluster
        
        print("\n[DEBUG] Inhibition Addition:")
        print(f"  Selected equations: {list(selected_equations.keys())}")
        print(f"  Inhibitions to add: {self.inhib}")
        
        node_id = cg.number_of_nodes()
        
        for var, inhib_name in self.inhib.items():
            original_eq = f"f_{var}"
            print(f"\n  Checking {inhib_name} for original equation: {original_eq}")
            
            added_to = []
            for selected_eq in selected_equations.keys():
                eq_lineage = lineage.get(selected_eq, [selected_eq])
                
                print(f"    {selected_eq}: lineage = {eq_lineage}")
                
                if original_eq in eq_lineage and selected_eq in eq_to_cluster:
                    target_cluster = eq_to_cluster[selected_eq]
                    inhib_node = f"I_{node_id}"
                    node_id += 1
                    
                    cg.add_node(inhib_node, variables=[], equations=[], 
                               exogenous=[inhib_name])
                    cg.add_edge(inhib_node, target_cluster)
                    added_to.append(target_cluster)
                    print(f"      → Added {inhib_name} to cluster {target_cluster}")
            
            if not added_to:
                print(f"    ⚠️  {inhib_name} NOT added anywhere!")
        
        return cg