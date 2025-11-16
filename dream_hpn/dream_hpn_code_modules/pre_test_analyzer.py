"""
MODULE 7: Pre-Test Analysis - UPDATED WITH DIRECTED EDGES AND CYCLE HIGHLIGHTING

Enhanced visualization:
- Clearly directed edges with arrowheads
- Cycles highlighted in red
- Edge type distinction (activate vs convert)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import networkx as nx
import sympy as sp


class PreTestAnalyzer:
    def __init__(self, pkn, base_params):
        self.pkn = pkn
        self.base_params = base_params
        self.vars = pkn['variables']
        self.k_in = pkn['exogenous']
        self.inhib = pkn['inhibitions']
        
        # Publication colors
        self.colors = {
            'exogenous': '#FFE5B4',
            'endogenous': '#E3F2FD',
            'inhibited': '#FFEBEE',
            'edge_activate': '#2E7D32',
            'edge_convert': '#1565C0',
            'edge_cycle': '#D32F2F',  # RED for cycles
        }
    
    def _find_cycles(self, G):
        """
        Find all edges that are part of cycles
        
        Returns: set of edges (u, v) that participate in cycles
        """
        cycle_edges = set()
        
        try:
            # Find all simple cycles
            cycles = list(nx.simple_cycles(G))
            
            # Collect all edges in cycles
            for cycle in cycles:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    cycle_edges.add((u, v))
        except:
            pass
        
        return cycle_edges
    
    def run_full_analysis(self, check_saturation=True, fast_mode=True):
        """Run pre-test analysis"""
        print("="*80)
        print(f"PRE-TEST ANALYSIS ({'FAST' if fast_mode else 'THOROUGH'})")
        print("="*80)
        
        self._print_pkn_summary()
        stability_ok = self._check_stability()
        
        if check_saturation and self.pkn.get('saturation', {}).get('mm'):
            self._check_saturation_conditions()
        
        if not fast_mode:
            self._display_ode_models()
        else:
            print("\n✓ Skipping ODE equations display (fast mode)")
        
        structures = self._run_pcoa_analysis(debug=not fast_mode)
        self._compare_structures_diagnostic()
        
        if not fast_mode:
            print("\n[Creating visualizations...]")
            self.create_publication_figure(structures)
        else:
            print("\n✓ Skipping visualizations (fast mode)")
        
        return stability_ok
    
    def create_publication_figure(self, pcoa_structures, results_df=None,
                                  save_path='pcoa_analysis.png', dpi=300):
        """Create publication-quality multi-panel figure"""
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Panel A: PKN
        ax_pkn = fig.add_subplot(gs[0, :])
        self._plot_pkn(ax_pkn)
        
        # Panels B1-B3: PCOA structures
        titles = ["B1. Mass Action", "B2. Michaelis-Menten", "B3. Hill"]
        for idx, (form, title) in enumerate(zip(['mass_action', 'mm', 'hill'], titles)):
            ax = fig.add_subplot(gs[1, idx])
            if form in pcoa_structures:
                structure, topo = pcoa_structures[form]
                self._plot_dag(ax, structure, topo, title)
        
        # Panel C: Comparison
        ax_comp = fig.add_subplot(gs[2, :2])
        self._plot_structure_table(ax_comp, pcoa_structures)
        
        # Panel D: Results summary
        ax_results = fig.add_subplot(gs[2, 2])
        if results_df is not None:
            self._plot_results_summary(ax_results, results_df)
        else:
            ax_results.axis('off')
            ax_results.text(0.5, 0.5, "Run experiment\nto see results",
                          ha='center', va='center', fontsize=10,
                          transform=ax_results.transAxes)
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: {save_path}")
        plt.show()
        
        return fig
    
    def _plot_pkn(self, ax):
        """Plot PKN with directed edges and cycle highlighting"""
        ax.set_title("A. Prior Knowledge Network (PKN)", 
                    fontsize=14, weight='bold', pad=15)
        
        G = nx.DiGraph()
        
        # Build graph
        G.add_node(self.k_in, node_type='exogenous')
        for v in self.vars:
            node_type = 'inhibited' if v in self.inhib else 'endogenous'
            G.add_node(v, node_type=node_type)
        
        edge_types = {}
        for edge in self.pkn['edges']:
            src, tgt = edge[0], edge[1] if len(edge) >= 2 else None
            etype = edge[2] if len(edge) >= 3 else 'activate'
            if tgt:
                G.add_edge(src, tgt)
                edge_types[(src, tgt)] = etype
        
        # Find cycles
        cycle_edges = self._find_cycles(G)
        
        if cycle_edges:
            print(f"\n  [PKN] Found {len(cycle_edges)} edges in cycles:")
            for u, v in sorted(cycle_edges):
                print(f"    {u} → {v}")
        
        # Hierarchical layout
        pos = self._hierarchical_layout(G)
        
        # Draw edges by type and cycle status
        for (u, v), etype in edge_types.items():
            is_cycle = (u, v) in cycle_edges
            
            if is_cycle:
                color = self.colors['edge_cycle']
                linewidth = 3.5
                label_text = "Cycle" if (u, v) == list(cycle_edges)[0] else None
            else:
                color = self.colors.get(f'edge_{etype}', '#666')
                linewidth = 2.5
                label_text = None
            
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Arrowstyle based on edge type
            if etype == 'convert':
                arrowstyle = '-|>'  # Filled arrow for conversion
            else:
                arrowstyle = '->'   # Open arrow for activation
            
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle=arrowstyle,
                color=color,
                linewidth=linewidth,
                mutation_scale=25,
                zorder=1,
                label=label_text
            )
            ax.add_patch(arrow)
        
        # Draw nodes
        for node in G.nodes():
            x, y = pos[node]
            node_type = G.nodes[node]['node_type']
            color = self.colors.get(node_type, self.colors['endogenous'])
            
            width = 1.0 if node == self.k_in else 0.7
            height = 0.45
            
            box = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.08",
                ec='black', fc=color, lw=2, zorder=10
            )
            ax.add_patch(box)
            
            label = node
            if node in self.inhib:
                label += f"\n({self.inhib[node]})"
            
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=10, weight='bold', zorder=11)
        
        # Legend
        legend_elements = [
            mpatches.Patch(fc=self.colors['exogenous'], ec='black', label='Exogenous'),
            mpatches.Patch(fc=self.colors['endogenous'], ec='black', label='Endogenous'),
            mpatches.Patch(fc=self.colors['inhibited'], ec='black', label='Inhibitable'),
            mpatches.Rectangle((0, 0), 1, 0.3, fc=self.colors['edge_activate'], label='Activate'),
            mpatches.Rectangle((0, 0), 1, 0.3, fc=self.colors['edge_convert'], label='Convert'),
            mpatches.Rectangle((0, 0), 1, 0.3, fc=self.colors['edge_cycle'], label='Cycle'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
        
        xs, ys = zip(*pos.values())
        ax.set_xlim(min(xs)-1.2, max(xs)+1.2)
        ax.set_ylim(min(ys)-1, max(ys)+1)
        ax.axis('off')
    
    def _plot_dag(self, ax, structure, topo_order, title):
        """Plot PCOA-derived DAG with directed edges"""
        ax.set_title(title, fontsize=11, weight='bold', pad=10)
        
        G = nx.DiGraph()
        
        for var, parents in structure.items():
            if var not in self.vars:
                continue
            G.add_node(var)
            for parent in parents:
                G.add_node(parent)
                G.add_edge(parent, var)
        
        # Find cycles in discovered structure
        cycle_edges = self._find_cycles(G)
        
        # Layout
        try:
            layers = list(nx.topological_generations(G))
            pos = {}
            for li, layer in enumerate(layers):
                for ni, node in enumerate(sorted(layer)):
                    pos[node] = (li * 2, (len(layer)-1)*0.6 - ni*1.2)
        except:
            pos = nx.spring_layout(G, k=1.5, seed=42)
        
        # Draw edges with cycle highlighting
        for u, v in G.edges():
            is_cycle = (u, v) in cycle_edges
            color = self.colors['edge_cycle'] if is_cycle else '#555'
            width = 3 if is_cycle else 2
            
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='-|>',
                color=color,
                linewidth=width,
                mutation_scale=15,
                zorder=1,
                connectionstyle='arc3,rad=0.1'
            )
            ax.add_patch(arrow)
        
        # Draw nodes
        for node in G.nodes():
            x, y = pos[node]
            
            if node == self.k_in:
                color = self.colors['exogenous']
                w, h = 0.6, 0.35
            elif node in self.inhib.values():
                color = '#FFF9C4'
                w, h = 0.5, 0.3
            else:
                color = self.colors['endogenous']
                w, h = 0.55, 0.35
            
            box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                                boxstyle="round,pad=0.05",
                                ec='black', fc=color, lw=1.5, zorder=10)
            ax.add_patch(box)
            
            ax.text(x, y, node, ha='center', va='center',
                   fontsize=8, weight='bold', zorder=11)
        
        xs, ys = zip(*pos.values()) if pos else ([0], [0])
        ax.set_xlim(min(xs)-0.8, max(xs)+0.8)
        ax.set_ylim(min(ys)-0.8, max(ys)+0.8)
        ax.axis('off')
    
    def _plot_structure_table(self, ax, pcoa_structures):
        """Clean table showing structural differences"""
        ax.axis('off')
        ax.set_title("C. Structural Differences", fontsize=12, weight='bold', pad=10)
        
        # Find differences
        differs = []
        for var in self.vars:
            ma = set(pcoa_structures.get('mass_action', ({}, []))[0].get(var, []))
            mm = set(pcoa_structures.get('mm', ({}, []))[0].get(var, []))
            hill = set(pcoa_structures.get('hill', ({}, []))[0].get(var, []))
            
            if not (ma == mm == hill):
                differs.append((var, ma, mm, hill))
        
        if differs:
            y = 0.85
            ax.text(0.5, y, f"Variables with different parent sets: {len(differs)}",
                   ha='center', fontsize=10, weight='bold',
                   transform=ax.transAxes)
            
            y -= 0.1
            ax.text(0.05, y, "Variable", fontsize=9, weight='bold', transform=ax.transAxes)
            ax.text(0.25, y, "Mass Action", fontsize=9, weight='bold', transform=ax.transAxes)
            ax.text(0.50, y, "Michaelis-Menten", fontsize=9, weight='bold', transform=ax.transAxes)
            ax.text(0.75, y, "Hill", fontsize=9, weight='bold', transform=ax.transAxes)
            
            ax.plot([0.02, 0.98], [y-0.02, y-0.02], 'k-', lw=1, 
                   transform=ax.transAxes)
            
            y -= 0.08
            for var, ma, mm, hill in differs[:5]:
                ax.text(0.05, y, var, fontsize=8, weight='bold',
                       transform=ax.transAxes)
                ax.text(0.25, y, str(sorted(ma))[:20], fontsize=7,
                       transform=ax.transAxes, family='monospace')
                ax.text(0.50, y, str(sorted(mm))[:20], fontsize=7,
                       transform=ax.transAxes, family='monospace')
                ax.text(0.75, y, str(sorted(hill))[:20], fontsize=7,
                       transform=ax.transAxes, family='monospace')
                y -= 0.08
        else:
            y = 0.5
            ax.text(0.5, y, "✓ All formulations yield identical structure",
                   ha='center', va='center', fontsize=11,
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='#E8F5E9', alpha=0.9))
    
    def _plot_results_summary(self, ax, results_df):
        """Plot results summary"""
        ax.axis('off')
        ax.set_title("D. Results Summary", fontsize=12, weight='bold', pad=10)
        
        correct = results_df[results_df['correct'] == True]
        misspec = results_df[results_df['correct'] == False]
        
        y = 0.85
        
        if len(correct) > 0:
            ax.text(0.5, y, "Correct Specification:", fontsize=10, weight='bold',
                   ha='center', transform=ax.transAxes)
            y -= 0.08
            ax.text(0.5, y, f"SCM: {correct['scm_error'].mean():.1f}% ± {correct['scm_error'].std():.1f}%",
                   ha='center', fontsize=9, transform=ax.transAxes)
            y -= 0.06
            ax.text(0.5, y, f"ODE: {correct['ode_error'].mean():.1f}% ± {correct['ode_error'].std():.1f}%",
                   ha='center', fontsize=9, transform=ax.transAxes)
            y -= 0.12
        
        if len(misspec) > 0:
            ax.text(0.5, y, "Misspecification:", fontsize=10, weight='bold',
                   ha='center', transform=ax.transAxes)
            y -= 0.08
            ax.text(0.5, y, f"SCM: {misspec['scm_error'].mean():.1f}% ± {misspec['scm_error'].std():.1f}%",
                   ha='center', fontsize=9, transform=ax.transAxes)
            y -= 0.06
            ax.text(0.5, y, f"ODE: {misspec['ode_error'].mean():.1f}% ± {misspec['ode_error'].std():.1f}%",
                   ha='center', fontsize=9, transform=ax.transAxes)
            y -= 0.08
            
            scm_wins = misspec['scm_wins'].sum()
            win_rate = scm_wins / len(misspec) * 100
            
            ax.text(0.5, y, f"SCM Wins: {scm_wins}/{len(misspec)} ({win_rate:.0f}%)",
                   ha='center', fontsize=10, weight='bold',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='#E8F5E9' if win_rate == 100 else '#FFEBEE'))
    
    def _hierarchical_layout(self, G):
        """Create hierarchical layout for directed graph"""
        pos = {}
        
        try:
            # Compute shortest path distances from source
            layers_dict = {}
            for node in G.nodes():
                if node == self.k_in:
                    layers_dict[node] = 0
                else:
                    try:
                        layers_dict[node] = nx.shortest_path_length(G, self.k_in, node)
                    except:
                        layers_dict[node] = max(layers_dict.values(), default=0) + 1
            
            # Group by layer
            layer_groups = {}
            for node, layer in layers_dict.items():
                if layer not in layer_groups:
                    layer_groups[layer] = []
                layer_groups[layer].append(node)
            
            # Position nodes
            for layer, nodes in layer_groups.items():
                nodes_sorted = sorted(nodes)
                for i, node in enumerate(nodes_sorted):
                    x = layer * 3
                    y = (len(nodes_sorted) - 1) * 0.8 - i * 1.6
                    pos[node] = (x, y)
        
        except:
            pos = nx.spring_layout(G, k=2, seed=42)
        
        return pos
    
    def _print_pkn_summary(self):
        """Print PKN summary"""
        print(f"\n{'='*80}")
        print("PKN SUMMARY")
        print(f"{'='*80}")
        print(f"Variables ({len(self.vars)}): {', '.join(self.vars)}")
        print(f"Edges: {len(self.pkn['edges'])}")
        
        edge_types = {}
        for edge in self.pkn['edges']:
            etype = edge[2] if len(edge) >= 3 else 'activate'
            edge_types[etype] = edge_types.get(etype, 0) + 1
        
        for etype, count in edge_types.items():
            print(f"  {etype}: {count}")
        
        print(f"Exogenous: {self.k_in}")
        print(f"Inhibitions: {len(self.inhib)}")
        
        # Check for cycles
        G = nx.DiGraph()
        for edge in self.pkn['edges']:
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[1])
        
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                print(f"\nCycles detected: {len(cycles)}")
                for idx, cycle in enumerate(cycles[:3]):  # Show first 3
                    cycle_str = " → ".join(cycle + [cycle[0]])
                    print(f"  Cycle {idx+1}: {cycle_str}")
        except:
            print("\nNo cycles detected (DAG)")
    
    def _check_stability(self):
        """Check stability"""
        from ode_simulator import ODESimulator
        
        print(f"\n{'='*80}")
        print("STABILITY CHECK")
        print(f"{'='*80}")
        
        ode_sim = ODESimulator(self.pkn)
        all_stable = True
        
        for form in ['mass_action', 'mm', 'hill']:
            solver = ode_sim.build_solver(form, self.base_params[form])
            
            stable_count = 0
            for k in [0.8, 1.0, 1.5, 2.0]:
                ss = solver(k)
                if ss and max(ss.values()) < 100:
                    stable_count += 1
            
            rate = stable_count / 4 * 100
            print(f"  {form.upper():12s}: {stable_count}/4 ({rate:.0f}%)")
            
            if stable_count < 4:
                all_stable = False
        
        if all_stable:
            print("\n✅ All formulations stable")
        else:
            print("\n❌ Some formulations unstable")
        
        return all_stable
    
    def _check_saturation_conditions(self):
        """Check saturation"""
        from ode_simulator import ODESimulator
        
        print(f"\n{'='*80}")
        print("SATURATION CHECK")
        print(f"{'='*80}")
        
        ode_sim = ODESimulator(self.pkn)
        
        for form in ['mm', 'hill']:
            sat_vars = self.pkn.get('saturation', {}).get(form, [])
            if not sat_vars:
                continue
            
            solver = ode_sim.build_solver(form, self.base_params[form])
            ss = solver(1.0)
            
            if ss:
                threshold = self.base_params[form].get('Km' if form == 'mm' else 'K', 1.0)
                print(f"\n{form.upper()}:")
                for var in sat_vars:
                    if var in ss:
                        val = ss[var]
                        ratio = val / threshold
                        status = "✓ SAT" if ratio > 5 else "✗ NOT SAT"
                        print(f"  {var}: {val:.3f} / {threshold} = {ratio:.1f}× {status}")
    
    def _display_ode_models(self):
        """Display equations"""
        from equilibrium_generator import EquilibriumGenerator
        
        print(f"\n{'='*80}")
        print("EQUILIBRIUM EQUATIONS")
        print(f"{'='*80}")
        
        eq_gen = EquilibriumGenerator(self.pkn)
        
        for form in ['mass_action', 'mm', 'hill']:
            print(f"\n{form.upper()}:")
            eqs = eq_gen.generate_equations(form, with_inhibitions=False)
            
            for eq_name in sorted(eqs.keys()):
                print(f"  {eq_name}: {eqs[eq_name]} = 0")
    
    def _run_pcoa_analysis(self, debug=False):
        """Run PCOA and return structures"""
        from equilibrium_generator import EquilibriumGenerator
        from pcoa_structure_discovery import PCOADiscovery
        
        print(f"\n{'='*80}")
        print("PCOA STRUCTURE DISCOVERY")
        print(f"{'='*80}")
        
        eq_gen = EquilibriumGenerator(self.pkn)
        pcoa = PCOADiscovery(self.pkn)
        
        structures = {}
        
        for form in ['mass_action', 'mm', 'hill']:
            print(f"\n{form.upper()}:")
            
            eqs = eq_gen.generate_equations(form, with_inhibitions=False)
            structure, topo = pcoa.discover_structure(eqs, debug=debug)
            
            structures[form] = (structure, topo)
            
            if not debug:
                for var in self.vars:
                    parents = structure.get(var, [])
                    parent_str = ', '.join(parents) if parents else '[]'
                    print(f"  {var} ← [{parent_str}]")
        
        return structures
    
    def _compare_structures_diagnostic(self):
        """Print structure comparison"""
        if not hasattr(self, 'structures'):
            return
        
        print(f"\n{'='*80}")
        print("STRUCTURE COMPARISON")
        print(f"{'='*80}")
        
        differs = []
        for var in self.vars:
            ma = set(self.structures.get('mass_action', ({}, []))[0].get(var, []))
            mm = set(self.structures.get('mm', ({}, []))[0].get(var, []))
            hill = set(self.structures.get('hill', ({}, []))[0].get(var, []))
            
            if not (ma == mm == hill):
                differs.append(var)
        
        if differs:
            print(f"\n⚠️  Variables with different structures: {differs}")
            print("   This indicates saturation or nonlinear effects")
        else:
            print("\n✓ All formulations have identical structure")
    
    def visualize_for_publication(self, pcoa_structures, results_df=None):
        """Wrapper for easy publication figure creation"""
        return self.create_publication_figure(pcoa_structures, results_df)

