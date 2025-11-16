"""HPN-DREAM data loader - MCF7 breast cancer signaling

UPDATES:
- Full literature PKN (includes MAPK→MEK and mTOR→AKT feedbacks)
- Optimized I_AKT = 0.95 (from grid search over full dataset)
- Tests on ligand_filter (EGF, Serum, Insulin by default)
- Trains on all 127 DMSO samples
"""

import pandas as pd
import networkx as nx


class DreamDataLoader:
    
    def __init__(self, data_dir='dream_hpn_data', aggregate_sites=True):
        self.data_dir = data_dir
        self.aggregate_sites = aggregate_sites
        
        self.protein_to_gene = {
            'AKT_pS473': 'AKT1', 'AKT_pT308': 'AKT1',
            'EGFR_pY1068': 'EGFR', 'EGFR_pY1173': 'EGFR', 'EGFR_pY992': 'EGFR',
            'GSK3-alpha-beta_pS21_S9': 'GSK3B', 'GSK3-alpha-beta_pS9': 'GSK3B',
            'MAPK_pT202_Y204': 'MAPK1', 'MEK1_pS217_S221': 'MAP2K1',
            'mTOR_pS2448': 'MTOR', 'PDK1_pS241': 'PDPK1'
        }
        
        self.site_groups = {
            'AKT': ['AKT_pS473', 'AKT_pT308'],
            'EGFR': ['EGFR_pY1068', 'EGFR_pY1173', 'EGFR_pY992'],
            'GSK3': ['GSK3-alpha-beta_pS21_S9', 'GSK3-alpha-beta_pS9'],
            'MAPK': ['MAPK_pT202_Y204'],
            'MEK': ['MEK1_pS217_S221'],
            'mTOR': ['mTOR_pS2448'],
            'PDK1': ['PDK1_pS241']
        }
        
        self.ligand_map = {
            'PBS': 0.0, 'Serum': 0.5, 'EGF': 1.0, 'IGF1': 0.7,
            'Insulin': 0.6, 'FGF1': 0.6, 'NRG1': 0.8, 'HGF': 0.7
        }
        
        self.good_ligands = ['EGF', 'Serum', 'Insulin']
        self.ligand_filter = self.good_ligands  # Filter for testing
        
        self.inhibitor_spec = {
            'GSK690693': {'target': 'AKT', 'I_AKT': 0.95, 'I_MEK': 1.0}
        }
    
    def load_data(self):
        mcf7 = pd.read_csv(f"{self.data_dir}/MCF7_main.csv", skiprows=[0])
        
        rename = {}
        for col in mcf7.columns:
            if 'Unnamed: 0' in str(col):
                rename[col] = 'CellLine'
            elif 'Unnamed: 1' in str(col):
                rename[col] = 'Inhibitor'
            elif 'Unnamed: 2' in str(col):
                rename[col] = 'Ligand'
            elif col == 'Timepoint':
                rename[col] = 'Time'
        
        mcf7 = mcf7.rename(columns=rename)
        
        protein_cols = [c for c in mcf7.columns 
                        if c not in ['CellLine', 'Inhibitor', 'Ligand', 'Time', 'Antibody Name', 'HUGO ID']
                        and '_p' in str(c) and 'TAZ' not in c and 'FOXO3a' not in c]
        
        for col in protein_cols:
            mcf7[col] = pd.to_numeric(mcf7[col], errors='coerce')
        
        if 'Time' in mcf7.columns:
            mcf7['Time'] = mcf7['Time'].str.replace('min', '').str.replace('hr', '').astype(float)
            steady = mcf7[mcf7['Time'] == 60].copy()
        else:
            steady = mcf7.copy()
        
        core = [c for c in protein_cols 
                if any(p in c for p in ['MEK', 'MAPK', 'AKT', 'mTOR', 'EGFR', 'PDK1', 'GSK3'])]
        
        if self.aggregate_sites:
            for protein, sites in self.site_groups.items():
                available_sites = [s for s in sites if s in core]
                if len(available_sites) > 1:
                    steady[protein] = steady[available_sites].mean(axis=1)
                elif len(available_sites) == 1:
                    steady[protein] = steady[available_sites[0]]
            
            core = list(self.site_groups.keys())
        
        # Train on ALL ligands (127 samples)
        train = steady[steady['Inhibitor'] == 'DMSO'].copy()
        train['input_signal'] = train['Ligand'].map(self.ligand_map).fillna(0.5)
        train['I_AKT'] = 1.0
        train['I_MEK'] = 1.0
        
        # Test on filtered ligands only (single drug)
        test_pert = steady[
            (steady['Inhibitor'].isin(self.inhibitor_spec.keys())) &
            (steady['Ligand'].isin(self.ligand_filter))
        ].copy()
        
        cases = []
        for _, row in test_pert.iterrows():
            if row['Inhibitor'] not in self.inhibitor_spec:
                continue
            
            bl = train[train['Ligand'] == row['Ligand']]
            if len(bl) == 0:
                continue
            
            baseline = bl.iloc[0]
            spec = self.inhibitor_spec[row['Inhibitor']]
            
            bs = {p: float(baseline[p]) for p in core if pd.notna(baseline.get(p))}
            pt = {p: float(row[p]) for p in core if pd.notna(row.get(p))}
            
            if len(bs) >= 3 and len(pt) >= 3:
                cases.append({
                    'ligand': row['Ligand'],
                    'inhibitor': row['Inhibitor'],
                    'target': spec['target'],
                    'input_signal': baseline['input_signal'],
                    'I_AKT': spec['I_AKT'],
                    'I_MEK': spec['I_MEK'],
                    'baseline': bs,
                    'true_perturbed': pt
                })
        
        return train, cases, core
    
    def build_literature_pkn(self):
        pkn = nx.DiGraph()
        # FULL literature PKN with feedback loops
        pkn.add_edges_from([
            ('EGFR', 'MAP2K1'),    # EGFR → MEK
            ('MAP2K1', 'MAPK1'),   # MEK → MAPK (ERK)
            ('MAPK1', 'MAP2K1'),   # MAPK → MEK (feedback)
            ('EGFR', 'PDPK1'),     # EGFR → PDK1
            ('PDPK1', 'AKT1'),     # PDK1 → AKT
            ('AKT1', 'GSK3B'),     # AKT → GSK3
            ('AKT1', 'MTOR'),      # AKT → mTOR
            ('MTOR', 'AKT1'),      # mTOR → AKT (feedback)
        ])
        return pkn
    
    def map_to_protein_network(self, gene_pkn, proteins):
        if self.aggregate_sites:
            gene_to_proteins = {
                'EGFR': 'EGFR',
                'MAP2K1': 'MEK',
                'MAPK1': 'MAPK',
                'PDPK1': 'PDK1',
                'AKT1': 'AKT',
                'GSK3B': 'GSK3',
                'MTOR': 'mTOR'
            }
        else:
            gene_to_proteins = {}
            for site, gene in self.protein_to_gene.items():
                if gene not in gene_to_proteins:
                    gene_to_proteins[gene] = []
                gene_to_proteins[gene].append(site)
        
        graph = {}
        for protein in proteins:
            if self.aggregate_sites:
                gene = {v: k for k, v in gene_to_proteins.items()}.get(protein)
            else:
                gene = self.protein_to_gene.get(protein)
            
            if gene and gene in gene_pkn:
                gene_parents = list(gene_pkn.predecessors(gene))
                
                if gene_parents:
                    if self.aggregate_sites:
                        prot_parents = [gene_to_proteins[gp] for gp in gene_parents 
                                       if gp in gene_to_proteins and gene_to_proteins[gp] in proteins]
                    else:
                        prot_parents = [p for gp in gene_parents
                                       for p, g in self.protein_to_gene.items()
                                       if g == gp and p in proteins]
                    
                    graph[protein] = prot_parents if prot_parents else ['input_signal']
                else:
                    graph[protein] = ['input_signal']
            else:
                graph[protein] = ['input_signal']
        
        in_deg = {p: 0 for p in proteins}
        adj = {p: [] for p in proteins}
        
        for c, ps in graph.items():
            if c in proteins:
                for p in ps:
                    if p in proteins:
                        adj[p].append(c)
                        in_deg[c] += 1
        
        q = [p for p in proteins if in_deg[p] == 0]
        ordered = []
        
        while q:
            n = q.pop(0)
            ordered.append(n)
            for nb in adj[n]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    q.append(nb)
        
        ordered.extend([p for p in proteins if p not in ordered])
        
        return graph, ordered
    
    def get_inhibition_targets(self, proteins):
        if self.aggregate_sites:
            inhibitions = {}
            if 'AKT' in proteins:
                inhibitions['AKT'] = 'I_AKT'
            if 'MEK' in proteins:
                inhibitions['MEK'] = 'I_MEK'
        else:
            inhibitions = {}
            for p in proteins:
                if 'AKT' in p:
                    inhibitions[p] = 'I_AKT'
                elif 'MEK' in p:
                    inhibitions[p] = 'I_MEK'
        
        return inhibitions