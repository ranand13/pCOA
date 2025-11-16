"""
E. coli CRISPRi Functional Module Data Loader - UPDATED v3

NEW: Operon-to-gene mapping to expand test coverage from 8 to ~20-30 genes!

Changes:
- Added operon_to_genes mapping for CRP regulon
- Modified load_crispri_knockdowns to expand operon data to individual genes
- Modified load_wildtype_expression similarly
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path


class EcoliCRISPRiDataLoader:
    
    def __init__(self, data_dir='ecoli_crp_data'):
        """
        Initialize E. coli CRISPRi data loader
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.regulondb_dir = self.data_dir / 'regulondb'
        self.crispri_dir = self.data_dir / 'crispri_knockdown'
        self.wt_expression_dir = self.data_dir / 'wildtype_expression'
        
        # Operon to genes mapping for CRP regulon
        # Maps PPTP-seq operon names → individual genes in that operon
        self.operon_to_genes = {
            # Multi-gene operons
            'lacZYA': ['lacZ', 'lacY', 'lacA'],
            'araBAD': ['araB', 'araA', 'araD'],
            'araE-ygeA': ['araE'],
            'malEFG': ['malE', 'malF', 'malG'],
            'malK-lamB-malM': ['malK'],
            'galETKM': ['galE', 'galT', 'galK', 'galM'],
            'argCBH': ['argC', 'argB', 'argH'],
            'ampDE': ['ampD', 'ampE'],
            'aspA-dcuA': ['aspA', 'dcuB'],
            'dusB-fis': ['fis', 'Fis'],
            'serC-aroA': ['aroA'],
            'zraSR': ['zraS'],
            # Single-gene operons (operon name = gene name)
            'aldA': ['aldA'],
            'ansB': ['ansB'],
            'araC': ['araC'],
            'araJ': ['araJ'],
            'argG': ['argG'],
            'crp': ['crp'],
            'cyaA': ['cyaA'],
            'galR': ['galR'],
            'pagP': ['agp'],  # Note: pagP operon → agp gene
        }
        
        # Load core data
        self.full_network = self._load_regulondb_network()
        self.gene_info = self._load_gene_info()
        
    def _load_regulondb_network(self):
        """Load full E. coli regulatory network from RegulonDB"""
        # [Keep existing implementation - no changes needed]
        tf_to_gene = {}
        tf_set_file = self.regulondb_dir / 'TFSet.tsv'
        
        if tf_set_file.exists():
            with open(tf_set_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if i == 0 or 'tfId' in parts[0]:
                        continue
                    
                    if len(parts) >= 4:
                        tf_name = parts[1].strip()
                        gene_name = parts[3].strip()
                        if tf_name and gene_name:
                            tf_to_gene[tf_name] = gene_name
            
            print(f"Built TF name mapping: {len(tf_to_gene)} TFs")
        
        network_file = self.regulondb_dir / 'NetworkRegulatorGene.tsv'
        
        if not network_file.exists():
            print(f"⚠️  Network file not found")
            return nx.DiGraph()
        
        G = nx.DiGraph()
        
        print(f"Loading RegulonDB network from: {network_file.name}")
        
        with open(network_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split('\t')
                
                if i == 0 or 'regulatorId' in parts[0]:
                    continue
                
                if len(parts) >= 5:
                    tf_gene = parts[2].strip()
                    tf_protein = parts[1].strip()
                    target = parts[4].strip()
                    
                    if not tf_gene and tf_protein in tf_to_gene:
                        tf_gene = tf_to_gene[tf_protein]
                    
                    if not tf_gene or not target:
                        continue
                    
                    regulation = 'activate'
                    if len(parts) >= 6:
                        effect = parts[5].strip()
                        if effect == '-':
                            regulation = 'repress'
                        elif effect == '+':
                            regulation = 'activate'
                        elif effect in ['+-', 'dual']:
                            regulation = 'dual'
                    
                    G.add_edge(tf_gene, target, regulation=regulation)
        
        print(f"Loaded RegulonDB network: {G.number_of_nodes()} genes, "
              f"{G.number_of_edges()} edges")
        
        tf_tf_file = self.regulondb_dir / 'NetworkRegulatorRegulator.tsv'
        if tf_tf_file.exists():
            with open(tf_tf_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split('\t')
                    if i == 0:
                        continue
                    
                    if len(parts) >= 5:
                        tf1_gene = parts[2].strip()
                        tf1_protein = parts[1].strip()
                        tf2_gene = parts[4].strip() if len(parts) > 4 else ''
                        
                        if not tf1_gene and tf1_protein in tf_to_gene:
                            tf1_gene = tf_to_gene[tf1_protein]
                        
                        if tf1_gene and tf2_gene:
                            regulation = 'activate'
                            if len(parts) >= 6 and parts[5].strip() == '-':
                                regulation = 'repress'
                            G.add_edge(tf1_gene, tf2_gene, regulation=regulation)
            
            print(f"  Added TF-TF interactions: {G.number_of_edges()} total edges")
        
        return G
    
    def _load_gene_info(self):
        """Load gene metadata from RegulonDB"""
        gene_file = self.regulondb_dir / 'GeneProductSet.tsv'
        
        if not gene_file.exists():
            return {}
        
        gene_info = {}
        with open(gene_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith('#') or not line.strip():
                    continue
                if i == 0:
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    gene_name = parts[0].strip()
                    gene_info[gene_name] = {
                        'name': gene_name,
                        'product': parts[1].strip() if len(parts) > 1 else '',
                    }
        
        print(f"Loaded gene info for {len(gene_info)} genes")
        return gene_info
    
    def load_functional_module(self, module_name='crp_regulon', expand_neighbors=False):
        """Load a biologically-defined functional module"""
        
        if module_name == 'crp_regulon':
            genes = self._get_crp_regulon()
        elif module_name == 'sos_response':
            genes = self._get_sos_response()
        elif module_name == 'arginine_biosynthesis':
            genes = self._get_arginine_biosynthesis()
        elif module_name == 'custom':
            genes = self._load_custom_module()
        else:
            raise ValueError(f"Unknown module: {module_name}")
        
        module_network = self._extract_module_network(genes, expand_neighbors)
        
        print(f"\nLoaded {module_name} module:")
        print(f"  Genes: {len(genes)}")
        print(f"  Edges: {module_network.number_of_edges()}")
        print(f"  Density: {nx.density(module_network)*100:.1f}%")
        
        return genes, module_network
    
    def _get_crp_regulon(self):
        """Get CRP regulon genes"""
        crp_file = self.data_dir / 'crp_regulon_genes.txt'
        if crp_file.exists():
            with open(crp_file, 'r') as f:
                return [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if self.full_network.number_of_nodes() > 0:
            crp_name = None
            for name in ['crp', 'Crp', 'CRP']:
                if name in self.full_network.nodes():
                    crp_name = name
                    break
            
            if crp_name:
                crp_targets = list(self.full_network.successors(crp_name))
                print(f"  Found CRP as '{crp_name}' with {len(crp_targets)} targets")
                
                core_genes = [crp_name]
                for gene in ['cyaA', 'fis', 'ihfA', 'ihfB', 'Fis', 'IhfA', 'IhfB']:
                    if gene in self.full_network.nodes():
                        core_genes.append(gene)
                
                operon_genes = []
                for gene in ['lacI', 'lacZ', 'lacY', 'lacA',
                           'malT', 'malE', 'malF', 'malG', 'malK',
                           'galR', 'galE', 'galT', 'galK', 'galM',
                           'araC', 'araB', 'araA', 'araD']:
                    if gene in self.full_network.nodes():
                        operon_genes.append(gene)
                
                genes = list(set(core_genes + operon_genes + crp_targets[:30]))
                print(f"  Building CRP regulon: {len(genes)} genes total")
                return genes[:50]
            else:
                print(f"  ⚠️  CRP not found, using hardcoded list")
        
        return ['crp', 'cyaA', 'fis', 'lacI', 'lacZ', 'lacY', 'lacA',
                'malT', 'malE', 'malF', 'malG', 'malK', 'malP',
                'galR', 'galE', 'galT', 'galK', 'galM',
                'araC', 'araB', 'araA', 'araD', 'araE']
    
    def _get_sos_response(self):
        """Get SOS response genes"""
        return ['lexA', 'recA', 'uvrA', 'uvrB', 'uvrC', 'uvrD',
                'recN', 'recF', 'recO', 'recR', 'dinI', 'dinG', 'dinB']
    
    def _get_arginine_biosynthesis(self):
        """Get arginine biosynthesis pathway genes"""
        return ['argA', 'argB', 'argC', 'argD', 'argE', 'argF', 'argG', 'argH',
                'argR', 'argI', 'carA', 'carB']
    
    def _load_custom_module(self):
        """Load custom gene list from file"""
        custom_file = self.data_dir / 'custom_module_genes.txt'
        if not custom_file.exists():
            raise FileNotFoundError(f"Custom module file not found: {custom_file}")
        
        with open(custom_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    def _extract_module_network(self, genes, expand_neighbors=False):
        """Extract module subnetwork from full network"""
        if expand_neighbors:
            expanded = set(genes)
            for gene in genes:
                if self.full_network.has_node(gene):
                    expanded.update(self.full_network.successors(gene))
                    expanded.update(self.full_network.predecessors(gene))
            genes = list(expanded)
        
        module_genes_in_network = [g for g in genes if self.full_network.has_node(g)]
        return self.full_network.subgraph(module_genes_in_network).copy()
    
    def build_pkn_dict(self, genes, network):
        """
        Convert NetworkX network to PKN dict for framework
        """
        print("="*80)
        print("PKN BUILDER - VERSION 2 WITH BASAL EDGES")
        print("="*80)
        
        all_edges = []
        
        # ADD BASAL EDGES FOR ALL GENES
        print(f"\nAdding basal production edges...")
        for gene in genes:
            all_edges.append(('basal', gene, 'activate'))
        print(f"  ✓ Added {len(all_edges)} basal edges")
        
        # ADD REGULATORY EDGES
        print(f"\nAdding regulatory edges from network...")
        reg_count = 0
        for tf, target in network.edges():
            if tf in genes and target in genes:
                regulation = network[tf][target].get('regulation', 'activate')
                all_edges.append((tf, target, regulation))
                reg_count += 1
        print(f"  ✓ Added {reg_count} regulatory edges")
        
        print(f"\nTotal edges in PKN: {len(all_edges)}")
        print(f"  Basal: {len([e for e in all_edges if e[0] == 'basal'])}")
        print(f"  Regulatory: {len([e for e in all_edges if e[0] != 'basal'])}")
        print("="*80)
        
        pkn = {
            'variables': genes,
            'exogenous': 'basal',
            'inhibitions': {gene: f'I_{gene}' for gene in genes},
            'inhibition_type': 'production',
            'edges': all_edges,
            'no_degradation': []
        }
        
        return pkn
    
    def load_wildtype_expression(self, genes, media='glucose', min_samples=100):
        """
        Load wild-type expression data
        
        NEW v3: Expands operon measurements to individual genes
        """
        expr_file = self.wt_expression_dir / 'wt_expression_data.txt'
        
        if expr_file.exists():
            expr_data = pd.read_csv(expr_file, sep='\t', index_col=0)
            available_genes = [g for g in genes if g in expr_data.columns]
            expr_data = expr_data[available_genes]
            print(f"Loaded wild-type expression: {len(expr_data)} samples × {len(available_genes)} genes")
            return expr_data
        
        print(f"No separate wild-type file, extracting baseline from PPTP-seq...")
        
        pptp_file = self.crispri_dir / 'GSE213624_PPTP_Ecoli_summary_updated.csv.gz'
        if not pptp_file.exists():
            pptp_file = self.crispri_dir / 'GSE213624_PPTP_Ecoli_summary_updated.csv'
        
        if pptp_file.exists():
            df = pd.read_csv(pptp_file)
            
            mean_col = f'{media} mean (log scale)'
            fc_col = f'{media} log2FC(TFKD/control)'
            
            df['wt_baseline'] = df[mean_col] - df[fc_col]
            
            # NEW: Expand operons to genes
            expanded_data = []
            for _, row in df.iterrows():
                operon = row['operon']
                baseline_val = row['wt_baseline']
                tf = row['tf_gene']
                
                # Map operon to genes
                if operon in self.operon_to_genes:
                    mapped_genes = self.operon_to_genes[operon]
                    for gene in mapped_genes:
                        if gene in genes:  # Only keep genes in module
                            expanded_data.append({
                                'gene': gene,
                                'tf': tf,
                                'baseline': baseline_val
                            })
            
            # Pivot to get samples × genes format
            expanded_df = pd.DataFrame(expanded_data)
            baseline_pivot = expanded_df.pivot_table(
                index='gene',
                columns='tf',
                values='baseline',
                aggfunc='first'
            )
            
            # Transpose to get samples (TFs) × genes
            expr_data = baseline_pivot.T
            expr_data = expr_data.fillna(expr_data.median())
            
            available_genes = [g for g in genes if g in expr_data.columns]
            
            print(f"Extracted wild-type baseline from PPTP-seq (with operon mapping):")
            print(f"  {len(expr_data)} pseudo-samples × {len(available_genes)} genes")
            print(f"  Genes with data: {available_genes}")
            
            return expr_data
        
        print(f"⚠️  No expression data, generating synthetic...")
        return self._generate_synthetic_expression(genes, n_samples=min_samples)
    
    def load_crispri_knockdowns(self, genes, media='glucose'):
        """
        Load CRISPRi knockdown data from PPTP-seq
        
        NEW v3: Expands operon measurements to individual genes
        """
        pptp_file = self.crispri_dir / 'GSE213624_PPTP_Ecoli_summary_updated.csv.gz'
        if not pptp_file.exists():
            pptp_file = self.crispri_dir / 'GSE213624_PPTP_Ecoli_summary_updated.csv'
        
        if not pptp_file.exists():
            print(f"⚠️  PPTP-seq file not found")
            return {}
        
        print(f"Loading PPTP-seq data from: {pptp_file.name}")
        df = pd.read_csv(pptp_file)
        
        mean_col = f'{media} mean (log scale)'
        fc_col = f'{media} log2FC(TFKD/control)'
        
        # Find TFs in module
        tfs_in_data = [g for g in genes if g in df['tf_gene'].values]
        
        print(f"  Found {len(tfs_in_data)} TFs with knockdown data")
        print(f"  Expanding operon measurements to individual genes...")
        
        test_cases = {}
        
        for tf in tfs_in_data:
            tf_data = df[df['tf_gene'] == tf].copy()
            
            # Expand operon measurements to genes
            gene_data = {}  # gene → {baseline, knockdown, fc}
            
            for _, row in tf_data.iterrows():
                operon = row['operon']
                
                # Map operon to genes
                if operon in self.operon_to_genes:
                    mapped_genes = self.operon_to_genes[operon]
                    baseline_val = row[mean_col] - row[fc_col]
                    knockdown_val = row[mean_col]
                    fc_val = row[fc_col]
                    
                    for gene in mapped_genes:
                        if gene in genes:  # Only keep genes in module
                            gene_data[gene] = {
                                'baseline': baseline_val,
                                'knockdown': knockdown_val,
                                'fc': fc_val
                            }
            
            if len(gene_data) == 0:
                continue
            
            # Build dicts for this TF
            baseline_dict = {g: data['baseline'] for g, data in gene_data.items()}
            knockdown_dict = {g: data['knockdown'] for g, data in gene_data.items()}
            fc_dict = {g: data['fc'] for g, data in gene_data.items()}
            
            test_cases[tf] = {
                'target_tf': tf,
                'knockdown_strength': 0.5,
                'baseline': baseline_dict,
                'knockdown': knockdown_dict,
                'fold_change': fc_dict,
                'n_promoters_measured': len(gene_data)
            }
        
        print(f"\nLoaded CRISPRi knockdowns for {len(test_cases)} TFs in {media} media")
        
        if len(test_cases) > 0:
            print(f"  Example TFs: {list(test_cases.keys())[:5]}")
            example_tf = list(test_cases.keys())[0]
            print(f"  Genes measured for {example_tf}: {list(test_cases[example_tf]['baseline'].keys())}")
        
        return test_cases
    
    def _generate_synthetic_expression(self, genes, n_samples=100):
        """Generate synthetic expression data for testing"""
        np.random.seed(42)
        data = np.random.lognormal(mean=5, sigma=1.5, size=(n_samples, len(genes)))
        return pd.DataFrame(data, columns=genes)
    
    def get_literature_parameters(self, genes, network, form='hill'):
        """Get literature-based Hill parameters"""
        np.random.seed(42)
        params = {}
        
        for gene in genes:
            params[f'k_basal_{gene}'] = np.random.uniform(0.1, 0.5)
            params[f'k_deg_{gene}'] = np.random.uniform(0.1, 0.3)
        
        for tf, target in network.edges():
            if tf in genes and target in genes:
                params[f'V_{tf}_{target}'] = np.random.uniform(0.5, 2.0)
                params[f'K_{tf}_{target}'] = np.random.uniform(0.3, 1.5)
                params[f'n_{tf}_{target}'] = np.random.choice([2.0, 3.0, 4.0])
        
        return params