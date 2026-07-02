import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import itertools
import kmapper as km
import warnings
import sys
import os

# Create target output directory
output_dir = os.path.join('instants', 'c')
os.makedirs(output_dir, exist_ok=True)

# Υποστήριξη ελληνικών στο terminal των Windows
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# Γ.1 — Υλοποίηση Mapper εκ του Μηδενός

class CustomMapper:
    # Κατασκευαστής της κλάσης CustomMapper
    def __init__(self, n_intervals=10, overlap_frac=0.1, clustering_algo=None):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        if clustering_algo is None:
            self.clustering_algo = DBSCAN(eps=0.5, min_samples=2)
        else:
            self.clustering_algo = clustering_algo
            
    # Υπολογισμός των 1D διαστημάτων με επικάλυψη για τη συνάρτηση φιλτραρίσματος
    def _get_1d_intervals(self, filter_data, n, overlap):
        min_val = np.min(filter_data)
        max_val = np.max(filter_data)
        eps = 1e-8
        max_val += eps
        min_val -= eps
        L = max_val - min_val
        w = L / (1 + (n - 1) * (1 - overlap))
        step = w * (1 - overlap)
        
        intervals = []
        for i in range(n):
            start = min_val + i * step
            end = start + w
            intervals.append((start, end))
        return intervals

    # Κατασκευή του γραφήματος Mapper από τα δεδομένα εισόδου
    def fit(self, X, filter_values):
        X = np.array(X)
        filter_values = np.array(filter_values)
        
        is_1d = (filter_values.ndim == 1) or (filter_values.shape[1] == 1)
        f_dim = 1 if is_1d else filter_values.shape[1]
        
        if is_1d:
            f1 = filter_values.flatten()
            n_int = self.n_intervals if isinstance(self.n_intervals, int) else self.n_intervals[0]
            intervals_1d = self._get_1d_intervals(f1, n_int, self.overlap_frac)
            intervals = [[(s, e)] for s, e in intervals_1d]
        else:
            f1 = filter_values[:, 0]
            f2 = filter_values[:, 1]
            n_int = self.n_intervals if isinstance(self.n_intervals, tuple) else (self.n_intervals, self.n_intervals)
            int1 = self._get_1d_intervals(f1, n_int[0], self.overlap_frac)
            int2 = self._get_1d_intervals(f2, n_int[1], self.overlap_frac)
            intervals = list(itertools.product(int1, int2))
            
        nodes = []
        node_id_counter = 0
        
        for int_idx, interval_bounds in enumerate(intervals):
            in_interval = np.ones(len(X), dtype=bool)
            for d in range(f_dim):
                start, end = interval_bounds[d]
                f_d = filter_values.flatten() if f_dim == 1 else filter_values[:, d]
                in_interval &= (f_d >= start) & (f_d <= end)
                
            pts_idx = np.where(in_interval)[0]
            if len(pts_idx) == 0:
                continue
                
            X_subset = X[pts_idx]
            labels = self.clustering_algo.fit_predict(X_subset)
            
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label == -1: # Αγνοούμε το noise του DBSCAN
                    continue
                    
                cluster_pts = pts_idx[labels == label]
                nodes.append({
                    'id': node_id_counter,
                    'points': set(cluster_pts),
                    'size': len(cluster_pts),
                    'interval_idx': int_idx,
                    'mean_filter': np.mean(filter_values[cluster_pts], axis=0) if f_dim > 1 else np.mean(filter_values[cluster_pts])
                })
                node_id_counter += 1
                
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'], size=node['size'], points=list(node['points']), mean_filter=node['mean_filter'])
            
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if not nodes[i]['points'].isdisjoint(nodes[j]['points']):
                    weight = len(nodes[i]['points'].intersection(nodes[j]['points']))
                    G.add_edge(nodes[i]['id'], nodes[j]['id'], weight=weight)
                    
        return G

# Γ.2 — Εφαρμογή Mapper

# Δημιουργία των συνθετικών συνόλων δεδομένων (Figure-8 και Horse-shoe)
def generate_datasets_g2():
    np.random.seed(42)
    # 1. Figure-8 (σύνδεση δύο κύκλων)
    t1 = np.linspace(0, 2*np.pi, 300, endpoint=False)
    c1 = np.column_stack([np.cos(t1)-1, np.sin(t1)])
    c2 = np.column_stack([np.cos(t1)+1, np.sin(t1)])
    fig8 = np.vstack([c1, c2]) + np.random.normal(0, 0.05, (600, 2))
    
    # 2. Horse-shoe
    t2 = np.linspace(0, np.pi, 400)
    horseshoe = np.column_stack([np.cos(t2), np.sin(t2)]) + np.random.normal(0, 0.05, (400, 2))
    
    return {"Figure-8": fig8, "Horse-shoe": horseshoe}

# Υπολογισμός των τριών συναρτήσεων φιλτραρίσματος (PCA, L2-norm, Density)
def get_filter_functions(X):
    # 1. PCA 1η συνιστώσα
    f_pca = PCA(n_components=1).fit_transform(X).flatten()
    
    # 2. L2-norm
    f_l2 = np.linalg.norm(X, axis=1)
    
    # 3. Density estimation
    kde = KernelDensity(bandwidth=0.2).fit(X)
    f_density = np.exp(kde.score_samples(X))
    
    return {
        "PCA 1st Component": f_pca,
        "L2-norm": f_l2,
        "Density": f_density
    }

# Οπτικοποίηση του γραφήματος Mapper και αποθήκευση σε εικόνα
def visualize_mapper_nx(G, title, filename):
    plt.figure(figsize=(8, 6))
    if len(G.nodes) == 0:
        plt.title(f"{title} - No Nodes")
        plt.savefig(filename)
        plt.close()
        return
        
    pos = nx.spring_layout(G, seed=42)
    node_color = [G.nodes[n].get('mean_filter', 0) for n in G.nodes()]
    node_size = [G.nodes[n].get('size', 10) * 10 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

# Εκτέλεση των πειραμάτων και σύγκριση του custom Mapper με τον KeplerMapper
def run_g2():
    print("\n--- Γ.2: Εφαρμογή Mapper ---")
    datasets = generate_datasets_g2()
    
    clustering = DBSCAN(eps=0.3, min_samples=3)
    
    for ds_name, X in datasets.items():
        print(f"\nDataset: {ds_name}")
        filters = get_filter_functions(X)
        
        for f_name, f_val in filters.items():
            print(f"  Filter: {f_name}")
            
            #Custom Mapper
            custom_mapper = CustomMapper(n_intervals=10, overlap_frac=0.3, clustering_algo=clustering)
            G_custom = custom_mapper.fit(X, f_val)
            visualize_mapper_nx(G_custom, f"{ds_name} - {f_name} (Custom Mapper)", os.path.join(output_dir, f"g2_{ds_name.lower().replace('-','_')}_{f_name.lower().replace(' ','_')}_custom.png"))
            
            #kmapper verification
            mapper = km.KeplerMapper(verbose=0)
            graph = mapper.map(
                f_val, X,
                cover=km.Cover(n_cubes=10, perc_overlap=0.3),
                clusterer=clustering
            )
            html_filename = os.path.join(output_dir, f"g2_{ds_name.lower().replace('-','_')}_{f_name.lower().replace(' ','_')}_kmapper.html")
            mapper.visualize(graph, path_html=html_filename, title=f"{ds_name} - {f_name} (kmapper)")
            print(f"    Αποθηκεύτηκαν τα γραφήματα (Custom Image: instants/c/g2_..._custom.png & kmapper HTML: instants/c/g2_..._kmapper.html)")

# Ανάλυση ευαισθησίας του Mapper ως προς τον αριθμό διαστημάτων και το ποσοστό επικάλυψης
def sensitivity_analysis():
    print("\n--- Γ.2: Sensitivity Analysis ---")
    datasets = generate_datasets_g2()
    X = datasets["Figure-8"]
    f_val = PCA(n_components=1).fit_transform(X).flatten()
    
    n_intervals_list = [5, 10, 15, 20]
    overlap_frac_list = [0.1, 0.2, 0.3, 0.5]
    
    cc_matrix = np.zeros((len(n_intervals_list), len(overlap_frac_list)))
    
    for i, n_int in enumerate(n_intervals_list):
        for j, overlap in enumerate(overlap_frac_list):
            clustering = DBSCAN(eps=0.3, min_samples=3)
            custom_mapper = CustomMapper(n_intervals=n_int, overlap_frac=overlap, clustering_algo=clustering)
            G = custom_mapper.fit(X, f_val)
            
            if len(G.nodes) > 0:
                cc = nx.number_connected_components(G)
            else:
                cc = 0
            cc_matrix[i, j] = cc
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(cc_matrix, annot=True, fmt=".0f", cmap="YlGnBu", 
                xticklabels=overlap_frac_list, yticklabels=n_intervals_list)
    plt.xlabel("Overlap Fraction")
    plt.ylabel("Number of Intervals")
    plt.title("Sensitivity Analysis: Number of Connected Components\n(Figure-8, PCA Filter)")
    plt.savefig(os.path.join(output_dir, "g2_sensitivity_heatmap.png"))
    plt.close()
    print("Αποθηκεύτηκε: instants/c/g2_sensitivity_heatmap.png")

if __name__ == "__main__":
    run_g2()
    sensitivity_analysis()
