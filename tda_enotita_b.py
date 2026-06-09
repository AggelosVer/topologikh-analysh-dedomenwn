import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
import persim
import gudhi
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sparse
from scipy.linalg import eigh
import warnings
import sys

# Υποστήριξη ελληνικών στο terminal των Windows
sys.stdout.reconfigure(encoding='utf-8')

# Αγνοούμε κάποια warnings από το persim/gudhi
warnings.filterwarnings('ignore')

# =============================================================================
# Β.1 — Persistent Homology από Νέφη Σημείων
# =============================================================================

def generate_datasets_b1():
    np.random.seed(42)
    
    # 1. Τυχαία σημεία στο R2
    pts_random = np.random.uniform(0, 1, (100, 2))
    
    # 2. Κύκλος με θόρυβο
    t = np.linspace(0, 2*np.pi, 100, endpoint=False)
    pts_circle = np.column_stack([np.cos(t), np.sin(t)]) + np.random.normal(0, 0.1, (100, 2))
    
    # 3. Τόρος T2 στο R3
    theta = np.random.uniform(0, 2*np.pi, 150)
    phi = np.random.uniform(0, 2*np.pi, 150)
    R, r = 2, 0.5
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    pts_torus = np.column_stack([x, y, z])
    
    # 4. Σφαίρα S2
    z_s = np.random.uniform(-1, 1, 150)
    phi_s = np.random.uniform(0, 2*np.pi, 150)
    x_s = np.sqrt(1 - z_s**2) * np.cos(phi_s)
    y_s = np.sqrt(1 - z_s**2) * np.sin(phi_s)
    pts_sphere = np.column_stack([x_s, y_s, z_s])
    
    return {
        "Random R2": pts_random,
        "Noisy Circle": pts_circle,
        "Torus T2": pts_torus,
        "Sphere S2": pts_sphere
    }

def run_b1():
    print("\n--- Β.1: Persistent Homology από Νέφη Σημείων ---")
    datasets = generate_datasets_b1()
    
    for name, pts in datasets.items():
        print(f"Υπολογισμός για: {name}")
        # Χρησιμοποιούμε ripser για υπολογισμό H0, H1, H2
        res = ripser(pts, maxdim=2)
        dgms = res['dgms']
        
        plt.figure(figsize=(14, 5))
        
        # 1. Persistence Diagram
        plt.subplot(1, 2, 1)
        persim.plot_diagrams(dgms, show=False)
        plt.title(f"{name} - Persistence Diagram")
        
        # 2. Barcode Plot (Custom matplotlib plot για ασφάλεια)
        ax = plt.subplot(1, 2, 2)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        y_pos = 0
        
        # Εύρεση max death (εξαιρώντας το άπειρο) για οπτικοποίηση
        max_d = 0
        for dgm in dgms:
            for b, d in dgm:
                if not np.isinf(d) and d > max_d:
                    max_d = d
        vis_inf = max_d * 1.2 if max_d > 0 else 2.0
        
        for dim, dgm in enumerate(dgms):
            for b, d in dgm:
                if np.isinf(d):
                    d = vis_inf
                ax.plot([b, d], [y_pos, y_pos], color=colors[dim], lw=2)
                y_pos += 1
                
        ax.set_yticks([])
        ax.set_xlabel('Filtration Radius')
        ax.set_title(f"{name} - Barcode")
        
        # Legend
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=colors[dim], lw=2) for dim in range(len(dgms))]
        ax.legend(custom_lines, [f'H{dim}' for dim in range(len(dgms))], loc='lower right')
        
        plt.tight_layout()
        filename = f"b1_{name.replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Αποθηκεύτηκε: {filename}")

# =============================================================================
# Β.2 — Robustness σε Gaussian Θόρυβο
# =============================================================================

def run_b2():
    print("\n--- Β.2: Robustness σε Gaussian Θόρυβο ---")
    # Βασικό σχήμα: Καθαρός κύκλος
    t = np.linspace(0, 2*np.pi, 100, endpoint=False)
    base_circle = np.column_stack([np.cos(t), np.sin(t)])
    
    # Baseline diagram (H1)
    res_base = ripser(base_circle, maxdim=1)
    dgm_base = res_base['dgms'][1]
    
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.5]
    bottleneck_dists = []
    wasserstein_dists = []
    
    for sigma in sigmas:
        noisy_circle = base_circle + np.random.normal(0, sigma, base_circle.shape)
        res_noisy = ripser(noisy_circle, maxdim=1)
        dgm_noisy = res_noisy['dgms'][1]
        
        # Υπολογισμός αποστάσεων (για H1)
        # Χρειάζεται clean if dgm is empty, αλλά για κύκλο με αυτό το θόρυβο λογικά θα έχει H1.
        if len(dgm_noisy) == 0:
            dgm_noisy = np.array([[0, 0]]) # dummy για αποφυγή error
            
        bd = persim.bottleneck(dgm_base, dgm_noisy)
        wd = persim.wasserstein(dgm_base, dgm_noisy, matching=False)
        
        bottleneck_dists.append(bd)
        wasserstein_dists.append(wd)
        print(f"Sigma: {sigma:<5} | Bottleneck: {bd:.4f} | Wasserstein: {wd:.4f}")
        
    # Γράφημα απόστασης
    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, bottleneck_dists, marker='o', linestyle='-', label='Bottleneck Distance (L_inf)')
    plt.plot(sigmas, wasserstein_dists, marker='s', linestyle='--', label='Wasserstein Distance (W_2)')
    plt.xlabel('Noise Level (Sigma)')
    plt.ylabel('Distance')
    plt.title('Robustness of H1 Persistence Diagrams to Gaussian Noise')
    plt.legend()
    plt.grid(True)
    plt.savefig("b2_robustness.png")
    plt.close()
    print("Αποθηκεύτηκε: b2_robustness.png")

# =============================================================================
# Β.3 — Persistent Homology σε Γράφους
# =============================================================================

def clean_dgm(dgm, max_val):
    """Καθαρισμός diagram από άπειρα deaths για χρήση σε Persim Imager"""
    if len(dgm) == 0:
        return np.empty((0,2))
    dgm = np.array(dgm)
    dgm[dgm == np.inf] = max_val
    # Κρατάμε μόνο σημεία όπου το persistence > 0
    dgm = dgm[dgm[:, 1] > dgm[:, 0]]
    return dgm

def run_b3():
    print("\n--- Β.3: Persistent Homology σε Γράφους ---")
    np.random.seed(42)
    
    # Δημιουργία σημείων σε σχήμα κύκλου για να έχουμε έναν ουσιαστικό τοπολογικό κύκλο στο γράφο
    t = np.linspace(0, 2*np.pi, 30, endpoint=False)
    pts = np.column_stack([np.cos(t), np.sin(t)]) + np.random.normal(0, 0.05, (30, 2))
    dist_matrix = squareform(pdist(pts))
    
    # --- Υλοποίηση: Clique Complex & Sublevel Filtration (βάρη ακμών) ---
    st_edge = gudhi.SimplexTree()
    for i in range(len(pts)):
        st_edge.insert([i], filtration=0.0)
        
    threshold = 0.5
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if dist_matrix[i, j] < threshold:
                # Βάρος ακμής = γεωμετρική απόσταση
                st_edge.insert([i, j], filtration=dist_matrix[i, j])
                
    st_edge.expansion(2) # flag complex / clique complex up to triangles
    st_edge.persistence()
    print("1/2: Υπολογίστηκε το Sublevel Edge Filtration (Clique Complex).")
    
    # --- Υλοποίηση: Heat Kernel Signature (HKS) Filtration ---
    # 1. Υπολογισμός Graph Laplacian (L = D - W)
    W = np.zeros((len(pts), len(pts)))
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if dist_matrix[i, j] < threshold:
                W[i, j] = W[j, i] = np.exp(- (dist_matrix[i, j]**2) / 0.05)
                
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    # Ιδιοτιμές/Ιδιοδιανύσματα
    evals, evecs = eigh(L)
    
    # HKS
    t_hks = 5.0
    hks = np.zeros(len(pts))
    for i in range(len(pts)):
        hks[i] = np.sum(np.exp(-evals * t_hks) * (evecs[i, :]**2))
    
    # Κανονικοποίηση
    hks = hks / np.max(hks)
    
    st_hks = gudhi.SimplexTree()
    # Κόμβοι με filtration = HKS value
    for i in range(len(pts)):
        st_hks.insert([i], filtration=hks[i])
        
    # Ακμές με filtration = max(HKS(u), HKS(v))  (sublevel set)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if dist_matrix[i, j] < threshold:
                st_hks.insert([i, j], filtration=max(hks[i], hks[j]))
                
    st_hks.expansion(2)
    st_hks.persistence()
    print("2/2: Υπολογίστηκε το HKS Node Filtration.")
    
    # Ανάκτηση Diagrams (H1)
    dgm_edge = clean_dgm(st_edge.persistence_intervals_in_dimension(1), threshold + 0.1)
    dgm_hks = clean_dgm(st_hks.persistence_intervals_in_dimension(1), 1.5)

    # --- Συγκριτικό Plot των Persistence Diagrams ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if len(dgm_edge) > 0:
        persim.plot_diagrams([dgm_edge], labels=['H1'], show=False)
    plt.title('Persistence Diagram (Edge Sublevel)')
    
    plt.subplot(1, 2, 2)
    if len(dgm_hks) > 0:
        persim.plot_diagrams([dgm_hks], labels=['H1'], show=False)
    plt.title('Persistence Diagram (HKS Filtration)')
    
    plt.tight_layout()
    plt.savefig("b3_persistence_diagrams_comparison.png")
    plt.close()
    print("Αποθηκεύτηκε: b3_persistence_diagrams_comparison.png")
    
    # --- Persistence Images ---
    from persim import PersistenceImager
    
    plt.figure(figsize=(14, 5))
    
    # Imager Edge
    if len(dgm_edge) > 0:
        pimager_edge = PersistenceImager(birth_range=(0.0, 0.6), pers_range=(0.0, 0.6), pixel_size=0.01)
        pimg_edge = pimager_edge.transform(dgm_edge)
        
        plt.subplot(1, 2, 1)
        plt.imshow(pimg_edge.T, origin='lower', cmap='plasma')
        plt.title('Persistence Image (Edge Sublevel H1)')
        plt.colorbar(label='Density')
    else:
        plt.subplot(1, 2, 1)
        plt.title('No H1 features (Edge Sublevel)')
        
    # Imager HKS
    if len(dgm_hks) > 0:
        pimager_hks = PersistenceImager(birth_range=(0.0, 1.5), pers_range=(0.0, 1.5), pixel_size=0.02)
        pimg_hks = pimager_hks.transform(dgm_hks)
        
        plt.subplot(1, 2, 2)
        plt.imshow(pimg_hks.T, origin='lower', cmap='plasma')
        plt.title('Persistence Image (HKS Filtration H1)')
        plt.colorbar(label='Density')
    else:
        plt.subplot(1, 2, 2)
        plt.title('No H1 features (HKS)')
        
    plt.tight_layout()
    plt.savefig("b3_persistence_images.png")
    plt.close()
    print("Αποθηκεύτηκε: b3_persistence_images.png")

if __name__ == "__main__":
    run_b1()
    run_b2()
    run_b3()
