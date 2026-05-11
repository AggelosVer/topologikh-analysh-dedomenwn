import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
import gudhi
import plotly.graph_objects as go

# =============================================================================
# Α.1—Νέφος Σημείων & Vietoris–Rips (from scratch)
# =============================================================================

def generate_data():
    np.random.seed(42)
    
    # 1. Circle (2D)
    n_circle = 30
    t = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t)]) + np.random.normal(0, 0.05, (n_circle, 2))
    
    # 2. Torus (3D)
    n_torus = 100
    theta = np.random.uniform(0, 2*np.pi, n_torus)
    phi = np.random.uniform(0, 2*np.pi, n_torus)
    R = 2
    r = 0.5
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    torus = np.column_stack([x, y, z])
    
    # 3. Sphere (3D)
    n_sphere = 100
    z_s = np.random.uniform(-1, 1, n_sphere)
    phi_s = np.random.uniform(0, 2*np.pi, n_sphere)
    x_s = np.sqrt(1 - z_s**2) * np.cos(phi_s)
    y_s = np.sqrt(1 - z_s**2) * np.sin(phi_s)
    sphere = np.column_stack([x_s, y_s, z_s])
    
    # 4. Gaussian Clusters (2D)
    clusters, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=42)
    
    return circle, torus, sphere, clusters

def construct_vr_scratch(points, epsilon):
    n = len(points)
    dist_matrix = cdist(points, points)
    
    vertices = list(range(n))
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i, j] <= epsilon:
                edges.append((i, j))
                
    triangles = []
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i, j] <= epsilon:
                for k in range(j+1, n):
                    if dist_matrix[i, k] <= epsilon and dist_matrix[j, k] <= epsilon:
                        triangles.append((i, j, k))
                        
    return vertices, edges, triangles

def rank_z2(M):
    M = M.astype(int) % 2
    R, C = M.shape
    r = 0
    for c in range(C):
        pivot = -1
        for i in range(r, R):
            if M[i, c] == 1:
                pivot = i
                break
        if pivot != -1:
            M[[r, pivot]] = M[[pivot, r]]
            for i in range(r + 1, R):
                if M[i, c] == 1:
                    M[i] = (M[i] + M[r]) % 2
            r += 1
    return r

def compute_betti_scratch(vertices, edges, triangles):
    V = len(vertices)
    E = len(edges)
    T = len(triangles)
    
    # Boundary 1
    d1 = np.zeros((V, E), dtype=int)
    for j, (u, v) in enumerate(edges):
        d1[u, j] = 1
        d1[v, j] = 1
        
    # Boundary 2
    d2 = np.zeros((E, T), dtype=int)
    edge_map = {e: i for i, e in enumerate(edges)}
    for k, (u, v, w) in enumerate(triangles):
        e1 = (min(u,v), max(u,v))
        e2 = (min(v,w), max(v,w))
        e3 = (min(u,w), max(u,w))
        
        d2[edge_map[e1], k] = 1
        d2[edge_map[e2], k] = 1
        d2[edge_map[e3], k] = 1
        
    rank_d1 = rank_z2(d1)
    rank_d2 = rank_z2(d2)
    
    beta0 = V - rank_d1
    beta1 = (E - rank_d1) - rank_d2
    beta2 = T - rank_d2
    
    return beta0, beta1, beta2

# =============================================================================
# Α.2—Σύγκριση VR/Čech/Alpha μέσω GUDHI
# =============================================================================

def compare_complexes(points, r):
    print(f"\n--- Σύγκριση Συμπλεγμάτων για r = {r} ---")
    
    # 1. Vietoris-Rips
    # GUDHI uses max_edge_length. To correspond to ball radius r, edge length is 2r.
    rips = gudhi.RipsComplex(points=points, max_edge_length=2*r)
    st_rips = rips.create_simplex_tree(max_dimension=2)
    
    # 2. Čech (using DelaunayCechComplex)
    cech = gudhi.DelaunayCechComplex(points=points)
    st_cech = cech.create_simplex_tree()
    
    # 3. Alpha
    alpha = gudhi.AlphaComplex(points=points)
    st_alpha = alpha.create_simplex_tree()
    
    def get_counts_and_betti(st, is_alpha=False, r_val=None):
        simplices = [0, 0, 0] # dim 0, 1, 2
        for s, f in st.get_filtration():
            if is_alpha:
                if f <= r_val**2:
                    dim = len(s) - 1
                    if dim < 3:
                        simplices[dim] += 1
            else:
                dim = len(s) - 1
                if dim < 3:
                    simplices[dim] += 1
                    
        # For Betti numbers, we need to compute persistence or use the tree
        # For Alpha, we need to prune or compute on the filtered complex
        if is_alpha:
            st.prune_above_filtration(r_val**2)
            st.persistence()
            betti = st.betti_numbers()
        else:
            st.persistence()
            betti = st.betti_numbers()
            
        # Pad betti with zeros up to beta2
        betti_padded = betti + [0] * (3 - len(betti))
        return simplices, betti_padded[:3]

    counts_rips, betti_rips = get_counts_and_betti(st_rips)
    counts_cech, betti_cech = get_counts_and_betti(st_cech, is_alpha=True, r_val=r)
    counts_alpha, betti_alpha = get_counts_and_betti(st_alpha, is_alpha=True, r_val=r)
    
    print(f"{'Complex':<10} | {'0-simplices':<12} | {'1-simplices':<12} | {'2-simplices':<12} | {'b0':<4} | {'b1':<4} | {'b2':<4}")
    print("-" * 75)
    print(f"{'VR':<10} | {counts_rips[0]:<12} | {counts_rips[1]:<12} | {counts_rips[2]:<12} | {betti_rips[0]:<4} | {betti_rips[1]:<4} | {betti_rips[2]:<4}")
    print(f"{'Cech':<10} | {counts_cech[0]:<12} | {counts_cech[1]:<12} | {counts_cech[2]:<12} | {betti_cech[0]:<4} | {betti_cech[1]:<4} | {betti_cech[2]:<4}")
    print(f"{'Alpha':<10} | {counts_alpha[0]:<12} | {counts_alpha[1]:<12} | {counts_alpha[2]:<12} | {betti_alpha[0]:<4} | {betti_alpha[1]:<4} | {betti_alpha[2]:<4}")

def visualize_point_cloud(points, title):
    dim = points.shape[1]
    if dim == 2:
        plt.scatter(points[:, 0], points[:, 1], c='blue', s=20)
        plt.title(title)
        plt.axis('equal')
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=20)
        ax.set_title(title)
        
# =============================================================================
# Α.3—Filtration & Ανάλυση Κλίμακας
# =============================================================================

def filtration_analysis(points):
    r_vals = np.arange(0, 2.1, 0.1)
    b0_vals = []
    b1_vals = []
    
    print("\n--- Filtration Analysis ---")
    print(f"{'r':<4} | {'b0':<4} | {'b1':<4}")
    print("-" * 20)
    
    for r in r_vals:
        rips = gudhi.RipsComplex(points=points, max_edge_length=2*r)
        st = rips.create_simplex_tree(max_dimension=2)
        st.persistence()
        betti = st.betti_numbers()
        
        b0 = betti[0] if len(betti) > 0 else 0
        b1 = betti[1] if len(betti) > 1 else 0
        
        b0_vals.append(b0)
        b1_vals.append(b1)
        
        print(f"{r:.1f} | {b0:<4} | {b1:<4}")
        
    plt.figure(figsize=(10, 5))
    plt.plot(r_vals, b0_vals, label='$\\beta_0$ (Components)', marker='o')
    plt.plot(r_vals, b1_vals, label='$\\beta_1$ (Holes)', marker='s')
    plt.xlabel('Radius r')
    plt.ylabel('Betti Number')
    plt.title('Betti Numbers vs Filtration Radius')
    plt.legend()
    plt.grid(True)
    plt.savefig('filtration_betti.png')
    print("\nΤο γράφημα Betti numbers vs r αποθηκεύτηκε ως 'filtration_betti.png'")

def plot_complex_steps(points, r_vals):
    plt.figure(figsize=(15, 5))
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points, points)
    
    for idx, r in enumerate(r_vals):
        plt.subplot(1, len(r_vals), idx + 1)
        plt.scatter(points[:, 0], points[:, 1], c='blue', s=20)
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if dist_matrix[i, j] <= 2*r:
                    plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'r-', alpha=0.3)
        plt.title(f'r = {r}')
        plt.axis('equal')
    plt.savefig('complex_steps.png')
    print("Τα στιγμιότυπα του συμπλέγματος αποθηκεύτηκαν ως 'complex_steps.png'")

def visualize_complex_2d(points, edges, triangles, title):
    plt.figure(figsize=(8, 8))
    # Plot triangles
    for t in triangles:
        poly = plt.Polygon(points[list(t)], facecolor='green', alpha=0.3, edgecolor='none')
        plt.gca().add_patch(poly)
    # Plot edges
    for e in edges:
        plt.plot(points[list(e), 0], points[list(e), 1], 'r-', alpha=0.5)
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=30, zorder=3)
    plt.title(title)
    plt.axis('equal')
    plt.savefig('complex_2d_colored.png')
    print("Η 2D οπτικοποίηση αποθηκεύτηκε ως 'complex_2d_colored.png'")

def visualize_complex_3d_plotly(points, edges, triangles, title):
    import plotly.graph_objects as go
    
    # Points
    trace_points = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Vertices (Dim 0)'
    )
    
    # Edges
    edge_x = []
    edge_y = []
    edge_z = []
    for e in edges:
        edge_x.extend([points[e[0], 0], points[e[1], 0], None])
        edge_y.extend([points[e[0], 1], points[e[1], 1], None])
        edge_z.extend([points[e[0], 2], points[e[1], 2], None])
        
    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='red', width=2),
        name='Edges (Dim 1)'
    )
    
    # Triangles
    i = [t[0] for t in triangles]
    j = [t[1] for t in triangles]
    k = [t[2] for t in triangles]
    
    trace_triangles = go.Mesh3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        i=i, j=j, k=k,
        color='green',
        opacity=0.3,
        name='Triangles (Dim 2)'
    )
    
    fig = go.Figure(data=[trace_points, trace_edges, trace_triangles])
    fig.update_layout(title=title)
    fig.write_html('complex_3d.html')
    print("Η 3D οπτικοποίηση αποθηκεύτηκε ως 'complex_3d.html'")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Παραγωγή δεδομένων...")
    circle, torus, sphere, clusters = generate_data()
    
    # Οπτικοποίηση
    fig = plt.figure(figsize=(12, 10))
    
    # 1. Circle (2D)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(circle[:, 0], circle[:, 1], c='blue', s=20)
    ax1.set_title('Circle')
    ax1.axis('equal')
    
    # 2. Torus (3D)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.scatter(torus[:, 0], torus[:, 1], torus[:, 2], c='red', s=10)
    ax2.set_title('Torus')
    
    # 3. Sphere (3D)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(sphere[:, 0], sphere[:, 1], sphere[:, 2], c='purple', s=10)
    ax3.set_title('Sphere')
    
    # 4. Clusters (2D)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(clusters[:, 0], clusters[:, 1], c='green', s=20)
    ax4.set_title('Gaussian Clusters')
    ax4.axis('equal')
    
    plt.savefig('point_clouds.png')
    print("Τα νέφη σημείων αποθηκεύτηκαν ως 'point_clouds.png'")
    
    # Α.1 Έλεγχος From Scratch
    print("\n--- Α.1 From Scratch Έλεγχος (Κύκλος) ---")
    epsilon = 0.5 # Αντιστοιχεί σε r = 0.25
    v, e, t = construct_vr_scratch(circle, epsilon)
    print(f"Στοιχεία: {len(v)} κορυφές, {len(e)} ακμές, {len(t)} τρίγωνα")
    b0, b1, b2 = compute_betti_scratch(v, e, t)
    print(f"Betti numbers (Scratch): b0={b0}, b1={b1}, b2={b2}")
    
    # Σύγκριση με GUDHI για το ίδιο epsilon
    rips = gudhi.RipsComplex(points=circle, max_edge_length=epsilon)
    st = rips.create_simplex_tree(max_dimension=2)
    st.persistence()
    betti_gudhi = st.betti_numbers()
    print(f"Betti numbers (GUDHI): {betti_gudhi}")
    
    # Α.2 Σύγκριση
    # Χρησιμοποιούμε τον κύκλο για ευκολία
    compare_complexes(circle, r=0.3)
    
    # Α.3 Filtration
    filtration_analysis(circle)
    
    # Στιγμιότυπα συμπλέγματος
    plot_complex_steps(circle, [0.1, 0.2, 0.5, 1.0])
    
    # Έξτρα Οπτικοποίηση (Διαφορετικό χρώμα ανά διάσταση)
    # 2D (Κύκλος)
    print("\nΠαραγωγή 2D οπτικοποίησης με χρώματα ανά διάσταση...")
    v, e, t = construct_vr_scratch(circle, epsilon=0.5)
    visualize_complex_2d(circle, e, t, 'VR Complex (Circle) - Colors by Dimension')
    
    # 3D (Σφαίρα)
    print("Παραγωγή 3D οπτικοποίησης με Plotly...")
    # Χρησιμοποιούμε λιγότερα σημεία για τη σφαίρα για να φαίνεται καθαρά
    sphere_small = sphere[:30]
    v_s, e_s, t_s = construct_vr_scratch(sphere_small, epsilon=0.8)
    visualize_complex_3d_plotly(sphere_small, e_s, t_s, 'VR Complex (Sphere) - 3D Plotly')
    
    # plt.show()
