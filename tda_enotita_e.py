import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import warnings
import sys

# Υποστήριξη ελληνικών στο terminal των Windows
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# =============================================================================
# Ε.1 — Zigzag Persistent Homology
# =============================================================================
# Zigzag filtration από χρονοσειρά (sliding window) εφαρμοσμένο στα δεδομένα
# Ενότητας Δ (S&P 500). Υπολογισμός zigzag persistent homology.
# Σύγκριση & σχολιασμός διαφορών από κλασική επίμονη ομολογία.
# =============================================================================


def takens_embedding(x, delay=1, dimension=3):
    """Μετατρέπει μια 1D χρονοσειρά σε point cloud μέσω Takens Embedding."""
    N = len(x)
    embedded = []
    for i in range(N - (dimension - 1) * delay):
        pt = [x[i + j * delay] for j in range(dimension)]
        embedded.append(pt)
    return np.array(embedded)


def sliding_window_point_clouds(time_series, window_size=50, step=25, emb_dim=3, emb_delay=1):
    """
    Δημιουργεί μια ακολουθία point clouds από χρονοσειρά μέσω sliding window
    και Takens embedding.

    Parameters
    ----------
    time_series : array-like
        Η 1D χρονοσειρά.
    window_size : int
        Μέγεθος κάθε παραθύρου.
    step : int
        Βήμα ολίσθησης μεταξύ διαδοχικών παραθύρων.
    emb_dim : int
        Διάσταση Takens embedding.
    emb_delay : int
        Καθυστέρηση (delay) του Takens embedding.

    Returns
    -------
    windows : list of np.array
        Λίστα point clouds, κάθε ένα σχήμα (n_points, emb_dim).
    window_centers : list of int
        Κέντρα (μέσα indices) κάθε παραθύρου.
    """
    windows = []
    window_centers = []
    N = len(time_series)

    for start in range(0, N - window_size + 1, step):
        end = start + window_size
        segment = time_series[start:end]
        pc = takens_embedding(segment, delay=emb_delay, dimension=emb_dim)
        windows.append(pc)
        window_centers.append((start + end) // 2)

    return windows, window_centers


# =============================================================================
# Zigzag Persistent Homology — Υλοποίηση από το Μηδέν
# =============================================================================
# Η zigzag filtration κατασκευάζεται ως:
#   K_0 ↪ K_0∪K_1 ↩ K_1 ↪ K_1∪K_2 ↩ K_2 ↪ ...
# όπου τα βέλη αντιπροσωπεύουν inclusions.
#
# Αυτή η υλοποίηση χρησιμοποιεί τη μέθοδο "στιγμιαίων Betti αριθμών"
# (instantaneous Betti numbers) μέσω boundary matrix στα αντίστοιχα
# Rips complexes ανά χρονικό βήμα, και στη συνέχεια εκτιμά τα zigzag
# persistence intervals παρακολουθώντας τα births/deaths χαρακτηριστικών
# μεταξύ διαδοχικών βημάτων.
# =============================================================================


def build_rips_simplices(points, max_radius):
    """
    Κατασκευάζει Rips complex (0, 1, 2-simplices) για δοθέν point cloud.

    Returns
    -------
    vertices : set of ints
    edges : set of frozensets
    triangles : set of frozensets
    """
    n = len(points)
    dist_matrix = squareform(pdist(points))

    vertices = set(range(n))
    edges = set()
    triangles = set()

    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] <= 2 * max_radius:
                edges.add(frozenset([i, j]))

    for i in range(n):
        for j in range(i + 1, n):
            if frozenset([i, j]) not in edges:
                continue
            for k in range(j + 1, n):
                if (frozenset([i, k]) in edges and
                        frozenset([j, k]) in edges):
                    triangles.add(frozenset([i, j, k]))

    return vertices, edges, triangles


def compute_betti_numbers(n_vertices, edges, triangles):
    """
    Υπολογίζει τους αριθμούς Betti β₀ και β₁ μέσω πίνακα συνόρων.

    β₀ = dim(ker ∂₁) = n_vertices - rank(∂₁)
    β₁ = dim(ker ∂₂) - dim(im ∂₁) = (n_edges - rank(∂₂)) - rank(∂₁)
         Ή ισοδύναμα: β₁ = n_edges - rank(∂₁) - rank(∂₂)
         (Αυτό ισχύει αν: β₁ = nullity(∂₁) - rank(∂₂) = (n_edges - rank(∂₁)) - rank(∂₂))

    Χρησιμοποιούμε GF(2) (mod 2) αριθμητική.
    """
    edges_list = list(edges)
    triangles_list = list(triangles)
    n_edges = len(edges_list)
    n_triangles = len(triangles_list)

    # ∂₁: edges → vertices
    if n_edges == 0:
        rank_d1 = 0
    else:
        # Boundary matrix ∂₁ (n_vertices x n_edges) over GF(2)
        d1 = np.zeros((n_vertices, n_edges), dtype=int)
        for j, edge in enumerate(edges_list):
            verts = sorted(edge)
            d1[verts[0], j] = 1
            d1[verts[1], j] = 1
        rank_d1 = _gf2_rank(d1)

    # ∂₂: triangles → edges
    if n_triangles == 0:
        rank_d2 = 0
    else:
        edge_to_idx = {e: i for i, e in enumerate(edges_list)}
        d2 = np.zeros((n_edges, n_triangles), dtype=int)
        for j, tri in enumerate(triangles_list):
            verts = sorted(tri)
            # Τα 3 faces (ακμές) του τριγώνου
            faces = [
                frozenset([verts[0], verts[1]]),
                frozenset([verts[0], verts[2]]),
                frozenset([verts[1], verts[2]])
            ]
            for face in faces:
                if face in edge_to_idx:
                    d2[edge_to_idx[face], j] = 1
        rank_d2 = _gf2_rank(d2)

    beta_0 = n_vertices - rank_d1
    beta_1 = n_edges - rank_d1 - rank_d2

    return beta_0, max(0, beta_1)


def _gf2_rank(matrix):
    """Υπολογίζει τον βαθμό (rank) πίνακα πάνω από GF(2) μέσω Gaussian elimination."""
    M = matrix.copy() % 2
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        # Βρίσκουμε pivot row
        pivot = None
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap
        M[[rank, pivot]] = M[[pivot, rank]]
        # Eliminate
        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank


def zigzag_persistence_from_betti(betti_sequence):
    """
    Εξάγει zigzag persistence intervals παρακολουθώντας πώς αλλάζουν
    οι αριθμοί Betti μεταξύ διαδοχικών βημάτων της zigzag filtration.

    Όταν ο β_k αυξάνεται → birth ενός νέου k-χαρακτηριστικού
    Όταν ο β_k μειώνεται → death ενός k-χαρακτηριστικού

    Parameters
    ----------
    betti_sequence : list of (β₀, β₁)
        Ακολουθία Betti αριθμών ανά χρονικό βήμα.

    Returns
    -------
    intervals : list of (birth, death, dim)
        Zigzag persistence intervals.
    """
    intervals = []

    for dim in range(2):  # H0, H1
        active_features = []  # Stack: χρόνοι birth ενεργών features
        prev_betti = 0

        for t, betti_pair in enumerate(betti_sequence):
            curr_betti = betti_pair[dim]

            if curr_betti > prev_betti:
                # Γεννήθηκαν νέα features
                for _ in range(curr_betti - prev_betti):
                    active_features.append(t)
            elif curr_betti < prev_betti:
                # Πέθαναν features (LIFO: κλείνουμε τα πιο πρόσφατα)
                for _ in range(prev_betti - curr_betti):
                    if active_features:
                        birth = active_features.pop()
                        intervals.append((birth, t, dim))

            prev_betti = curr_betti

        # Features που επιμένουν μέχρι το τέλος
        for birth in active_features:
            intervals.append((birth, len(betti_sequence), dim))

    return intervals


def run_zigzag_filtration(point_clouds, max_radius):
    """
    Εκτελεί zigzag filtration:
      K_0 ↪ K_0∪K_1 ↩ K_1 ↪ K_1∪K_2 ↩ K_2 ↪ ...

    Υπολογίζει τους Betti αριθμούς σε κάθε βήμα και εξάγει
    zigzag persistence intervals.

    Parameters
    ----------
    point_clouds : list of np.ndarray
        Ακολουθία point clouds.
    max_radius : float
        Μέγιστη ακτίνα Rips.

    Returns
    -------
    intervals : list of (birth, death, dim)
    betti_sequence : list of (β₀, β₁)
    n_times : int
    """
    n_clouds = len(point_clouds)
    n_times = 2 * n_clouds - 1
    betti_sequence = []

    print(f"   Χρονικά βήματα zigzag filtration: {n_times}")

    for t in range(n_times):
        if t % 2 == 0:
            # t = 2i → K_i
            i = t // 2
            pc = point_clouds[i]
            verts, edges, tris = build_rips_simplices(pc, max_radius)
            b0, b1 = compute_betti_numbers(len(pc), edges, tris)
            betti_sequence.append((b0, b1))
            if t % 4 == 0:  # Εκτύπωση κάθε 2 windows
                print(f"     t={t} (K_{i}): β₀={b0}, β₁={b1}, "
                      f"|V|={len(verts)}, |E|={len(edges)}, |Δ|={len(tris)}")
        else:
            # t = 2i+1 → K_i ∪ K_{i+1}
            i = t // 2
            pc_union = np.vstack([point_clouds[i], point_clouds[i + 1]])
            verts, edges, tris = build_rips_simplices(pc_union, max_radius)
            b0, b1 = compute_betti_numbers(len(pc_union), edges, tris)
            betti_sequence.append((b0, b1))

    intervals = zigzag_persistence_from_betti(betti_sequence)

    return intervals, betti_sequence, n_times


def classical_ph_per_window(point_clouds, max_radius):
    """
    Υπολογίζει κλασική persistent homology ανά παράθυρο.
    Χρησιμοποιεί ripser αν είναι διαθέσιμο, αλλιώς gudhi, αλλιώς from scratch.

    Returns
    -------
    betti_per_window : list of (β₀, β₁)
    """
    betti_per_window = []

    # Δοκιμάζουμε ripser πρώτα
    use_ripser = False
    use_gudhi = False
    try:
        from ripser import ripser
        use_ripser = True
        print("   (Χρήση ripser)")
    except ImportError:
        try:
            import gudhi
            use_gudhi = True
            print("   (Χρήση GUDHI)")
        except ImportError:
            print("   (Χρήση from-scratch Betti computation)")

    for i, pc in enumerate(point_clouds):
        if use_ripser:
            result = ripser(pc, maxdim=1, thresh=2 * max_radius)
            dgms = result['dgms']
            # β₀: connected components that live beyond 2 * max_radius
            b0 = max(1, np.sum(np.isinf(dgms[0][:, 1]))) if len(dgms[0]) > 0 else 1
            b1 = np.sum(np.isinf(dgms[1][:, 1])) if len(dgms) > 1 and len(dgms[1]) > 0 else 0
        elif use_gudhi:
            import gudhi
            rips_complex = gudhi.RipsComplex(points=pc, max_edge_length=2 * max_radius)
            st = rips_complex.create_simplex_tree(max_dimension=2)
            st.compute_persistence()
            b0 = len([val for val in st.persistence_intervals_in_dimension(0) if np.isinf(val[1])])
            b1 = len([val for val in st.persistence_intervals_in_dimension(1) if np.isinf(val[1])])
        else:
            verts, edges, tris = build_rips_simplices(pc, max_radius)
            b0, b1 = compute_betti_numbers(len(pc), edges, tris)

        betti_per_window.append((b0, b1))

    return betti_per_window


def classical_ph_on_union(point_clouds, max_radius):
    """
    Κλασική PH στην ένωση όλων των point clouds.

    Returns
    -------
    dgms : list of np.arrays ή None
    betti : (β₀, β₁)
    """
    all_points = np.vstack(point_clouds)

    # Υποδειγματοληψία αν πολύ μεγάλο
    if len(all_points) > 400:
        idx = np.random.choice(len(all_points), 400, replace=False)
        all_points = all_points[idx]

    print(f"   Κλασική PH στην ένωση ({len(all_points)} σημεία)...")

    try:
        from ripser import ripser
        result = ripser(all_points, maxdim=1, thresh=2 * max_radius)
        dgms = result['dgms']
        return dgms
    except ImportError:
        pass

    try:
        import gudhi
        rips_complex = gudhi.RipsComplex(points=all_points, max_edge_length=2 * max_radius)
        st = rips_complex.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        h0 = np.array(st.persistence_intervals_in_dimension(0))
        h1 = np.array(st.persistence_intervals_in_dimension(1))
        return [h0, h1]
    except ImportError:
        pass

    # From scratch
    verts, edges, tris = build_rips_simplices(all_points, max_radius)
    b0, b1 = compute_betti_numbers(len(all_points), edges, tris)
    return [np.array([[0, np.inf]] * b0), np.array([[0, max_radius]] * b1)]


# =============================================================================
# Οπτικοποιήσεις
# =============================================================================


def plot_zigzag_diagram(intervals, n_times, filename):
    """
    Οπτικοποιεί το zigzag persistence diagram (scatter + barcode).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {0: '#2980b9', 1: '#e67e22', 2: '#27ae60'}
    dim_labels = {0: '$H_0$', 1: '$H_1$'}

    # 1. Persistence Diagram
    ax = axes[0]
    for (b, d, dim) in intervals:
        if dim > 1:
            continue
        color = colors.get(dim, 'gray')
        ax.scatter(b, d, c=color, s=40, alpha=0.7, edgecolors='k', linewidths=0.5, zorder=3)

    max_val = n_times
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Birth (zigzag time step)', fontsize=12)
    ax.set_ylabel('Death (zigzag time step)', fontsize=12)
    ax.set_title('Zigzag Persistence Diagram', fontsize=14, fontweight='bold')

    dims_present = sorted(set(entry[2] for entry in intervals if entry[2] <= 1))
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[d], markersize=8, label=dim_labels[d])
                       for d in dims_present]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Barcode
    ax = axes[1]
    y_pos = 0
    for (b, d, dim) in sorted(intervals, key=lambda x: (x[2], x[0])):
        if dim > 1:
            continue
        color = colors.get(dim, 'gray')
        ax.plot([b, d], [y_pos, y_pos], color=color, lw=1.8, alpha=0.8)
        y_pos += 1

    ax.set_xlabel('Zigzag Time Step', fontsize=12)
    ax.set_yticks([])
    ax.set_title('Zigzag Barcode', fontsize=14, fontweight='bold')
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   -> Αποθηκεύτηκε: {filename}")


def plot_classical_diagram(dgms, filename, title_suffix=""):
    """
    Οπτικοποιεί κλασικό persistence diagram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#2980b9', '#e67e22', '#27ae60']

    # 1. Persistence Diagram
    ax = axes[0]
    max_d = 0
    for dim, dgm in enumerate(dgms):
        if dim > 1:
            continue
        dgm = np.array(dgm) if not isinstance(dgm, np.ndarray) else dgm
        if len(dgm) == 0:
            continue
        for b, d in dgm:
            if not np.isinf(d):
                max_d = max(max_d, d)
                ax.scatter(b, d, c=colors[dim], s=40, alpha=0.7,
                           edgecolors='k', linewidths=0.5, zorder=3)

    vis_max = max_d * 1.2 if max_d > 0 else 2.0
    # Infinite deaths
    for dim, dgm in enumerate(dgms):
        if dim > 1:
            continue
        dgm = np.array(dgm) if not isinstance(dgm, np.ndarray) else dgm
        if len(dgm) == 0:
            continue
        for b, d in dgm:
            if np.isinf(d):
                ax.scatter(b, vis_max, c=colors[dim], s=40, alpha=0.7,
                           marker='^', edgecolors='k', linewidths=0.5, zorder=3)

    ax.plot([0, vis_max], [0, vis_max], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title(f'Classical Persistence Diagram {title_suffix}', fontsize=14, fontweight='bold')

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[d], markersize=8, label=f'$H_{d}$')
                       for d in range(min(2, len(dgms)))]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Barcode
    ax = axes[1]
    y_pos = 0
    for dim, dgm in enumerate(dgms):
        if dim > 1:
            continue
        dgm = np.array(dgm) if not isinstance(dgm, np.ndarray) else dgm
        if len(dgm) == 0:
            continue
        for b, d in dgm:
            death = d if not np.isinf(d) else vis_max
            ax.plot([b, death], [y_pos, y_pos], color=colors[dim], lw=1.8, alpha=0.8)
            y_pos += 1

    ax.set_xlabel('Filtration Radius', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'Classical Barcode {title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   -> Αποθηκεύτηκε: {filename}")


def plot_betti_comparison(betti_classical, betti_zigzag_seq, window_centers, n_times, filename):
    """
    Συγκρίνει αριθμούς Betti μεταξύ κλασικής PH (ανά window) και zigzag.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # H0
    ax = axes[0]
    classical_b0 = [b[0] for b in betti_classical]
    ax.plot(window_centers, classical_b0, 'b-o', markersize=5,
            label='Κλασική PH (ανά παράθυρο)', alpha=0.8, linewidth=1.5)

    zigzag_b0 = [b[0] for b in betti_zigzag_seq]
    zz_x = np.linspace(window_centers[0], window_centers[-1], len(zigzag_b0))
    ax.plot(zz_x, zigzag_b0, 'r-s', markersize=4,
            label='Zigzag PH (ανά zigzag step)', alpha=0.7, linewidth=1.5)

    ax.set_ylabel('$\\beta_0$ (Συνεκτικές Συνιστώσες)', fontsize=12)
    ax.set_title('Σύγκριση $H_0$: Κλασική vs Zigzag Persistent Homology', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # H1
    ax = axes[1]
    classical_b1 = [b[1] for b in betti_classical]
    ax.plot(window_centers, classical_b1, 'b-o', markersize=5,
            label='Κλασική PH (ανά παράθυρο)', alpha=0.8, linewidth=1.5)

    zigzag_b1 = [b[1] for b in betti_zigzag_seq]
    ax.plot(zz_x, zigzag_b1, 'r-s', markersize=4,
            label='Zigzag PH (ανά zigzag step)', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Χρόνος (index χρονοσειράς)', fontsize=12)
    ax.set_ylabel('$\\beta_1$ (1-κύκλοι / τρύπες)', fontsize=12)
    ax.set_title('Σύγκριση $H_1$: Κλασική vs Zigzag Persistent Homology', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   -> Αποθηκεύτηκε: {filename}")


def plot_zigzag_schematic(n_windows, filename):
    """
    Σχεδιάζει σχηματικό διάγραμμα zigzag filtration.
    """
    fig, ax = plt.subplots(figsize=(min(16, 2 + 2.5 * min(n_windows, 5)), 3.5))

    n_show = min(n_windows, 5)
    x_positions = []
    labels = []

    pos = 0
    for i in range(n_show):
        x_positions.append(pos)
        labels.append(f'$K_{{{i}}}$')
        pos += 2.0

        if i < n_show - 1:
            x_positions.append(pos)
            labels.append(f'$K_{{{i}{i + 1}}}$')
            pos += 2.0

    if n_windows > n_show:
        x_positions.append(pos)
        labels.append('$\\cdots$')

    y = 0.5

    for idx, (x, label) in enumerate(zip(x_positions, labels)):
        if label == '$\\cdots$':
            ax.text(x, y, label, ha='center', va='center', fontsize=18)
        else:
            is_single = ',' not in label.replace('cdots', '')
            bbox_color = '#d5f4e6' if is_single else '#fef3cd'
            ax.text(x, y, label, ha='center', va='center', fontsize=15,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=bbox_color,
                              edgecolor='#2c3e50', alpha=0.9, linewidth=1.5))

    # Βέλη
    for idx in range(len(x_positions) - 1):
        if labels[idx + 1] == '$\\cdots$':
            break
        x_start = x_positions[idx] + 0.6
        x_end = x_positions[idx + 1] - 0.6

        if idx % 2 == 0:
            # Forward: K_i ↪ K_{i,i+1}
            ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5,
                                        connectionstyle='arc3,rad=0.0'))
        else:
            # Backward: K_{i,i+1} ↩ K_{i+1}
            ax.annotate('', xy=(x_start, y), xytext=(x_end, y),
                        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2.5,
                                        connectionstyle='arc3,rad=0.0'))

    ax.set_xlim(-1, x_positions[-1] + 1)
    ax.set_ylim(-0.8, 1.8)
    ax.axis('off')
    ax.set_title('Zigzag Filtration: $K_0 \\hookrightarrow K_{01} \\hookleftarrow K_1 '
                 '\\hookrightarrow K_{12} \\hookleftarrow K_2 \\hookrightarrow \\cdots$',
                 fontsize=13, fontweight='bold', pad=15)

    legend_elements = [
        Line2D([0], [0], color='#27ae60', lw=2.5, label='Forward inclusion ($\\hookrightarrow$)'),
        Line2D([0], [0], color='#c0392b', lw=2.5, label='Backward inclusion ($\\hookleftarrow$)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=10, ncol=2,
              bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   -> Αποθηκεύτηκε: {filename}")


def plot_sliding_windows(time_series, window_centers, window_size, filename, max_show=6):
    """
    Οπτικοποιεί τα sliding windows πάνω στη χρονοσειρά.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(time_series, 'k-', alpha=0.6, linewidth=0.8, label='Χρονοσειρά S&P 500')

    colors = plt.cm.Set2(np.linspace(0, 1, max_show))
    n_show = min(len(window_centers), max_show)
    step = max(1, len(window_centers) // n_show)

    for idx, i in enumerate(range(0, len(window_centers), step)):
        if idx >= max_show:
            break
        center = window_centers[i]
        start = max(0, center - window_size // 2)
        end = min(len(time_series), center + window_size // 2)
        ax.axvspan(start, end, alpha=0.15, color=colors[idx])
        ax.axvline(center, alpha=0.4, color=colors[idx], linestyle='--', linewidth=0.8)

    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Τυποποιημένη Τιμή', fontsize=12)
    ax.set_title('Sliding Windows στη Χρονοσειρά S&P 500', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   -> Αποθηκεύτηκε: {filename}")


def plot_persistence_summary(intervals_zz, betti_classical, betti_zigzag_seq, n_times, filename):
    """
    Δημιουργεί ένα συγκεντρωτικό γράφημα σύγκρισης.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    colors_dim = {0: '#2980b9', 1: '#e67e22'}

    # (0,0) Zigzag PD
    ax = axes[0, 0]
    for (b, d, dim) in intervals_zz:
        if dim > 1:
            continue
        ax.scatter(b, d, c=colors_dim.get(dim, 'gray'), s=35, alpha=0.7,
                   edgecolors='k', linewidths=0.3)
    ax.plot([0, n_times], [0, n_times], 'k--', alpha=0.3)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title('Zigzag Persistence Diagram', fontweight='bold')
    ax.grid(True, alpha=0.2)

    # (0,1) Zigzag Barcode
    ax = axes[0, 1]
    y_pos = 0
    for (b, d, dim) in sorted(intervals_zz, key=lambda x: (x[2], x[0])):
        if dim > 1:
            continue
        ax.plot([b, d], [y_pos, y_pos], color=colors_dim.get(dim, 'gray'), lw=1.5, alpha=0.8)
        y_pos += 1
    ax.set_xlabel('Zigzag Time')
    ax.set_yticks([])
    ax.set_title('Zigzag Barcode', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

    # (1,0) β₀ over time
    ax = axes[1, 0]
    zigzag_b0 = [b[0] for b in betti_zigzag_seq]
    classical_b0 = [b[0] for b in betti_classical]
    ax.plot(zigzag_b0, 'r-', linewidth=1.5, alpha=0.8, label='Zigzag β₀')
    # Τα classical αντιστοιχούν στα even steps
    even_steps = list(range(0, len(betti_zigzag_seq), 2))
    ax.plot(even_steps, classical_b0, 'b--o', markersize=4, linewidth=1, alpha=0.7, label='Classical β₀')
    ax.set_xlabel('Zigzag Time Step')
    ax.set_ylabel('$\\beta_0$')
    ax.set_title('$\\beta_0$ Evolution: Zigzag vs Classical', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # (1,1) β₁ over time
    ax = axes[1, 1]
    zigzag_b1 = [b[1] for b in betti_zigzag_seq]
    classical_b1 = [b[1] for b in betti_classical]
    ax.plot(zigzag_b1, 'r-', linewidth=1.5, alpha=0.8, label='Zigzag β₁')
    ax.plot(even_steps, classical_b1, 'b--o', markersize=4, linewidth=1, alpha=0.7, label='Classical β₁')
    ax.set_xlabel('Zigzag Time Step')
    ax.set_ylabel('$\\beta_1$')
    ax.set_title('$\\beta_1$ Evolution: Zigzag vs Classical', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9', markersize=8, label='$H_0$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22', markersize=8, label='$H_1$'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    plt.suptitle('Ε.1 — Σύγκριση Zigzag vs Κλασικής Persistent Homology (S&P 500)',
                 fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   -> Αποθηκεύτηκε: {filename}")


def run_e1():
    """
    Ε.1 — Zigzag Persistent Homology

    Εφαρμόζει zigzag filtration μέσω sliding window στα δεδομένα S&P 500
    (Ενότητα Δ.2), υπολογίζει zigzag persistent homology,
    και συγκρίνει με κλασική persistent homology.
    """
    print("=" * 70)
    print("Ε.1 — Zigzag Persistent Homology")
    print("=" * 70)

    # -------------------------------------------------------------------
    # 1. Φόρτωση δεδομένων S&P 500 (ίδια πηγή με Ενότητα Δ.2)
    # -------------------------------------------------------------------
    print("\n1. Φόρτωση δεδομένων S&P 500...")
    try:
        import yfinance as yf
        sp500 = yf.download('^GSPC', period='2y', interval='1d', progress=False)
        if sp500.empty:
            raise ValueError("Κενά δεδομένα")
        close_prices = sp500['Close'].values.flatten()
        print(f"   Φορτώθηκαν {len(close_prices)} ημερήσιες τιμές κλεισίματος.")
    except Exception as e:
        print(f"   Αποτυχία φόρτωσης S&P 500 ({e}).")
        print("   Χρήση συνθετικής χρονοσειράς (ημίτονο + τάση + θόρυβος)...")
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 500)
        close_prices = np.sin(t) + 0.3 * np.sin(3 * t) + 0.1 * t + np.random.normal(0, 0.1, len(t))

    scaler = StandardScaler()
    close_scaled = scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

    # -------------------------------------------------------------------
    # 2. Δημιουργία Sliding Window Point Clouds
    # -------------------------------------------------------------------
    print("\n2. Δημιουργία sliding window point clouds...")
    window_size = 50
    step = 25
    emb_dim = 3
    emb_delay = 1

    point_clouds, window_centers = sliding_window_point_clouds(
        close_scaled, window_size=window_size, step=step,
        emb_dim=emb_dim, emb_delay=emb_delay
    )
    print(f"   Δημιουργήθηκαν {len(point_clouds)} point clouds")
    print(f"   Μέγεθος κάθε cloud: {point_clouds[0].shape}")
    print(f"   Window size = {window_size}, step = {step}, emb_dim = {emb_dim}")

    # Οπτικοποίηση sliding windows
    plot_sliding_windows(close_scaled, window_centers, window_size, "e1_sliding_windows.png")

    # Σχηματικό διάγραμμα zigzag filtration
    plot_zigzag_schematic(len(point_clouds), "e1_zigzag_schematic.png")

    # -------------------------------------------------------------------
    # 3. Υποδειγματοληψία & Προετοιμασία
    # -------------------------------------------------------------------
    print("\n3. Προετοιμασία point clouds για zigzag filtration...")

    # Κρατάμε αντιπροσωπευτικό αριθμό windows
    max_clouds = 10
    if len(point_clouds) > max_clouds:
        step_sample = len(point_clouds) // max_clouds
        sampled_indices = list(range(0, len(point_clouds), step_sample))[:max_clouds]
        sampled_clouds = [point_clouds[i] for i in sampled_indices]
        sampled_centers = [window_centers[i] for i in sampled_indices]
        print(f"   Δειγματοληψία: {len(sampled_clouds)} από {len(point_clouds)} windows")
    else:
        sampled_clouds = point_clouds
        sampled_centers = window_centers

    # Υποδειγματοληψία σημείων ανά cloud
    max_pts = 25
    reduced_clouds = []
    for pc in sampled_clouds:
        if len(pc) > max_pts:
            idx = np.random.choice(len(pc), max_pts, replace=False)
            reduced_clouds.append(pc[idx])
        else:
            reduced_clouds.append(pc)

    print(f"   Σημεία ανά cloud (μετά υποδειγματοληψίας): {reduced_clouds[0].shape[0]}")

    # Εκτίμηση κατάλληλης ακτίνας
    all_dists = []
    for pc in reduced_clouds:
        dists = pdist(pc)
        all_dists.extend(dists)
    all_dists = np.array(all_dists)
    max_radius = np.percentile(all_dists, 15)
    print(f"   Ακτίνα Rips (15th percentile αποστάσεων): {max_radius:.4f}")

    # -------------------------------------------------------------------
    # 4. Zigzag Persistent Homology
    # -------------------------------------------------------------------
    print("\n4. Υπολογισμός Zigzag Persistent Homology...")
    intervals_zz, betti_zigzag_seq, n_times = run_zigzag_filtration(reduced_clouds, max_radius)

    # Στατιστικά zigzag
    print(f"\n   Συνολικά zigzag intervals: {len(intervals_zz)}")
    for dim in range(2):
        dim_intervals = [(b, d) for (b, d, dm) in intervals_zz if dm == dim]
        if dim_intervals:
            persistences = [d - b for b, d in dim_intervals]
            print(f"   H{dim}: {len(dim_intervals)} intervals, "
                  f"μέση persistence = {np.mean(persistences):.3f}, "
                  f"μέγιστη = {np.max(persistences):.3f}")
        else:
            print(f"   H{dim}: 0 intervals")

    # Οπτικοποίηση zigzag
    plot_zigzag_diagram(intervals_zz, n_times, "e1_zigzag_persistence.png")

    # -------------------------------------------------------------------
    # 5. Κλασική Persistent Homology (ανά παράθυρο)
    # -------------------------------------------------------------------
    print("\n5. Υπολογισμός Κλασικής Persistent Homology (ανά παράθυρο)...")
    betti_classical = classical_ph_per_window(reduced_clouds, max_radius)

    # Κλασική PH στην ένωση
    print("   Κλασική PH στην ένωση...")
    union_dgms = classical_ph_on_union(reduced_clouds, max_radius)
    plot_classical_diagram(union_dgms, "e1_classical_persistence.png", "(Ένωση S&P 500 Windows)")

    # -------------------------------------------------------------------
    # 6. Σύγκριση & Οπτικοποιήσεις
    # -------------------------------------------------------------------
    print("\n6. Σύγκριση Zigzag vs Κλασική PH...")

    # Σύγκριση Betti curves
    plot_betti_comparison(betti_classical, betti_zigzag_seq, sampled_centers, n_times,
                          "e1_betti_comparison.png")

    # Συγκεντρωτικό γράφημα
    plot_persistence_summary(intervals_zz, betti_classical, betti_zigzag_seq, n_times,
                             "e1_summary_comparison.png")

    # -------------------------------------------------------------------
    # 7. Αποτελέσματα & Αναφορά
    # -------------------------------------------------------------------

    # Αριθμητική σύγκριση
    print("--- Αριθμητική Σύγκριση ---")
    print(f"   Αριθμός sliding windows: {len(point_clouds)}")
    print(f"   Windows στη zigzag: {len(reduced_clouds)}")
    print(f"   Σημεία ανά window: {reduced_clouds[0].shape[0]}")
    print(f"   Ακτίνα Rips: {max_radius:.4f}")

    # Zigzag
    zz_h0 = [(b, d) for b, d, dim in intervals_zz if dim == 0]
    zz_h1 = [(b, d) for b, d, dim in intervals_zz if dim == 1]
    print(f"\n   Zigzag PH:")
    print(f"     H₀ intervals: {len(zz_h0)}")
    print(f"     H₁ intervals: {len(zz_h1)}")
    if zz_h0:
        print(f"     Μέγιστη persistence H₀: {max(d - b for b, d in zz_h0):.1f} zigzag steps")
    if zz_h1:
        print(f"     Μέγιστη persistence H₁: {max(d - b for b, d in zz_h1):.1f} zigzag steps")

    # Κλασική
    mean_b0 = np.mean([b[0] for b in betti_classical])
    mean_b1 = np.mean([b[1] for b in betti_classical])
    std_b0 = np.std([b[0] for b in betti_classical])
    std_b1 = np.std([b[1] for b in betti_classical])
    print(f"\n   Κλασική PH (ανά window):")
    print(f"     Μέσος β₀: {mean_b0:.2f} ± {std_b0:.2f}")
    print(f"     Μέσος β₁: {mean_b1:.2f} ± {std_b1:.2f}")

    print(f"\n   Αποθηκεύτηκαν τα γραφήματα:")
    print(f"     - e1_sliding_windows.png")
    print(f"     - e1_zigzag_schematic.png")
    print(f"     - e1_zigzag_persistence.png")
    print(f"     - e1_classical_persistence.png")
    print(f"     - e1_betti_comparison.png")
    print(f"     - e1_summary_comparison.png")


if __name__ == "__main__":
    run_e1()
