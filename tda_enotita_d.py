import sklearn.utils
import sklearn.utils.validation
orig_check_array = sklearn.utils.validation.check_array
def patched_check_array(array, *args, **kwargs):
    if 'ensure_all_finite' in kwargs:
        kwargs['force_all_finite'] = kwargs.pop('ensure_all_finite')
    return orig_check_array(array, *args, **kwargs)
sklearn.utils.validation.check_array = patched_check_array
sklearn.utils.check_array = patched_check_array

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, roc_auc_score

from ripser import ripser
from persim import plot_diagrams, PersistenceImager
import kmapper as km

import sys
import warnings
import os

# Create target output directory
output_dir = os.path.join('instants', 'd')
os.makedirs(output_dir, exist_ok=True)

# Υποστήριξη ελληνικών στο terminal των Windows
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

def takens_embedding(X_1d, delay=1, dimension=3):
    """Μετατρέπει ένα 1D array σε point cloud μέσω Takens Embedding."""
    N = len(X_1d)
    embedded = []
    for i in range(N - (dimension - 1)*delay):
        pt = [X_1d[i + j*delay] for j in range(dimension)]
        embedded.append(pt)
    return np.array(embedded)

def run_d1():
    print("--- Δ.1 Ανάλυση Πρώτου Dataset (Breast Cancer) ---")
    
    # 1. Φόρτωση και Προεπεξεργασία
    print("1. Φόρτωση και τυποποίηση δεδομένων...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA για να διευκολύνουμε τον υπολογισμό H0, H1 
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    
    # Διαχωρισμός ανά κλάση (0: Malignant, 1: Benign)
    X_malignant = X_pca[y == 0]
    X_benign = X_pca[y == 1]
    
    # 2. Persistent Homology H0, H1 μέσω ripser
    print("2. Υπολογισμός Persistent Homology ανά κλάση (ripser)...")
    res_mal = ripser(X_malignant, maxdim=1)
    res_ben = ripser(X_benign, maxdim=1)
    
    # Persistence Diagrams
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_diagrams(res_mal['dgms'], title="Malignant - Persistence Diagram")
    plt.subplot(1, 2, 2)
    plot_diagrams(res_ben['dgms'], title="Benign - Persistence Diagram")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "d1_persistence_diagrams.png"))
    plt.close()
    
    # Barcodes
    def plot_barcode(dgms, title, filename):
        plt.figure(figsize=(8, 4))
        colors = ['#1f77b4', '#ff7f0e']
        y_pos = 0
        for dim, dgm in enumerate(dgms):
            for b, d in dgm:
                if np.isinf(d): d = 10.0 # Οπτικό όριο
                plt.plot([b, d], [y_pos, y_pos], color=colors[dim], lw=2)
                y_pos += 1
        plt.yticks([])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    plot_barcode(res_mal['dgms'], "Malignant - Barcode", os.path.join(output_dir, "d1_barcode_malignant.png"))
    plot_barcode(res_ben['dgms'], "Benign - Barcode", os.path.join(output_dir, "d1_barcode_benign.png"))
    print("   -> Αποθηκεύτηκαν τα διαγράμματα: instants/d/d1_persistence_diagrams.png και instants/d/d1_barcode_*.png")
    
    # 3. Εφαρμογή Mapper
    print("3. Δημιουργία διαδραστικού Mapper Graph (kmapper)...")
    mapper = km.KeplerMapper(verbose=0)
    lens = mapper.fit_transform(X_scaled, projection=PCA(n_components=2))
    graph = mapper.map(
        lens,
        X_scaled,
        cover=km.Cover(n_cubes=10, perc_overlap=0.3),
        clusterer=DBSCAN(eps=2.5, min_samples=3)
    )
    
    # Φτιάχνουμε custom tooltips για να βλέπουμε κλάσεις
    tooltip_html = np.array([f"Class: {'Benign' if c == 1 else 'Malignant'}" for c in y])
    
    mapper.visualize(
        graph,
        path_html=os.path.join(output_dir, "d1_mapper_graph.html"),
        title="Breast Cancer - KeplerMapper",
        color_values=y,
        color_function_name="Class (0=Mal, 1=Ben)",
        custom_tooltips=tooltip_html
    )
    print("   -> Αποθηκεύτηκε το γράφημα Mapper στο 'instants/d/d1_mapper_graph.html'.")
    
    # 4. TDA ως features -> ML Pipeline
    print("4. Εξαγωγή TDA features (Persistence Images) και εκπαίδευση Random Forest...")
    all_h0 = []
    
    for i in range(X_scaled.shape[0]):
        # Μετατροπή κάθε ασθενούς (30 features) σε point cloud μέσω Takens
        pts = takens_embedding(X_scaled[i], delay=1, dimension=3)
        dgms = ripser(pts, maxdim=1)['dgms']
        
        h0 = dgms[0]
        h0[h0[:, 1] == np.inf, 1] = 5.0 # Αντικατάσταση άπειρου
        
        all_h0.append(h0)
        
    # Fit Imagers (προσθέτουμε ένα dummy point για να διασφαλίσουμε width > 0 στον άξονα birth)
    pimager_h0 = PersistenceImager(pixel_size=0.1)
    pimager_h0.fit(all_h0 + [np.array([[0.0, 0.0], [1.0, 1.0]])])
    pimgs_h0 = pimager_h0.transform(all_h0)
    
    # Flatten τα images σε 1D vector για κάθε ασθενή
    X_features = []
    for img0 in pimgs_h0:
        X_features.append(img0.flatten())
    X_features = np.array(X_features)
    
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("\n=== Αποτελέσματα Ταξινόμησης με TDA Features ===")
    print(f"Random Forest Accuracy : {acc:.4f}")
    print(f"Random Forest AUC      : {auc:.4f}")

def run_d2():
    print("\n--- Δ.2 Ανάλυση Δεύτερου Dataset (S&P 500) ---")
    
    try:
        import yfinance as yf
        import umap
        import pandas as pd
        import seaborn as sns
        import plotly.express as px
        import gudhi
        from sklearn.manifold import TSNE
        from sklearn.pipeline import Pipeline
        from sklearn.utils import resample
        from gtda.time_series import TakensEmbedding
        from gtda.homology import VietorisRipsPersistence
        from gtda.diagrams import PersistenceEntropy
    except ImportError as e:
        print(f"Σφάλμα εισαγωγής βιβλιοθηκών: {e}")
        print("Παρακαλώ εγκαταστήστε: pip install giotto-tda yfinance umap-learn")
        return

    # 1. Φόρτωση δεδομένων S&P 500
    print("1. Φόρτωση δεδομένων S&P 500 (τελευταία 2 έτη)...")
    sp500 = yf.download('^GSPC', period='2y', interval='1d', progress=False)
    if sp500.empty:
        print("Αποτυχία φόρτωσης δεδομένων.")
        return
    
    # Χρησιμοποιούμε τις τιμές κλεισίματος (Close)
    close_prices = sp500['Close'].values.flatten()
    scaler = StandardScaler()
    close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()
    
    # 2. Giotto-TDA Pipeline
    print("2. Δημιουργία Giotto-TDA Pipeline...")
    embedding_dim = 3
    delay = 1
    
    # Το gtda δέχεται (n_samples, n_timestamps)
    X_ts = [close_prices_scaled]
    
    tda_pipeline = Pipeline([
        ('embedding', TakensEmbedding(dimension=embedding_dim, time_delay=delay)),
        ('persistence', VietorisRipsPersistence(homology_dimensions=(0, 1, 2), n_jobs=-1)),
        ('entropy', PersistenceEntropy(normalize=True))
    ])
    
    features = tda_pipeline.fit_transform(X_ts)
    print(f"   -> Persistent Entropy (H0, H1, H2): {features[0]}")
    
    # Εξαγωγή του point cloud για τα βήματα 3 και 4
    embedder = TakensEmbedding(dimension=embedding_dim, time_delay=delay)
    X_embedded = embedder.fit_transform(X_ts)[0] # Παίρνουμε το 1o (και μοναδικό) sample, σχήμα: (n_points, 3)
    
    # [ΠΡΟΣΘΗΚΗ] Επαλήθευση H2 Ομολογίας με GUDHI (Βιβλιοθήκη Gudhi)
    print("   -> Επαλήθευση/Υπολογισμός H2 ομολογίας με GUDHI (Alpha Complex)...")
    alpha_complex = gudhi.AlphaComplex(points=X_embedded)
    st = alpha_complex.create_simplex_tree()
    st.compute_persistence()
    betti_numbers = st.betti_numbers()
    print(f"      Gudhi Betti Numbers: {betti_numbers}")

    # [ΠΡΟΣΘΗΚΗ] Διαδραστικό 3D Plot με Plotly (Βιβλιοθήκη Plotly)
    print("   -> Δημιουργία διαδραστικού 3D γραφήματος με Plotly...")
    df_embedded = pd.DataFrame(X_embedded, columns=['Dim1', 'Dim2', 'Dim3'])
    fig_3d = px.scatter_3d(df_embedded, x='Dim1', y='Dim2', z='Dim3', title="S&P 500 3D Takens Embedding", opacity=0.7)
    fig_3d.write_html(os.path.join(output_dir, "d2_sp500_3d_embedding.html"))
    
    # Save static 3D plot using matplotlib for LaTeX report
    fig_static = plt.figure(figsize=(8, 6))
    ax = fig_static.add_subplot(111, projection='3d')
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c='blue', alpha=0.6, s=10)
    ax.set_title("S&P 500 3D Takens Embedding")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "d2_sp500_3d_embedding.png"), dpi=150)
    plt.close()
    
    # 3. Βαθιά σύγκριση με PCA, t-SNE, UMAP (χρήση Seaborn και Pandas)
    print("3. Σύγκριση TDA point cloud με PCA, t-SNE, UMAP...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_embedded)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_embedded)
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_embedded)
    
    # Οπτικοποίηση με Seaborn
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    df_pca = pd.DataFrame(X_pca, columns=['Dim1', 'Dim2'])
    sns.scatterplot(data=df_pca, x='Dim1', y='Dim2', ax=axs[0], color='blue', alpha=0.6)
    axs[0].set_title("PCA")
    
    df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
    sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', ax=axs[1], color='green', alpha=0.6)
    axs[1].set_title("t-SNE")
    
    df_umap = pd.DataFrame(X_umap, columns=['Dim1', 'Dim2'])
    sns.scatterplot(data=df_umap, x='Dim1', y='Dim2', ax=axs[2], color='red', alpha=0.6)
    axs[2].set_title("UMAP")
    
    plt.suptitle("Μείωση Διάστασης S&P 500")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "d2_sp500_dim_reduction.png"))
    plt.close()
    print("   -> Αποθηκεύτηκαν τα γραφήματα: 'instants/d/d2_sp500_dim_reduction.png' (Seaborn) και 'instants/d/d2_sp500_3d_embedding.html' (Plotly)")
    
    # 4. Bootstrap Resampling (n=100, 80%) για CIs
    print("4. Bootstrap resampling (n=100) για διαστήματα εμπιστοσύνης (95%) της Persistent Entropy...")
    n_iterations = 100
    entropy_samples = []
    
    n_points = len(X_embedded)
    n_sample_size = int(0.8 * n_points)
    
    vr = VietorisRipsPersistence(homology_dimensions=(0, 1, 2), n_jobs=-1)
    pe = PersistenceEntropy(normalize=True)
    
    for i in range(n_iterations):
        # Sampling 80% των σημείων με επανατοποθέτηση
        X_boot = resample(X_embedded, n_samples=n_sample_size, random_state=i)
        
        # Το vr.fit_transform δέχεται input σχήματος (n_samples, n_points, n_dimensions)
        diagrams = vr.fit_transform([X_boot])
        ent = pe.fit_transform(diagrams)[0]
        entropy_samples.append(ent)
        
    entropy_samples = np.array(entropy_samples) # Shape: (100, 3)
    
    ci_lower = np.percentile(entropy_samples, 2.5, axis=0)
    ci_upper = np.percentile(entropy_samples, 97.5, axis=0)
    mean_entropy = np.mean(entropy_samples, axis=0)
    
    print("\n   === Διαστήματα Εμπιστοσύνης 95% (Persistent Entropy) ===")
    dims = ["H0", "H1", "H2"]
    for i in range(3):
        print(f"   {dims[i]}: Mean = {mean_entropy[i]:.4f}, 95% CI = [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]")

if __name__ == "__main__":
    run_d1()
    run_d2()
