import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ============================================================
# 2. DUOMENŲ UŽKROVIMAS
# ============================================================

def load_sunspot_data(filepath='../data/sunspot.txt'):
    try:
        data = pd.read_csv(filepath, sep=r'\s+', header=None, names=['Metai', 'Aktyvumas'])
        print(f"[OK] Duomenys užkrauti: {len(data)} eilutės")
        print(f"     Metų diapazonas : {int(data['Metai'].min())} – {int(data['Metai'].max())}")
        print(f"     Aktyvumo diapazonas: {data['Aktyvumas'].min():.1f} – {data['Aktyvumas'].max():.1f}")
        return data
    except Exception as e:
        raise RuntimeError(f"Klaida skaitant failą: {e}")


sunspot_data = load_sunspot_data()


# ============================================================
# 4. SAULĖS DĖMIŲ AKTYVUMO GRAFIKAS
# ============================================================

def plot_sunspot_activity(data):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(data['Metai'], data['Aktyvumas'], color='crimson', linewidth=0.9, label='Aktyvumas')
    ax.fill_between(data['Metai'], data['Aktyvumas'], alpha=0.15, color='crimson')
    ax.set_title('Saulės dėmių aktyvumas (1700–2014 m.)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metai', fontsize=12)
    ax.set_ylabel('Dėmių kiekis', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

#plot_sunspot_activity(sunspot_data)


# ============================================================
# 5. MATRICŲ P IR T PARUOŠIMAS
# ============================================================

def prepare_matrices(data, n):
    """
    Sukuria įvesties matricą P (eilė n) ir išvesties vektorių T.
    P[k] = [a(k-n), ..., a(k-1)],  T[k] = a(k)
    """
    series = data['Aktyvumas'].values.astype(float)
    num_samples = len(series) - n
    P = np.array([series[k - n:k] for k in range(n, len(series))])
    T = series[n:].reshape(-1, 1)
    return P, T


# ============================================================
# 6. 3D KORELIACIJŲ DIAGRAMA  (n=2)
# ============================================================

def plot_3d_correlation(P, T, n=2):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = P[:, 0], P[:, 1], T.flatten()
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.55, s=8)
    ax.set_title(f'Saulės dėmių aktyvumo koreliacija (n={n})', fontsize=13, fontweight='bold')
    ax.set_xlabel('a(k-2) – aktyvumas prieš 2 m.', fontsize=10)
    ax.set_ylabel('a(k-1) – aktyvumas prieš 1 m.', fontsize=10)
    ax.set_zlabel('a(k) – prognozuojamas aktyvumas', fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.55, label='Aktyvumo lygis')
    plt.tight_layout()
    plt.show()

P2, T2 = prepare_matrices(sunspot_data, n=2)
print(f"\nMatricos (n=2): P={P2.shape}, T={T2.shape}")
#plot_3d_correlation(P2, T2, n=2)


# ============================================================
# 7. DUOMENŲ SKAIDYMAS
# ============================================================

def split_data(P, T, train_size=200):
    return P[:train_size], T[:train_size], P[train_size:], T[train_size:]


# ============================================================
# 8–9. TIESINĖ REGRESIJA (mažiausių kvadratų metodas)
# ============================================================

def train_linear_model(Pu, Tu):
    """Apskaičiuoja svorius analitiškai: w = (P^T P)^{-1} P^T T"""
    ones = np.ones((Pu.shape[0], 1))
    Pu_aug = np.hstack([Pu, ones])
    weights, _, _, _ = np.linalg.lstsq(Pu_aug, Tu, rcond=None)
    w = weights[:-1].flatten()
    b = float(weights[-1].flatten()[0])
    return w, b


def linear_predict(P, w, b):
    return (P @ w + b).reshape(-1, 1)


# --- n=2 ---
Pu2, Tu2, Pv2, Tv2 = split_data(P2, T2, train_size=200)
w_lin2, b_lin2 = train_linear_model(Pu2, Tu2)
print("\n--- Tiesinės regresijos koeficientai (n=2) ---")
for i, wi in enumerate(w_lin2):
    print(f"  w{i+1} = {wi:.4f}")
print(f"  b  = {b_lin2:.4f}")


# ============================================================
# 10–12. VERIFIKACIJA IR KLAIDŲ ANALIZĖ
# ============================================================

def plot_verification(T_actual, T_pred, title, years=None):
    fig, ax = plt.subplots(figsize=(13, 5))
    x = years if years is not None else np.arange(len(T_actual))
    ax.plot(x, T_actual, label='Tikrieji duomenys (T)', color='steelblue', linewidth=1.1)
    ax.plot(x, T_pred,  label='Prognozė (Ts)', color='crimson', linestyle='--', linewidth=1.1, alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Metai' if years is not None else 'Laiko taškas', fontsize=11)
    ax.set_ylabel('Saulės dėmių kiekis', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_errors(errors, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Klaidos laike
    axes[0].plot(errors, color='darkorange', linewidth=0.9)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_title(f'{title_prefix}: Prognozės klaidos e(k)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Laiko taškai', fontsize=11)
    axes[0].set_ylabel('Klaida (Tikra − Prognozė)', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Histograma
    axes[1].hist(errors.flatten(), bins=25, color='skyblue', edgecolor='black', alpha=0.85)
    axes[1].axvline(0, color='red', linewidth=1.4, label='0')
    axes[1].axvline(np.mean(errors), color='navy', linewidth=1.4,
                    linestyle='--', label=f'Vidurkis={np.mean(errors):.1f}')
    axes[1].set_title(f'{title_prefix}: Klaidų histograma', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Klaidos reikšmė', fontsize=11)
    axes[1].set_ylabel('Dažnumas', fontsize=11)
    axes[1].legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def calculate_metrics(errors, label):
    e = errors.flatten()
    mse = np.mean(e ** 2)
    mad = np.median(np.abs(e))

    print(f"\n{'='*45}")
    print(f"  Metrikos — {label}")
    print(f"{'='*45}")
    print(f"  MSE  : {mse:>10.2f}   {'[OK]' if mse <= 300 else '[VIRŠYTA RIBA 300]'}")
    print(f"  MAD  : {mad:>10.2f}")
    return dict(mse=mse, mad=mad)


# Apmokymo verifikacija
years_u = sunspot_data['Metai'].values[2:202]
Tsu2 = linear_predict(Pu2, w_lin2, b_lin2)
#plot_verification(Tu2, Tsu2, "Tiesinė regresija – apmokymo rinkinys (n=2)", years_u)

# Testavimo verifikacija
years_v = sunspot_data['Metai'].values[202:]
Tsv2 = linear_predict(Pv2, w_lin2, b_lin2)
#plot_verification(Tv2, Tsv2, "Tiesinė regresija – testavimo rinkinys (n=2)", years_v)

eu2 = Tu2 - Tsu2
ev2 = Tv2 - Tsv2

#plot_errors(eu2, "Apmokymo rinkinys (n=2)")
#plot_errors(ev2, "Testavimo rinkinys (n=2)")

metrics_u2 = calculate_metrics(eu2, "Apmokymo duomenys (tiesinė regresija, n=2)")
metrics_v2 = calculate_metrics(ev2, "Testavimo duomenys (tiesinė regresija, n=2)")


# ============================================================
# 15–17. TIESINIS NEURONAS SU NORMALIZACIJA
# ============================================================

class TiesinisNeuronas:
    def __init__(self, lr=0.05, n_iter=1000, mse_goal=300, norm='zscore'):
        self.lr = lr
        self.n_iter = n_iter
        self.mse_goal = mse_goal
        self.norm = norm

        self.w_ = None
        self.b_ = None

        # Normalizacijos parametrai
        self.X_mean = None
        self.X_std  = None
        self.y_mean = None
        self.y_std  = None

        # Metrikos po kiekvienos epochos
        self.history = {
            'mse':  [],
            'mad':  [],
            'bias': [],
            'rmse': []
        }

    # ----------------------------------------------------------
    def _normalize_X(self, X):
        if self.norm == 'zscore':
            return (X - self.X_mean) / (self.X_std + 1e-8)
        else:  # minmax
            return (X - self.X_min) / (self.X_range + 1e-8)

    def _normalize_y(self, y):
        if self.norm == 'zscore':
            return (y - self.y_mean) / (self.y_std + 1e-8)
        else:
            return (y - self.y_min) / (self.y_range + 1e-8)

    def _denormalize_y(self, y_norm):
        if self.norm == 'zscore':
            return y_norm * self.y_std + self.y_mean
        else:
            return y_norm * self.y_range + self.y_min

    # ----------------------------------------------------------
    def fit(self, X, y):
        y_flat = y.flatten()

        # Normalizacijos statistikos
        if self.norm == 'zscore':
            self.X_mean = X.mean(axis=0)
            self.X_std  = X.std(axis=0)
            self.y_mean = y_flat.mean()
            self.y_std  = y_flat.std()
        else:
            self.X_min   = X.min(axis=0)
            self.X_range = X.max(axis=0) - X.min(axis=0)
            self.y_min   = y_flat.min()
            self.y_range = y_flat.max() - y_flat.min()

        Xn = self._normalize_X(X)
        yn = self._normalize_y(y_flat)

        # Svorių inicializacija
        rng = np.random.default_rng(42)
        self.w_ = rng.normal(0, 0.01, Xn.shape[1])
        self.b_ = 0.0

        for epoch in range(self.n_iter):
            # Prognozė normalizuotoje erdvėje
            pred_n = Xn @ self.w_ + self.b_
            err_n  = yn - pred_n

            # Gradientinis nusileidimas 
            self.w_ += self.lr * (Xn.T @ err_n) / len(yn)
            self.b_ += self.lr * err_n.mean()

            # Metrikos org skalėje
            pred_orig = self._denormalize_y(pred_n)
            e_orig = y_flat - pred_orig
            mse  = np.mean(e_orig ** 2)
            mad  = np.median(np.abs(e_orig))

            self.history['mse'].append(mse)
            self.history['mad'].append(mad)

            if np.isnan(mse) or np.isinf(mse):
                print(f"  Divergavimas epochoje {epoch}.")
                return self

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"  Epocha {epoch+1:>5} | MSE={mse:>8.2f} | MAD={mad:>7.2f} |")

            if mse <= self.mse_goal:
                print(f"  Tikslas pasiektas epochoje {epoch+1}: MSE={mse:.2f}")
                break

        return self

    # ----------------------------------------------------------
    def predict(self, X):
        Xn = self._normalize_X(X)
        pred_n = Xn @ self.w_ + self.b_
        return self._denormalize_y(pred_n).reshape(-1, 1)

    def get_weights_original_scale(self):
        if self.norm == 'zscore':
            w_orig = self.w_ * self.y_std / (self.X_std + 1e-8)
            b_orig = (self.b_ * self.y_std + self.y_mean
                      - np.sum(w_orig * self.X_mean))
        else:
            w_orig = self.w_ * self.y_range / (self.X_range + 1e-8)
            b_orig = (self.b_ * self.y_range + self.y_min
                      - np.sum(w_orig * self.X_min))
        return w_orig, float(b_orig)


# ============================================================
# SVORIŲ PALYGINIMAS: ANALITINIS vs ITERACINIS
# ============================================================

def compare_weights(w_lr, b_lr, w_nn, b_nn, n_order):
    all_w_lr  = np.append(w_lr,  b_lr)
    all_w_nn  = np.append(w_nn,  b_nn)
    labels    = [f'w{i+1}' for i in range(len(w_lr))] + ['b']
    abs_diff  = np.abs(all_w_lr - all_w_nn)
    rel_diff  = np.where(
        np.abs(all_w_lr) > 1e-6,
        abs_diff / np.abs(all_w_lr) * 100,
        np.nan
    )

    col = 22
    print(f"\n{'='*80}")
    print(f"  SVORIŲ PALYGINIMAS  (n={n_order})")
    print(f"{'='*80}")
    print(f"  {'Koef.':<{col}} {'Ties.reg.':>15} {'Neuronas':>15} "
          f"{'|delta|':>10} {'delta %':>8}")
    print(f"  {'-'*74}")
    for lbl, wlr, wnn, ad, rd in zip(labels, all_w_lr, all_w_nn, abs_diff, rel_diff):
        rd_str = f"{rd:>7.2f}%" if not np.isnan(rd) else "   n/a  "
        print(f"  {lbl:<{col}} {wlr:>15.4f} {wnn:>15.4f} {ad:>10.4f} {rd_str}")
    print(f"{'='*80}")

# ============================================================
# APMOKYMAS SU SKIRTINGOMIS EILĖMIS
# ============================================================

def run_experiment(data, n_order, train_size=200,
                   lr=0.05, n_iter=1000, mse_goal=250, label=''):
    print(f"\n{'#'*55}")
    print(f"  EKSPERIMENTAS: n={n_order}, lr={lr}  {label}")
    print(f"{'#'*55}")

    P, T = prepare_matrices(data, n_order)
    Pu, Tu, Pv, Tv = split_data(P, T, train_size)

    #Tiesinė regresija
    w_lr, b_lr = train_linear_model(Pu, Tu)
    Tsu_lr = linear_predict(Pu, w_lr, b_lr)
    Tsv_lr = linear_predict(Pv, w_lr, b_lr)
    m_u_lr = calculate_metrics(Tu - Tsu_lr, f"LinReg apmokymas n={n_order}")
    m_v_lr = calculate_metrics(Tv - Tsv_lr, f"LinReg testavimas n={n_order}")

    # --- Neuronas ---
    neuron = TiesinisNeuronas(lr=lr, n_iter=n_iter, mse_goal=mse_goal)
    neuron.fit(Pu, Tu)

    Tsu_nn = neuron.predict(Pu)
    Tsv_nn = neuron.predict(Pv)

    eu_nn = Tu - Tsu_nn
    ev_nn = Tv - Tsv_nn

    m_u_nn = calculate_metrics(eu_nn, f"Neuronas apmokymas n={n_order}")
    m_v_nn = calculate_metrics(ev_nn, f"Neuronas testavimas n={n_order}")

    w_orig, b_orig = neuron.get_weights_original_scale()
    compare_weights(w_lr, b_lr, w_orig, b_orig, n_order)

    # --- Mokymosi eigos grafikas ---
    #plot_learning_curve(neuron, n_order, lr, mse_goal)

    # --- Verifikacijos grafikai ---
    yrs = sunspot_data['Metai'].values
    #plot_verification(Tu, Tsu_nn, f"Neuronas – apmokymo rinkinys (n={n_order})", yrs[n_order: n_order + train_size])
    #plot_verification(Tv, Tsv_nn, f"Neuronas – testavimo rinkinys (n={n_order})", yrs[n_order + train_size:])

    #plot_errors(eu_nn, f"Neuronas, apmokymo rinkinys (n={n_order})")
    #plot_errors(ev_nn, f"Neuronas, testavimo rinkinys (n={n_order})")

    return {
        'neuron':   neuron,
        'linreg':  {'w': w_lr, 'b': b_lr,
                    'metrics_u': m_u_lr, 'metrics_v': m_v_lr},
        'nn':       {'w': w_orig, 'b': b_orig,
                     'metrics_u': m_u_nn, 'metrics_v': m_v_nn},
    }


def plot_learning_curve(neuron, n_order, lr, mse_goal):
    hist = neuron.history
    epochs = range(1, len(hist['mse']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MSE ir RMSE eiga
    axes[0].semilogy(epochs, hist['mse'],  label='MSE',  color='navy')
    axes[0].semilogy(epochs, hist['rmse'], label='RMSE', color='teal', linestyle='--')
    axes[0].axhline(mse_goal, color='red', linestyle=':', label=f'MSE tikslas ({mse_goal})')
    axes[0].set_title(f'Mokymosi eiga (n={n_order}, lr={lr})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epocha', fontsize=11)
    axes[0].set_ylabel('Klaida (log skalė)', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, which='both', linestyle='--', alpha=0.5)

    # MAD ir Bias eiga
    axes[1].plot(epochs, hist['mad'],  label='MAD',  color='darkorange')
    axes[1].plot(epochs, hist['bias'], label='Bias', color='purple', linestyle='--')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title(f'MAD ir Bias eiga (n={n_order}, lr={lr})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epocha', fontsize=11)
    axes[1].set_ylabel('Klaidos matavimas', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# ============================================================
# 18–19. LR PALYGINIMO EKSPERIMENTAS
# ============================================================

def experiment_lr_comparison(data, n_order=10, train_size=200):
    P, T = prepare_matrices(data, n_order)
    Pu, Tu, _, _ = split_data(P, T, train_size)

    lrs = [0.5, 0.1, 0.05, 0.01, 0.005]
    fig, ax = plt.subplots(figsize=(12, 6))

    for lr in lrs:
        m = TiesinisNeuronas(lr=lr, n_iter=300, mse_goal=10)
        m.fit(Pu, Tu)
        if not np.isnan(m.history['mse'][-1]):
            ax.semilogy(m.history['mse'], label=f'lr={lr}')
        else:
            print(f"  lr={lr}: divergavimas")

    ax.axhline(300, color='red', linestyle=':', label='MSE riba (300)')
    ax.set_title(f'Mokymosi greičio lr įtaka konvergencijai (n={n_order})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Epocha', fontsize=11)
    ax.set_ylabel('MSE (log skalė)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ============================================================
# 20. MODELIO EILĖS n PALYGINIMAS
# ============================================================

def experiment_n_order_comparison(data, n_values=(2, 6, 10),
                                   train_size=200, lr=0.05, n_iter=1000):
    N_GOAL = {2: 200, 6: 200, 10: 200}

    results = {}
    print("\n" + "="*55)
    print("  MODELIO EILĖS ĮTAKA PROGNOZAVIMO KOKYBEI")
    print("="*55)

    for n in n_values:
        mse_goal = N_GOAL.get(n, 220)
        P, T = prepare_matrices(data, n)
        Pu, Tu, Pv, Tv = split_data(P, T, train_size)

        neuron = TiesinisNeuronas(lr=lr, n_iter=n_iter, mse_goal=mse_goal)
        neuron.fit(Pu, Tu)

        Tsv = neuron.predict(Pv)
        Tsu = neuron.predict(Pu)
        mse_u = float(np.mean((Tu - Tsu) ** 2))
        mse_v = float(np.mean((Tv - Tsv) ** 2))
        mad_v = float(np.median(np.abs((Tv - Tsv).flatten())))

        results[n] = dict(Tv=Tv, Tsv=Tsv, mse_u=mse_u, mse_v=mse_v,
                          mad_v=mad_v, history=neuron.history)
        print(f"  n={n:>2} | Apmokymo MSE={mse_u:>7.2f} | "
              f"Testavimo MSE={mse_v:>7.2f} | Testavimo MAD={mad_v:>6.2f}")

    # Prognozės grafikai šalia vienas kito
    fig, axes = plt.subplots(len(n_values), 1, figsize=(14, 4 * len(n_values)), sharex=False)
    for ax, n in zip(axes, n_values):
        r = results[n]
        ax.plot(r['Tv'],  label='Tikrieji', color='steelblue', linewidth=1.0)
        ax.plot(r['Tsv'], label='Prognozė', color='crimson',
                linestyle='--', linewidth=1.0, alpha=0.85)
        ax.set_title(f"n={n} | Testavimo MSE={r['mse_v']:.1f}, MAD={r['mad_v']:.1f}",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Laiko taškai', fontsize=10)
        ax.set_ylabel('Dėmių kiekis', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # MSE/MAD suvestinė juostinė diagrama
    ns   = list(n_values)
    mses = [results[n]['mse_v'] for n in ns]
    mads = [results[n]['mad_v'] for n in ns]
    x    = np.arange(len(ns))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, mses, w, label='MSE (testavimas)', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + w/2, mads, w, label='MAD (testavimas)', color='darkorange', alpha=0.85)
    ax.axhline(300, color='red', linestyle=':', label='MSE riba (300)')
    ax.bar_label(bars1, fmt='%.1f', fontsize=9)
    ax.bar_label(bars2, fmt='%.1f', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'n={n}' for n in ns], fontsize=11)
    ax.set_title('Modelio eilės n įtaka prognozavimo kokybei', fontsize=13, fontweight='bold')
    ax.set_ylabel('Klaidos reikšmė', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return results


# ============================================================
# PAGRINDINĖ PROGRAMA
# ============================================================

print("\n" + "="*55)
print("  PALEIDŽIAMI EKSPERIMENTAI")
print("="*55)

# Eksperimentas n=2
res2  = run_experiment(sunspot_data, n_order=2,  lr=0.05, n_iter=1000,  mse_goal=200)

# Eksperimentas n=6
#res6  = run_experiment(sunspot_data, n_order=6,  lr=0.05, n_iter=1000,  mse_goal=200)

# Eksperimentas n=10
#res10 = run_experiment(sunspot_data, n_order=10, lr=0.05, n_iter=1000,  mse_goal=200)

# lr palyginimas
print("\n--- lr palyginimo eksperimentas ---")
experiment_lr_comparison(sunspot_data, n_order=10)

# n eilės palyginimas
experiment_n_order_comparison(sunspot_data, n_values=(2, 6, 10))

print("\n[✓] Scenarijus baigtas.")