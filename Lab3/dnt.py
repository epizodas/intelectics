import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. DUOMENŲ UŽKROVIMAS IR APŽVALGA
# ============================================================

def load_and_describe(filepath='../data/csgo.csv'):
    df = pd.read_csv(filepath, index_col=0)
    print(f"[OK] Įkelta eilučių: {len(df)}, stulpelių: {len(df.columns)}")
    print(f"\nStulpeliai: {df.columns.tolist()}")
    print(f"\nKills statistika prieš diskretizaciją:")
    print(df['kills'].describe())
    return df


# ============================================================
# 2. TIKSLO ATRIBUTO DISKRETIZACIJA
#    kills -> 4 kategorijos pagal kvartilius
#    0–10  -> 0 "Žemas"
#    11–14 -> 1 "Vidutinis"
#    15–18 -> 2 "Aukštas"
#    19+   -> 3 "Labai aukštas"
# ============================================================

KILL_BINS   = [-1, 10, 14, 18, 999]
KILL_LABELS = [0, 1, 2, 3]
KILL_NAMES  = {0: '0–10 (Žemas)', 1: '11–14 (Vidutinis)',
               2: '15–18 (Aukštas)', 3: '19+ (Labai aukštas)'}

def discretize_kills(df):
    df = df.copy()
    df['kills_cat'] = pd.cut(df['kills'],
                             bins=KILL_BINS,
                             labels=KILL_LABELS).astype(int)
    print("\nKills diskretizacija:")
    for k, name in KILL_NAMES.items():
        cnt = (df['kills_cat'] == k).sum()
        pct = cnt / len(df) * 100
        print(f"  {name:25s}: {cnt:5d} ({pct:.1f}%)")
    return df


def prepare_features(df):
    drop_cols = ['kills', 'kills_cat', 'day', 'month', 'year', 'date', 'rownames', 'team_a_rounds', 'team_b_rounds']

    y = df['kills_cat'].values

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Koduojame 'result' ir 'map' jei yra
    le = LabelEncoder()
    for col in ['result', 'map']:
        if col in X.columns:
            X[col] = le.fit_transform(X[col].astype(str))

    feature_names = X.columns.tolist()
    X = X.values.astype(float)

    print(f"\nPožymių skaičius: {X.shape[1]}")
    print(f"Požymiai: {feature_names}")
    return X, y, feature_names


# ============================================================
# 3. DNT REALIZACIJA (NumPy)
#    Architektūra: [input] -> [H1] -> [H2] -> [output(4)]
#    Aktyvacija  : ReLU (paslėpti sluoksniai), Softmax (išvestis)
# ============================================================

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_act(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z) ** 2

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def one_hot(y, n_classes):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

def cross_entropy_loss(y_oh, y_pred):
    return -np.mean(np.sum(y_oh * np.log(y_pred + 1e-9), axis=1))


class DNT:
    def __init__(self, layers=(64, 32), lr=0.01, epochs=200,
                 activation='relu', batch_size=64, dropout=0.0):
        self.layers     = layers
        self.lr         = lr
        self.epochs     = epochs
        self.activation = activation
        self.batch_size = batch_size
        self.dropout    = dropout
        self.weights    = []
        self.biases     = []
        self.loss_history = []
        self.val_loss_history = []

    def _act(self, z):
        if self.activation == 'relu':    return relu(z)
        if self.activation == 'sigmoid': return sigmoid(z)
        if self.activation == 'tanh':    return tanh_act(z)
        return relu(z)

    def _act_d(self, z):
        if self.activation == 'relu':    return relu_deriv(z)
        if self.activation == 'sigmoid': return sigmoid_deriv(z)
        if self.activation == 'tanh':    return tanh_deriv(z)
        return relu_deriv(z)

    def _init_weights(self, n_input, n_output):
        sizes = [n_input] + list(self.layers) + [n_output]
        self.weights = []
        self.biases  = []
        for i in range(len(sizes) - 1):
            # He inicializacija (tinka ReLU)
            scale = np.sqrt(2.0 / sizes[i])
            self.weights.append(np.random.randn(sizes[i], sizes[i+1]) * scale)
            self.biases.append(np.zeros((1, sizes[i+1])))

    def _forward(self, X, training=False):
        self._zs = []
        self._as = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            self._zs.append(z)
            if i < len(self.weights) - 1:
                a = self._act(z)
                # Dropout
                if training and self.dropout > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout) / (1 - self.dropout)
                    a = a * mask
            else:
                a = softmax(z)
            self._as.append(a)
        return self._as[-1]

    def _backward(self, y_oh):
        n = y_oh.shape[0]
        # Gradientas išvesties sluoksnio (softmax + cross-entropy)
        delta = self._as[-1] - y_oh
        grads_w = []
        grads_b = []
        for i in reversed(range(len(self.weights))):
            dW = self._as[i].T @ delta / n
            db = delta.mean(axis=0, keepdims=True)
            grads_w.insert(0, dW)
            grads_b.insert(0, db)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._act_d(self._zs[i-1])
        return grads_w, grads_b

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_classes = len(np.unique(y_train))
        self._init_weights(X_train.shape[1], n_classes)
        self.loss_history     = []
        self.val_loss_history = []

        for epoch in range(self.epochs):
            # Mini-batch SGD
            idx = np.random.permutation(len(X_train))
            X_s, y_s = X_train[idx], y_train[idx]
            for start in range(0, len(X_s), self.batch_size):
                Xb = X_s[start:start + self.batch_size]
                yb = y_s[start:start + self.batch_size]
                y_oh = one_hot(yb, n_classes)
                self._forward(Xb, training=True)
                gw, gb = self._backward(y_oh)
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * gw[i]
                    self.biases[i]  -= self.lr * gb[i]

            # Epochos nuostoliai
            y_oh_full = one_hot(y_train, n_classes)
            pred_full = self._forward(X_train, training=False)
            loss = cross_entropy_loss(y_oh_full, pred_full)
            self.loss_history.append(loss)

            if X_val is not None:
                y_oh_val = one_hot(y_val, n_classes)
                pred_val = self._forward(X_val, training=False)
                val_loss = cross_entropy_loss(y_oh_val, pred_val)
                self.val_loss_history.append(val_loss)

        return self

    def predict(self, X):
        return np.argmax(self._forward(X, training=False), axis=1)

    def predict_proba(self, X):
        return self._forward(X, training=False)


# ============================================================
# 4. 10 INTERVALŲ KRYŽMINĖ PATIKRA
# ============================================================

def cross_validate(model_fn, X, y, n_splits=10, scaler_fn=StandardScaler):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs   = []
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        sc = scaler_fn()
        X_tr = sc.fit_transform(X_tr)
        X_vl = sc.transform(X_vl)

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_vl)

        acc  = accuracy_score(y_vl, y_pred)
        loss = cross_entropy_loss(one_hot(y_vl, 4),
                                  model.predict_proba(X_vl))
        fold_accs.append(acc)
        fold_losses.append(loss)
        print(f"  Intervalas {fold+1:>2}: tikslumas={acc:.4f}, nuostoliai={loss:.4f}")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"  {'─'*44}")
    print(f"  Vidutinis tikslumas : {mean_acc:.4f} ± {std_acc:.4f}")
    return fold_accs, fold_losses, mean_acc, std_acc


# ============================================================
# PAGALBINĖS VIZUALIZACIJOS
# ============================================================

def plot_cv_results(fold_accs, fold_losses, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    folds = np.arange(1, len(fold_accs) + 1)

    axes[0].bar(folds, fold_accs, color='steelblue', alpha=0.85, edgecolor='black')
    axes[0].axhline(np.mean(fold_accs), color='red', linestyle='--',
                    label=f'Vidurkis={np.mean(fold_accs):.4f}')
    axes[0].set_title(f'{title}\nTikslumas pagal intervalą', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Intervalas', fontsize=11)
    axes[0].set_ylabel('Tikslumas', fontsize=11)
    axes[0].set_xticks(folds)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    axes[1].bar(folds, fold_losses, color='darkorange', alpha=0.85, edgecolor='black')
    axes[1].axhline(np.mean(fold_losses), color='navy', linestyle='--',
                    label=f'Vidurkis={np.mean(fold_losses):.4f}')
    axes[1].set_title(f'{title}\nNuostoliai pagal intervalą', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Intervalas', fontsize=11)
    axes[1].set_ylabel('Cross-entropy nuostoliai', fontsize=11)
    axes[1].set_xticks(folds)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_experiment_comparison(results_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66', '#77BEDB']
    bars = ax.barh(results_df['Eksperimentas'], results_df['Vid. tikslumas'],
                   color=colors[:len(results_df)], alpha=0.85, edgecolor='black')

    # Patobulėjimo rodyklės
    baseline = results_df['Vid. tikslumas'].iloc[0]
    for i, (_, row) in enumerate(results_df.iterrows()):
        delta = row['Vid. tikslumas'] - baseline
        color = 'green' if delta >= 0 else 'red'
        sign  = '+' if delta >= 0 else ''
        ax.text(row['Vid. tikslumas'] + 0.002, i,
                f"{row['Vid. tikslumas']:.4f}  ({sign}{delta*100:.2f}%)",
                va='center', fontsize=9, color=color)

    ax.axvline(baseline, color='navy', linestyle='--', linewidth=1.2,
               label=f'Pradinis: {baseline:.4f}')
    ax.set_xlabel('Vidutinis tikslumas (10-fold CV)', fontsize=11)
    ax.set_title('Eksperimentų palyginimas – tikslumas', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, min(1.0, results_df['Vid. tikslumas'].max() + 0.08))
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    # ── Duomenys ────────────────────────────────────────────
    df = load_and_describe('../data/csgo.csv')
    df = discretize_kills(df)
    X, y, feat_names = prepare_features(df)

    # Bendrasis train/test padalinimas (palyginimui paskutiniame žingsnyje)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    sc_global = StandardScaler()
    X_train_s = sc_global.fit_transform(X_train)
    X_test_s  = sc_global.transform(X_test)

    experiment_log = []   # {'Eksperimentas', 'Aprašymas', 'Vid. tikslumas', 'Std'}

    # ────────────────────────────────────────────────────────
    # PRADINIS MODELIS
    # Architektūra: [n_feat] -> 64 -> 32 -> 4
    # lr=0.01, epochs=200, ReLU, batch=64
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  PRADINIS MODELIS")
    print("="*55)

    def base_model():
        return DNT(layers=(64, 32), lr=0.01, epochs=200,
                   activation='relu', batch_size=64)

    accs0, losses0, mean0, std0 = cross_validate(base_model, X_train, y_train)
    plot_cv_results(accs0, losses0, 'Pradinis modelis: [64->32], lr=0.01, ReLU')
    experiment_log.append({
        'Eksperimentas': 'E0 – Pradinis',
        'Aprašymas': '[64->32], lr=0.01, ReLU, batch=64, ep=200',
        'Vid. tikslumas': mean0, 'Std': std0
    })

    # ────────────────────────────────────────────────────────
    # 1 EKSPERIMENTAS: didesnis lr (0.05)
    # Hipotezė: lėtesnis pradinis lr gali stabdyti konvergenciją
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EKS. 1: Didesnis mokymosi greitis lr=0.05")
    print("="*55)

    def exp1_model():
        return DNT(layers=(64, 32), lr=0.05, epochs=200,
                   activation='relu', batch_size=64)

    accs1, losses1, mean1, std1 = cross_validate(exp1_model, X_train, y_train)
    plot_cv_results(accs1, losses1, 'Eks.1: lr=0.05')
    experiment_log.append({
        'Eksperimentas': 'E1 – lr=0.05',
        'Aprašymas': '[64->32], lr=0.05, ReLU, batch=64, ep=200',
        'Vid. tikslumas': mean1, 'Std': std1
    })

    # ────────────────────────────────────────────────────────
    # 2 EKSPERIMENTAS: gilesnis tinklas [128->64->32]
    # Hipotezė: daugiau parametrų – geresnė aproksimacija
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EKS. 2: Gilesnis tinklas [128->64->32]")
    print("="*55)

    def exp2_model():
        return DNT(layers=(128, 64, 32), lr=0.05, epochs=200,
                   activation='relu', batch_size=64)

    accs2, losses2, mean2, std2 = cross_validate(exp2_model, X_train, y_train)
    plot_cv_results(accs2, losses2, 'Eks.2: [128->64->32], lr=0.05')
    experiment_log.append({
        'Eksperimentas': 'E2 – Gilesnis tinklas',
        'Aprašymas': '[128->64->32], lr=0.05, ReLU, batch=64, ep=200',
        'Vid. tikslumas': mean2, 'Std': std2
    })

    # ────────────────────────────────────────────────────────
    # 3 EKSPERIMENTAS: Tanh aktyvacija vietoj ReLU
    # Hipotezė: Tanh gali geriau veikti su normalizuotais duomenimis
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EKS. 3: Tanh aktyvacija")
    print("="*55)

    def exp3_model():
        return DNT(layers=(128, 64, 32), lr=0.05, epochs=200,
                   activation='tanh', batch_size=64)

    accs3, losses3, mean3, std3 = cross_validate(exp3_model, X_train, y_train)
    plot_cv_results(accs3, losses3, 'Eks.3: [128->64->32], lr=0.05, Tanh')
    experiment_log.append({
        'Eksperimentas': 'E3 – Tanh aktyvacija',
        'Aprašymas': '[128->64->32], lr=0.05, Tanh, batch=64, ep=200',
        'Vid. tikslumas': mean3, 'Std': std3
    })

    # ────────────────────────────────────────────────────────
    # 4 EKSPERIMENTAS: Mažesnis batch + daugiau epochų
    # Hipotezė: smulkesnis batch leidžia tikslesnį gradientą
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EKS. 4: Mažesnis batch (32) + 400 epochų")
    print("="*55)

    def exp4_model():
        return DNT(layers=(128, 64, 32), lr=0.05, epochs=400,
                   activation='relu', batch_size=32)

    accs4, losses4, mean4, std4 = cross_validate(exp4_model, X_train, y_train)
    plot_cv_results(accs4, losses4, 'Eks.4: batch=32, ep=400')
    experiment_log.append({
        'Eksperimentas': 'E4 – Mažesnis batch',
        'Aprašymas': '[128->64->32], lr=0.05, ReLU, batch=32, ep=400',
        'Vid. tikslumas': mean4, 'Std': std4
    })

    # ────────────────────────────────────────────────────────
    # 5 EKSPERIMENTAS: Dropout reguliarizacija (0.2)
    # Hipotezė: dropout sumažina persimokymą
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EKS. 5: Dropout = 0.2")
    print("="*55)

    def exp5_model():
        return DNT(layers=(128, 64, 32), lr=0.05, epochs=400,
                   activation='relu', batch_size=32, dropout=0.2)

    accs5, losses5, mean5, std5 = cross_validate(exp5_model, X_train, y_train)
    plot_cv_results(accs5, losses5, 'Eks.5: Dropout=0.2')
    experiment_log.append({
        'Eksperimentas': 'E5 – Dropout 0.2',
        'Aprašymas': '[128->64->32], lr=0.05, ReLU, batch=32, ep=400, drop=0.2',
        'Vid. tikslumas': mean5, 'Std': std5
    })

    # ────────────────────────────────────────────────────────
    # EKSPERIMENTŲ SUVESTINĖ
    # ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(experiment_log)
    print("\n" + "="*70)
    print(f"{'EKSPERIMENTŲ SUVESTINĖ':^70}")
    print("="*70)
    print(results_df[['Eksperimentas', 'Aprašymas',
                       'Vid. tikslumas', 'Std']].to_string(index=False,
        formatters={'Vid. tikslumas': '{:.4f}'.format,
                    'Std':            '{:.4f}'.format}))
    delta = results_df['Vid. tikslumas'].iloc[-1] - results_df['Vid. tikslumas'].iloc[0]
    print(f"\nBendras pagerėjimas (E0 -> E5): {delta*100:+.2f} proc. punktų")
    print("="*70)

    plot_experiment_comparison(results_df)

    # Mokymosi eiga (geriausias modelis su val set)
    X_tr2, X_vl2, y_tr2, y_vl2 = train_test_split(
        X_train_s, y_train, test_size=0.15, stratify=y_train, random_state=42)
    m_vis = exp5_model()
    m_vis.fit(X_tr2, y_tr2, X_val=X_vl2, y_val=y_vl2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(m_vis.loss_history,     label='Apmokymo nuostoliai',  color='navy')
    ax.plot(m_vis.val_loss_history, label='Validavimo nuostoliai', color='crimson',
            linestyle='--')
    ax.set_title('Geriausio DNT mokymosi eiga', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epocha', fontsize=11)
    ax.set_ylabel('Cross-entropy nuostoliai', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\npabaiga")


if __name__ == '__main__':
    main()