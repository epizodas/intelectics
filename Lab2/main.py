from data_loader import load_dataset, prepare_data_for_tree
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
from visualizer import plot_my_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def experiment_forest_size(X_train, X_test, y_train, y_test, depth):
    tree_counts = [3, 4, 5, 6, 7, 8, 9]
    rf_results = []

    for count in tree_counts:
        model = RandomForestClassifier(n_estimators=count, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        rf_results.append({
            'Medžių kiekis': count,
            'Tikslumas': acc
        })
    
    return pd.DataFrame(rf_results)

def experiment_depths(X_train, X_test, y_train, y_test):
    depths = [2, 4, 6, 8, 10, 15, 20]
    results = []

    for depth in depths:
        start_time = time.time()
        
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Testuojame
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Maks. gylis': depth,
            'Tikslumas': acc,
            'Laikas (s)': duration
        })

    return pd.DataFrame(results)

def run_initial_model(X, X_train, X_test, y_train, y_test, final_comparison):
    clf = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=None)
    clf.fit(X_train, y_train)

    plot_my_tree(clf, X.columns, clf.classes_, ax=None, title="Sprendimų medis")

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    final_comparison.append({'Modelis': 'Pirminis sprendimų medis', 'Accuracy': accuracy, 'Precision': prec, 'Recall': rec})
    
    print(f"\nModelio tikslumas (Accuracy): {accuracy:.4f}")
    print("\nIšsami klasifikavimo ataskaita:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    return clf

def run_depth_experiment(X_train, X_test, y_train, y_test, final_comparison):
    depth_results = experiment_depths(X_train, X_test, y_train, y_test)
    print("\n8 punkto rezultatai (Eksperimentas su gyliu):")
    print(depth_results)

    best_depth_row = depth_results.loc[depth_results['Tikslumas'].idxmax()]
    best_depth = int(best_depth_row['Maks. gylis'].item())
    best_acc = best_depth_row['Tikslumas']
    print(f"\nGeriausias nustatytas gylis: {best_depth}")

    # Retrain with best depth to get full metrics
    best_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    best_clf.fit(X_train, y_train)
    y_pred_best = best_clf.predict(X_test)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred_best, average='macro', zero_division=0)
    final_comparison.append({'Modelis': f'Geriausias sprendimų medis (gylis={best_depth})', 'Accuracy': best_acc, 'Precision': prec, 'Recall': rec})

    return best_depth

def run_forest_experiment(X, X_train, X_test, y_train, y_test, best_depth, final_comparison):
    # 9 punktas
    rf_5 = RandomForestClassifier(n_estimators=5, max_depth=best_depth, random_state=42)
    rf_5.fit(X_train, y_train)
    y_pred_rf5 = rf_5.predict(X_test)
    acc_rf5 = accuracy_score(y_test, y_pred_rf5)

    print(f"\n9 punktas: Atsitiktinio miško (5 medžiai, gylis {best_depth}) tikslumas: {acc_rf5:.4f}")

    print("Generuojami 5 medžių miško grafikai...")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
    axes_flat = axes.flatten()
    for index in range(0, 5):
        plot_my_tree(rf_5.estimators_[index], X.columns, rf_5.classes_, ax=axes_flat[index], title=f"Medis Nr. {index+1}")
    axes_flat[5].axis('off')
    plt.tight_layout()
    plt.show()

    print("Generuojama 5 medžių miško Confusion Matrix...")
    cm_rf5 = confusion_matrix(y_test, y_pred_rf5, labels=rf_5.classes_)
    disp_rf5 = ConfusionMatrixDisplay(confusion_matrix=cm_rf5, display_labels=rf_5.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp_rf5.plot(ax=ax, cmap='YlGnBu', xticks_rotation=45)
    plt.title(f"Confusion Matrix: Random Forest (n=5, gylis={best_depth})")
    plt.show()

    # 10 punktas
    tree_variations = [5, 7, 9, 10, 11, 13]
    forest_results = []

    print("\n10 punktas: Eksperimentuojama su miško dydžiu...")
    for count in tree_variations:
        model = RandomForestClassifier(n_estimators=count, max_depth=best_depth, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        forest_results.append({'Medžių kiekis': count, 'Tikslumas': acc})

    df_forest_results = pd.DataFrame(forest_results)
    print(df_forest_results)

    best_n_row = df_forest_results.loc[df_forest_results['Tikslumas'].idxmax()]
    best_n = int(best_n_row['Medžių kiekis'].item())
    best_rf_acc = best_n_row['Tikslumas']

    best_rf_model = RandomForestClassifier(n_estimators=best_n, max_depth=best_depth, random_state=42)
    best_rf_model.fit(X_train, y_train)
    y_pred_best_rf = best_rf_model.predict(X_test)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred_best_rf, average='macro', zero_division=0)
    final_comparison.append({'Modelis': f'Geriausias atsitiktinis miškas (n={best_n})', 'Accuracy': best_rf_acc, 'Precision': prec, 'Recall': rec})

    print(f"Generuojama geriausio miško (n={best_n}) Confusion Matrix...")
    cm_best_rf = confusion_matrix(y_test, y_pred_best_rf, labels=best_rf_model.classes_)
    disp_best_rf = ConfusionMatrixDisplay(confusion_matrix=cm_best_rf, display_labels=best_rf_model.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp_best_rf.plot(ax=ax, cmap='Greens', xticks_rotation=45)
    plt.title(f"Confusion Matrix: Geriausias miškas (n={best_n}, gylis={best_depth})")
    plt.show()

def print_final_comparison(final_comparison):
    print("\n" + "="*60)
    print(f"{'MODELIŲ PALYGINIMAS':^60}")
    print("="*60)
    comparison_df = pd.DataFrame(final_comparison)
    print(comparison_df.to_string(index=False, formatters={'Accuracy': '{:,.4f}'.format, 'Precision': '{:,.4f}'.format, 'Recall': '{:,.4f}'.format}))
    print("="*60)

def main():
    file_path = '../data/csgo.csv'
    print("Starting")

    df = load_dataset(file_path)
    
    if df is not None:
        final_comparison = []
        X, y = prepare_data_for_tree(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Viso duomenų: {len(X)}")
        print(f"Apmokymo imtis: {len(X_train)}")
        print(f"Testavimo imtis: {len(X_test)}")
        print("\nPožymiai (X) naudojami modelyje:", X.columns.tolist())

        run_initial_model(X, X_train, X_test, y_train, y_test, final_comparison)
        best_depth = run_depth_experiment(X_train, X_test, y_train, y_test, final_comparison)
        run_forest_experiment(X, X_train, X_test, y_train, y_test, best_depth, final_comparison)
        print_final_comparison(final_comparison)
    else:
        print("error")

if __name__ == "__main__":
    main()