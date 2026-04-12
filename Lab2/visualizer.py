import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def plot_my_tree(model, features, classes, ax=None, title="Sprendimų medis"):
    from sklearn.tree import plot_tree
    if ax is None:
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
    
    plot_tree(model, 
              feature_names=features, 
              class_names=classes, 
              filled=True, 
              rounded=True, 
              max_depth=2,
              fontsize=9,
              ax=ax)
    ax.set_title(title)