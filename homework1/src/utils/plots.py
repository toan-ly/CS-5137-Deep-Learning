import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_density(df, num_cols=4):
    """
    Plot the density distribution of each numerical feature in the DataFrame.
    """
    num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_rows = (len(num_features) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_features):
        # sns.kdeplot(df[col], ax=axes[i], color='red')
        sns.histplot(df[col], ax=axes[i], kde=True, stat='density', bins=30, alpha=0.3)
        axes[i].set_title(f'{col}')

    plt.tight_layout()
    plt.show()

def plot_boxwhisker(df, num_cols=4):
    """
    Plot the box and whisker plot of each numerical feature in the DataFrame.
    """
    num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_rows = (len(num_features) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_features):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'{col}')

    plt.tight_layout()
    plt.show()