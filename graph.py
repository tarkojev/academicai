import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup and Data Loading
FILE_PATH = 'runs/comparison.csv'
OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use a clean style for professional-looking graphs
sns.set_theme(style="whitegrid")

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: {FILE_PATH} not found.")
    exit()

# 2. Data Preprocessing
# Ensure numeric columns are actually numeric
numeric_cols = ['accuracy', 'f1_macro', 'latency_ms', 'size_mb', 'n_per_class']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle n_per_class for plotting: fill NaN (full runs) with a label
df['data_regime'] = df['n_per_class'].fillna('Full-Set').astype(str)

# 3. Plotting Functions

def plot_accuracy_rank(df):
    """Generates a ranked bar chart of all runs."""
    plt.figure(figsize=(10, 12))
    sorted_df = df.sort_values('accuracy', ascending=False)
    sns.barplot(data=sorted_df, x='accuracy', y='run_name', hue='type', dodge=False)
    plt.title('Performance Ranking: Accuracy by Run')
    plt.xlabel('Accuracy (0.0 - 1.0)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_accuracy_ranking.png')
    print("Generated: Accuracy Ranking")

def plot_few_shot_scaling(df):
    """Plots the learning curve for few-shot experiments."""
    fs_df = df.dropna(subset=['n_per_class']).copy()
    if fs_df.empty: return
    
    plt.figure(figsize=(10, 6))
    # Aggregate multiple seeds into a line with confidence intervals
    sns.lineplot(data=fs_df, x='n_per_class', y='accuracy', hue='model', 
                 style='type', markers=True, markersize=10)
    
    plt.xscale('log')
    plt.title('Scaling Analysis: Accuracy vs. Samples Per Class')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Accuracy')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_few_shot_scaling.png')
    print("Generated: Few-Shot Scaling")

def plot_efficiency_tradeoff(df):
    """Plots Latency vs Accuracy to find the 'Sweet Spot'."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='latency_ms', y='accuracy', 
                    hue='model', style='type', s=150, alpha=0.8)
    
    plt.title('Efficiency Trade-off: Accuracy vs. Latency')
    plt.xlabel('Inference Latency (ms)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_latency_tradeoff.png')
    print("Generated: Latency Trade-off")

def plot_type_comparison(df):
    """Specifically compares Student vs Teacher vs Ensemble performance."""
    plt.figure(figsize=(10, 6))
    # Filter for the 1000-shot regime to see the most stable comparison
    fs_1000 = df[df['n_per_class'] == 1000]
    
    sns.boxplot(data=fs_1000, x='type', y='accuracy', palette="Set2")
    plt.title('Performance Distribution at N=1000 (Student vs Teacher vs Ensemble)')
    plt.savefig('plots/04_type_comparison.png')

# 4. Execute
if __name__ == "__main__":
    plot_accuracy_rank(df)
    plot_few_shot_scaling(df)
    plot_efficiency_tradeoff(df)
    plot_type_comparison(df)
    print(f"\nAll plots saved to the '{OUTPUT_DIR}' folder.")