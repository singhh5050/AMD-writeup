import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the benchmark results
df = pd.read_csv('../data/benchmark_results.csv')

# Set up the plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot execution times
ax1.plot(df['Matrix_Size'], df['CPU_Time_ms'], marker='o', label='CPU')
ax1.plot(df['Matrix_Size'], df['Naive_GPU_Time_ms'], marker='s', label='Naive GPU')
ax1.plot(df['Matrix_Size'], df['Tiled_GPU_Time_ms'], marker='^', label='Tiled GPU')
ax1.plot(df['Matrix_Size'], df['Optimized_GPU_Time_ms'], marker='*', label='Optimized GPU')

ax1.set_xlabel('Matrix Size')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Matrix Multiplication Performance')
ax1.set_xscale('log2')
ax1.set_yscale('log2')
ax1.grid(True)
ax1.legend()

# Plot speedups
ax2.plot(df['Matrix_Size'], df['Naive_Speedup'], marker='s', label='Naive GPU')
ax2.plot(df['Matrix_Size'], df['Tiled_Speedup'], marker='^', label='Tiled GPU')
ax2.plot(df['Matrix_Size'], df['Optimized_Speedup'], marker='*', label='Optimized GPU')

ax2.set_xlabel('Matrix Size')
ax2.set_ylabel('Speedup over CPU')
ax2.set_title('GPU Speedup Comparison')
ax2.set_xscale('log2')
ax2.grid(True)
ax2.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('../data/benchmark_plots.png', dpi=300, bbox_inches='tight')

# Create a summary table
summary = pd.DataFrame({
    'Implementation': ['CPU', 'Naive GPU', 'Tiled GPU', 'Optimized GPU'],
    'Best Time (ms)': [
        df['CPU_Time_ms'].min(),
        df['Naive_GPU_Time_ms'].min(),
        df['Tiled_GPU_Time_ms'].min(),
        df['Optimized_GPU_Time_ms'].min()
    ],
    'Max Speedup': [1.0, df['Naive_Speedup'].max(),
                    df['Tiled_Speedup'].max(),
                    df['Optimized_Speedup'].max()]
})

# Save summary to CSV
summary.to_csv('../data/performance_summary.csv', index=False)

print("Plots and summary saved to data directory") 