import matplotlib.pyplot as plt
import numpy as np

# Use a standard font and style
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-darkgrid')

# Data
datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4']

train_rmse = [0.0542, 0.0574, 0.0200, 0.0962]
test_rmse = [0.1036, 0.0794, 0.0891, 0.2203]
train_r2 = [0.9710, 0.9319, 0.9953, 0.9078]
test_r2 = [0.7953, 0.6189, 0.8889, 0.3709]
cv_rmse = [0.1315, 0.1634, 0.1580, 0.3251]
cv_rel_err = [59.78, 20.14, 58.98, 198.56]  # %

# Plot
x = np.arange(len(datasets))
width = 0.5

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()
bar_colors = ['#223A5E', '#125B50', '#5C5470', '#7D5A5A']

def add_labels(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v + max(values)*0.02, f'{v:.3f}' if v < 1 else f'{v:.2f}', 
                ha='center', va='bottom', fontsize=12, color='#222222')

# Train RMSE
axs[0].bar(x, train_rmse, width, color=bar_colors)
axs[0].set_title('Train RMSE', fontsize=16)
add_labels(axs[0], train_rmse)
axs[0].set_xticks(x)
axs[0].set_xticklabels(datasets, fontsize=13)

# Test RMSE
axs[1].bar(x, test_rmse, width, color=bar_colors)
axs[1].set_title('Test RMSE', fontsize=16)
add_labels(axs[1], test_rmse)
axs[1].set_xticks(x)
axs[1].set_xticklabels(datasets, fontsize=13)

# Train R²
axs[2].bar(x, train_r2, width, color=bar_colors)
axs[2].set_title('Train $R^2$', fontsize=16)
add_labels(axs[2], train_r2)
axs[2].set_xticks(x)
axs[2].set_xticklabels(datasets, fontsize=13)

# Test R²
axs[3].bar(x, test_r2, width, color=bar_colors)
axs[3].set_title('Test $R^2$', fontsize=16)
add_labels(axs[3], test_r2)
axs[3].set_xticks(x)
axs[3].set_xticklabels(datasets, fontsize=13)

# Cross-validation RMSE
axs[4].bar(x, cv_rmse, width, color=bar_colors)
axs[4].set_title('CV RMSE', fontsize=16)
add_labels(axs[4], cv_rmse)
axs[4].set_xticks(x)
axs[4].set_xticklabels(datasets, fontsize=13)

# Cross-validation Mean Relative Error
axs[5].bar(x, cv_rel_err, width, color=bar_colors)
axs[5].set_title('CV Mean Relative Error (%)', fontsize=16)
add_labels(axs[5], cv_rel_err)
axs[5].set_xticks(x)
axs[5].set_xticklabels(datasets, fontsize=13)

plt.suptitle('Comparison of Key Metrics Across Four Datasets', fontsize=20, y=1.03)
plt.tight_layout()
plt.show()
