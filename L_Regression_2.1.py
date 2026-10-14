import pandas as pd, numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

d = pd.read_csv("2_1_dataset.csv", skiprows=1)
feats = ["log_value_diff","caps_diff","age_diff","attack_record_diff","height_diff","travel_km_diff","attack_share_diff","cohesion_diff"]
y = d["goal_difference"]; X = d[feats]; n, p = X.shape

# ---------- 1. collinearity (pairwise correlations between explanatory variables, then VIF) ----------
cm = X.corr()
corr_threshold, vif_threshold = 0.8, 5.0
largest_corr = cm.where(~np.eye(p, dtype=bool)).abs().max().max()
corr_flags = [
    (cm.index[i], cm.columns[j], cm.iloc[i, j])
    for i in range(p)
    for j in range(i + 1, p)
    if abs(cm.iloc[i, j]) >= corr_threshold
]
Xc = sm.add_constant(X)
vif_values = pd.Series(
    {feature: float(vif(Xc.values, i)) for i, feature in enumerate(feats, 1)},
    name="VIF",
)
vif_flags = vif_values[vif_values >= vif_threshold]

print("\n=== Collinearity screening ===")
print(f"Largest absolute pairwise correlation: {largest_corr:.3f}")
print(f"Pairwise correlation matrix (threshold |r| >= {corr_threshold:.1f}):")
print(cm.round(2).to_string())
fig, ax = plt.subplots(figsize=(9, 7))
image = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(p), labels=feats, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(p), labels=feats)
for i in range(p):
    for j in range(p):
        ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", color="black")
ax.set_title("Correlations Between Explanatory Variables")
fig.colorbar(image, ax=ax)
fig.tight_layout()
heatmap_path = "collinearity_correlation_heatmap.png"
fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
print(f"Correlation heatmap saved to: {heatmap_path}")
plt.show()

if corr_flags:
    print("Flagged variable pairs:")
    print(pd.DataFrame(corr_flags, columns=["variable_1", "variable_2", "correlation"]).to_string(index=False))
else:
    print(f"No variable pairs reached |r| >= {corr_threshold:.1f}.")

print(f"\nVariance inflation factors (threshold VIF >= {vif_threshold:.1f}):")
print(vif_values.round(2).to_frame().to_string())
if not vif_flags.empty:
    print("Flagged variables:")
    print(vif_flags.round(2).to_string())
else:
    print(f"No variables reached VIF >= {vif_threshold:.1f}.")

if not corr_flags and vif_flags.empty:
    print("Screening result: PASS - no concerning multicollinearity detected.")
else:
    print("Screening result: REVIEW - investigate the flagged correlations and/or VIF values.")

