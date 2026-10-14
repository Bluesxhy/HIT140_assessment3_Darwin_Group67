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

# ---------- 2. outliers in the response (1.5 x IQR rule) ----------
q1, q3 = y.quantile([.25, .75]); lo, hi = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
out = d[(y < lo) | (y > hi)]
print(f"\nResponse outlier fences: [{lo:.1f}, {hi:.1f}]  -> {len(out)} outlying matches:")
print(out[["fifa_match_no","team_A","team_B","goal_difference"]].to_string(index=False))

# ---------- 3. 80/20 train-test split, evaluation with the course metrics ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(Xtr, ytr); base = DummyRegressor().fit(Xtr, ytr)
adj = lambda r2, n_, p_: 1 - (1 - r2) * (n_ - 1) / (n_ - p_ - 1)
def metrics(model, A, b):
    pr = model.predict(A); mse = mean_squared_error(b, pr); rmse = np.sqrt(mse)
    return dict(MAE=mean_absolute_error(b, pr), MSE=mse, RMSE=rmse, NRMSE=rmse/(y.max()-y.min()), R2=r2_score(b, pr))
tr_m, te_m, ba_m = metrics(lr, Xtr, ytr), metrics(lr, Xte, yte), metrics(base, Xte, yte)
print(f"\nTrain n={len(Xtr)}, test n={len(Xte)}")
print(pd.DataFrame({"train": tr_m, "test": te_m, "baseline(test)": ba_m}).round(3).to_string())
print(f"Adjusted R2: train = {adj(tr_m['R2'], len(Xtr), p):.3f}")

# ---------- 4. full-data OLS: coefficients, significance, assumptions ----------
fit = sm.OLS(y, sm.add_constant(X)).fit(); print("\n", fit.summary())
print(f"\nR2 = {fit.rsquared:.4f} | Adj R2 = {fit.rsquared_adj:.4f} | F = {fit.fvalue:.3f} (p = {fit.f_pvalue:.2e})")
print("Shapiro p =", round(stats.shapiro(fit.resid).pvalue, 4), "| Breusch-Pagan p =", round(het_breuschpagan(fit.resid, sm.add_constant(X))[1], 4),
      "| Durbin-Watson =", round(sm.stats.durbin_watson(fit.resid), 3))
z = (X - X.mean()) / X.std(ddof=1); zf = sm.OLS((y - y.mean())/y.std(ddof=1), sm.add_constant(z)).fit()
print("\nStandardised coefficients:\n", pd.DataFrame({"beta": zf.params[feats], "p": zf.pvalues[feats]}).round(3).to_string())

# ---------- 5. stability: repeated 5-fold CV ----------
rm, ba, r2 = [], [], []
for s in range(20):
    cv = KFold(5, shuffle=True, random_state=s)
    pr = cross_val_predict(LinearRegression(), X, y, cv=cv); pb = cross_val_predict(DummyRegressor(), X, y, cv=cv)
    rm.append(np.sqrt(np.mean((y-pr)**2))); ba.append(np.sqrt(np.mean((y-pb)**2))); r2.append(r2_score(y, pr))
print(f"\nRepeated 5-fold CV: RMSE = {np.mean(rm):.3f} | baseline RMSE = {np.mean(ba):.3f} | CV R2 = {np.mean(r2):.3f}")

# ---------- 6. sensitivity: refit without the outlying matches ----------
keep = ~((y < lo) | (y > hi)); f2 = sm.OLS(y[keep], sm.add_constant(X[keep])).fit()
print(f"Without {int((~keep).sum())} outlying matches: n={int(keep.sum())}, R2={f2.rsquared:.3f}, adjR2={f2.rsquared_adj:.3f}")

# ---------- 7. symmetry check (swap Team A and Team B) ----------
fs = sm.OLS(-y, sm.add_constant(-X)).fit()
print("Swap check - max |slope difference| =", float(np.abs(fit.params[feats] - fs.params[feats]).max()))

# ---------- plots ----------
cvp = cross_val_predict(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=1))
plt.figure(figsize=(6,5)); plt.scatter(cvp, y, alpha=.7); plt.plot([-5,6.5],[-5,6.5],"r--")
plt.xlabel("Predicted goal difference (cross-validated)"); plt.ylabel("Actual goal difference"); plt.title("Actual vs Predicted"); plt.tight_layout(); plt.show()
plt.figure(figsize=(6,4.5)); plt.scatter(fit.fittedvalues, fit.resid, alpha=.7); plt.axhline(0, color="r", ls="--")
plt.xlabel("Fitted values"); plt.ylabel("Residuals"); plt.title("Residuals vs Fitted"); plt.tight_layout(); plt.show()
sm.qqplot(fit.resid, line="s"); plt.title("Normal Q-Q Plot of Residuals"); plt.tight_layout(); plt.show()
