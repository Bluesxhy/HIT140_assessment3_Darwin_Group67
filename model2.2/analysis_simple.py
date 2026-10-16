import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# First, have a look at the data and check if anything is missing or repeated.
data = pd.read_csv("outputs/model_2_2_dataset.csv")

print("First five rows:")
print(data.head())
print("\nDataset shape:", data.shape)
print("Number of matches:", data["match_id"].nunique())
print("\nDataset information:")
data.info()
print("\nMissing values:")
print(data.isna().sum())
print("\nDuplicate team-match records:")
print(data.duplicated(subset=["match_id", "team"]).sum())

# The first four columns identify the teams and matches, so leave those out.
# X has the eight variables, and y is the goals scored in the current match.
numbers = data.iloc[:, 4:13]
X = data.iloc[:, 4:12]
y = data["goals_scored"]

# Put each variable on its own row so the summary is easier to read.
summary = numbers.describe().T.round(2)
print("\nDescriptive statistics:")
print(summary.to_string())
summary.to_csv("outputs/descriptive_statistics.csv")

# Count the goal totals and put them in order before drawing the bars.
goals = y.value_counts().sort_index()
print("\nNumber of team records for each goal total:")
print(goals)

plt.figure()
plt.bar(goals.index, goals.values)
plt.xlabel("Goals scored")
plt.ylabel("Number of team records")
plt.title("Distribution of goals scored")
plt.xticks(goals.index)
plt.savefig("outputs/goals_distribution.png", dpi=300)
plt.show()

# These are the longest rest periods. Their previous match dates still need checking.
rest = data.sort_values("team_rest_days", ascending=False)
print("\nFive records with the longest rest periods:")
print(rest[
    ["match_id", "match_date", "team", "opponent", "team_rest_days"]
].head().to_string(index=False))

# Each dot is one team in one match. Transparency helps when dots overlap.
plt.figure()
plt.scatter(data["team_avg_goals_previous_5"], y, alpha=0.5)
plt.xlabel("Average goals in previous five matches")
plt.ylabel("Goals scored in current match")
plt.title("Previous scoring performance and current goals")
plt.savefig("outputs/previous_goals_scatter.png", dpi=300)
plt.show()

# Check the linear relationships with goals, leaving out goals correlated with itself.
corr = numbers.corr()
goal_corr = corr["goals_scored"].drop("goals_scored")
print("\nCorrelation of each X variable with goals scored:")
print(goal_corr.round(3))
corr.to_csv("outputs/correlation_table.csv")

# Also check whether some of the X variables are strongly related to each other.
x_corr = X.corr()
print("\nCorrelations between the X variables:")
print(x_corr.round(2).to_string())
x_corr.to_csv("outputs/x_correlation.csv")

# Add an intercept in a separate table, so X still means the original eight variables.
X_constant = sm.add_constant(X)
model = sm.OLS(y, X_constant)
results = model.fit()

# This is ordinary OLS. The two team records in each match may not be independent,
# so keep that limitation in mind when discussing the p-values.
print("\nMultiple linear regression results:")
print(results.summary())
with open("outputs/regression_results.txt", "w") as file:
    file.write(results.summary().as_text())

# VIF checks how much each X overlaps with the other X variables together.
# Start at 1 because column 0 is the intercept, which we are not reporting here.
vif_values = []
for i in range(1, len(X_constant.columns)):
    value = variance_inflation_factor(X_constant.values, i)
    vif_values.append(value)

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = vif_values
print("\nVIF results:")
print(vif.round(2).to_string(index=False))
vif.to_csv("outputs/vif_results.csv", index=False)

# These fitted values use the same data as the model, so they are not a test score.
fitted = results.fittedvalues
residuals = results.resid

# Look for curves or changes in the spread of the residuals around zero.
plt.figure()
plt.scatter(fitted, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted goals")
plt.ylabel("Residuals")
plt.title("Residuals vs fitted values")
plt.savefig("outputs/residuals_plot.png", dpi=300)
plt.show()

# If the residuals are roughly normal, the dots should follow the line fairly closely.
sm.qqplot(residuals, line="s")
plt.title("Normal Q-Q plot of residuals")
plt.savefig("outputs/residuals_qq_plot.png", dpi=300)
plt.show()

# Split matches rather than rows, so both teams from a match stay together.
match_ids = data["match_id"].unique()
train_ids, test_ids = train_test_split(
    match_ids, test_size=0.2, random_state=42
)
train = data[data["match_id"].isin(train_ids)]
test = data[data["match_id"].isin(test_ids)]

X_train = train.iloc[:, 4:12]
y_train = train["goals_scored"]
X_test = test.iloc[:, 4:12]
y_test = test["goals_scored"]

# This model only learns from the training set. It adds the intercept automatically.
test_model = LinearRegression()
test_model.fit(X_train, y_train)
predictions = test_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
test_r2 = r2_score(y_test, predictions)

print("\nTraining records:", len(train))
print("Test records:", len(test))
print("Test R-squared:", round(test_r2, 3))

# For a simple comparison, predict the training average for every test record.
# The test set should not be used to calculate this average.
average = y_train.mean()
baseline = [average] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline)
baseline_mse = mean_squared_error(y_test, baseline)
baseline_rmse = baseline_mse ** 0.5

# Both models use the same test records, so their errors can be compared here.
evaluation = pd.DataFrame({
    "Model": ["Linear regression", "Training mean baseline"],
    "MAE": [mae, baseline_mae],
    "RMSE": [rmse, baseline_rmse]
})
evaluation = evaluation.round(3)
print("\nModel evaluation:")
print(evaluation.to_string(index=False))
evaluation.to_csv("outputs/model_evaluation.csv", index=False)

# Save the full-data coefficients separately to help write the regression equation.
coefficients = results.params
coefficients.to_csv("outputs/regression_coefficients.csv", header=["Coefficient"])

# Get the coefficients from the model fitted on all 208 records.
# The constant is the intercept, followed by the eight X coefficients.
coefficients = results.params

print("\nFull-data regression coefficients:")
print(coefficients.round(4).to_string())

# Save the coefficients so we can use them when writing the equation.
coefficients.to_csv("outputs/regression_coefficients.csv", header=["Coefficient"])