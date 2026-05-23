# ============================================================
#  CodeAlpha Internship — Task 4: Sales Prediction
#  Dataset : Advertising.csv (downloaded from Kaggle)
#  Place Advertising.csv in the same folder as this script
# ============================================================

# ── 1. Imports ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 2. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("Advertising.csv")
df.columns = df.columns.str.strip()

# Drop unnamed index column if exists
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

print("✅ Dataset loaded!")
print("Shape   :", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nMissing values:", df.isnull().sum().sum())
print("\nBasic Statistics:")
print(df.describe())

# ── 3. EDA Visualizations ────────────────────────────────────

# 3a. Sales Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Sales'], bins=20, kde=True, color='steelblue')
plt.title("Sales Distribution", fontsize=14)
plt.xlabel("Sales (in thousands)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("sales_distribution.png", dpi=150)
plt.show()
print("📊 Saved: sales_distribution.png")

# 3b. Advertising Spend vs Sales (all 3 platforms)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
platforms = ['TV', 'Radio', 'Newspaper']
colors    = ['#4C72B0', '#55A868', '#C44E52']

for ax, platform, color in zip(axes, platforms, colors):
    ax.scatter(df[platform], df['Sales'], alpha=0.6,
               color=color, edgecolors='white', linewidth=0.3)
    # Trend line
    z = np.polyfit(df[platform], df['Sales'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[platform].min(), df[platform].max(), 100)
    ax.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--')
    ax.set_title(f"{platform} Ads vs Sales", fontsize=13)
    ax.set_xlabel(f"{platform} Budget ($000s)")
    ax.set_ylabel("Sales ($000s)")

plt.suptitle("Advertising Spend vs Sales by Platform", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("adspend_vs_sales.png", dpi=150)
plt.show()
print("📊 Saved: adspend_vs_sales.png")

# 3c. Correlation Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
print("📊 Saved: correlation_heatmap.png")

# 3d. Ad Budget Distribution per Platform
plt.figure(figsize=(8, 5))
df[['TV', 'Radio', 'Newspaper']].plot(kind='box', color=dict(
    boxes='steelblue', whiskers='steelblue',
    medians='red', caps='steelblue'))
plt.title("Ad Budget Distribution per Platform", fontsize=14)
plt.ylabel("Budget ($000s)")
plt.tight_layout()
plt.savefig("budget_distribution.png", dpi=150)
plt.show()
print("📊 Saved: budget_distribution.png")

# 3e. Total Ad Spend vs Sales
df['Total_Ad_Spend'] = df['TV'] + df['Radio'] + df['Newspaper']
plt.figure(figsize=(8, 5))
plt.scatter(df['Total_Ad_Spend'], df['Sales'], alpha=0.6,
            color='purple', edgecolors='white', linewidth=0.3)
z = np.polyfit(df['Total_Ad_Spend'], df['Sales'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['Total_Ad_Spend'].min(), df['Total_Ad_Spend'].max(), 100)
plt.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--')
plt.title("Total Ad Spend vs Sales", fontsize=14)
plt.xlabel("Total Ad Spend ($000s)")
plt.ylabel("Sales ($000s)")
plt.tight_layout()
plt.savefig("total_spend_vs_sales.png", dpi=150)
plt.show()
print("📊 Saved: total_spend_vs_sales.png")

# ── 4. Preprocessing ─────────────────────────────────────────
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain samples : {X_train.shape[0]}")
print(f"Test  samples : {X_test.shape[0]}")

# ── 5. Train 3 Models ────────────────────────────────────────
models = {
    "Linear Regression" : LinearRegression(),
    "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {"model": model, "y_pred": y_pred,
                     "MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2*100:.2f}%")

# ── 6. Actual vs Predicted ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, r) in zip(axes, results.items()):
    ax.scatter(y_test, r["y_pred"], alpha=0.6, color='steelblue',
               edgecolors='white', linewidth=0.3)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], 'r--', linewidth=1.5)
    ax.set_title(f"{name}\nR²: {r['R2']*100:.1f}%")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
plt.suptitle("Actual vs Predicted Sales", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()
print("📊 Saved: actual_vs_predicted.png")

# ── 7. Model Comparison ──────────────────────────────────────
names      = list(results.keys())
r2_scores  = [r["R2"]*100 for r in results.values()]
mae_scores = [r["MAE"]    for r in results.values()]
x = np.arange(len(names))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
b1 = ax1.bar(x, r2_scores, width=0.5,
             color=['#4C72B0','#55A868','#C44E52'], alpha=0.85)
ax1.set_ylabel("R² Score (%)")
ax1.set_title("R² Score Comparison")
ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=10, ha='right')
ax1.set_ylim(0, 110)
ax1.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)

b2 = ax2.bar(x, mae_scores, width=0.5,
             color=['#4C72B0','#55A868','#C44E52'], alpha=0.85)
ax2.set_ylabel("MAE")
ax2.set_title("MAE Comparison (Lower is Better)")
ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=10, ha='right')
ax2.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)

plt.suptitle("Model Comparison — Sales Prediction", fontsize=13)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("📊 Saved: model_comparison.png")

# ── 8. Feature Importance ────────────────────────────────────
rf_model   = results["Random Forest"]["model"]
importance = pd.Series(rf_model.feature_importances_,
                        index=['TV', 'Radio', 'Newspaper'])
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(7, 4))
colors = ['#4C72B0', '#55A868', '#C44E52']
bars = plt.bar(importance.index, importance.values, color=colors, alpha=0.85)
plt.title("Feature Importance — Which Platform Drives Sales?", fontsize=14)
plt.ylabel("Importance Score")
plt.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("📊 Saved: feature_importance.png")

# ── 9. Business Insights ─────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["R2"])
best      = results[best_name]

print("\n" + "="*50)
print("  📌 BUSINESS INSIGHTS")
print("="*50)
print(f"  Best Platform for Sales : {importance.index[0]}")
print(f"  TV correlation with Sales   : {df['TV'].corr(df['Sales']):.3f}")
print(f"  Radio correlation with Sales: {df['Radio'].corr(df['Sales']):.3f}")
print(f"  Newspaper correlation       : {df['Newspaper'].corr(df['Sales']):.3f}")
print(f"\n  🏆 Best Model : {best_name}")
print(f"  R²   : {best['R2']*100:.2f}%")
print(f"  MAE  : {best['MAE']:.2f}")
print("="*50)

print("\n✅ Task 4 complete! Saare charts save ho gaye hain.")
print("\n📁 GitHub par ye files upload karo (Task4 folder mein):")
print("   - task4_sales_prediction.py")
print("   - Advertising.csv")
print("   - sales_distribution.png")
print("   - adspend_vs_sales.png")
print("   - correlation_heatmap.png")
print("   - budget_distribution.png")
print("   - total_spend_vs_sales.png")
print("   - actual_vs_predicted.png")
print("   - model_comparison.png")
print("   - feature_importance.png")
