
#  DIABETES PREDICTION MODEL - v2

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import shap
import pickle
import warnings
warnings.filterwarnings("ignore")

#Load Dataset
df = pd.read_csv("diabetes.csv")
print("Dataset loaded:", df.shape)

#Clean Data
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.fillna(df.median(), inplace=True)
print("Missing values handled")

#Morphological Features
df['cvamp']       = df['Glucose'] / df['Glucose'].mean()
df['stdamp']      = df['BloodPressure'] / df['BloodPressure'].mean()
df['PowerHF']     = (df['BMI'] * df['Insulin']) / 1000
df['age_glucose'] = df['Age'] * df['Glucose'] / 1000
df['glucose_bmi'] = df['Glucose'] * df['BMI'] / 1000
df['insulin_bmi'] = df['Insulin'] / (df['BMI'] + 1)
print("Morphological features added")

#Features & Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']
feature_names = X.columns.tolist()

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Scale
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print(f"SMOTE applied — balanced size: {X_train_bal.shape}")

#Compare Models
print("\nComparing models:")
scale_pos = (y_train_bal == 0).sum() / (y_train_bal == 1).sum()

models = {
    'XGBoost': XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=3,
        random_state=42,
        class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    ),
}

cv         = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
best_score = 0
best_name  = ""
best_model = None

for name, m in models.items():
    scores = cross_val_score(m, X_train_bal, y_train_bal, cv=cv, scoring='accuracy')
    avg    = scores.mean()
    print(f"  {name:22s} → CV Accuracy: {avg:.2%}  (±{scores.std():.2%})")
    if avg > best_score:
        best_score = avg
        best_name  = name
        best_model = m


print(f"\nBest Model: {best_name} ({best_score:.2%})")

#Train Best Model
print(f"\nTraining {best_name}")
best_model.fit(X_train_bal, y_train_bal)

y_pred   = best_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#SHAP Analysis
print("\n Running SHAP analysis")
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

if isinstance(shap_values, list):
    shap_array = np.abs(shap_values[1]).mean(axis=0)
else:
    shap_array = np.abs(shap_values).mean(axis=0)
    if shap_array.ndim > 1:
        shap_array = shap_array[:, 1]

if len(shap_array) != len(feature_names):
    shap_array = shap_array[:len(feature_names)]

shap_importance = pd.DataFrame({
    'Feature':         feature_names,
    'SHAP_Importance': shap_array
}).sort_values('SHAP_Importance', ascending=False)

print("\nFeature Importance (SHAP):")
print(shap_importance.to_string(index=False))

#Select Important Features
threshold          = shap_importance['SHAP_Importance'].mean()
important_features = shap_importance[
    shap_importance['SHAP_Importance'] >= threshold
]['Feature'].tolist()
print(f"\nSelected features: {important_features}")

#Retrain on Important Features
X_train_df    = pd.DataFrame(X_train_bal,   columns=feature_names)
X_test_df     = pd.DataFrame(X_test_scaled, columns=feature_names)
X_train_final = X_train_df[important_features]
X_test_final  = X_test_df[important_features]

best_model.fit(X_train_final, y_train_bal)
y_pred_final = best_model.predict(X_test_final)
final_acc    = accuracy_score(y_test, y_pred_final)
print(f"\nFinal Accuracy (after SHAP selection): {final_acc:.2%}")

#Save Everything
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("features.pkl", "wb") as f:
    pickle.dump({
        "all_features":       feature_names,
        "important_features": important_features,
        "best_model_name":    best_name
    }, f)

print("\nmodel.pkl saved")
print("scaler.pkl saved")
print("features.pkl saved")
print(f"\nBest: {best_name} | Final Accuracy: {final_acc:.2%}")
