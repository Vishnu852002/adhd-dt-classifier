"""
zigzag_predict_dt.py

Performs Decision Tree Classification on the ZigZag Predict dataset
using Sequential Forward Floating Selection (SFFS) and
Leave-One-Out Cross Validation (LOOCV).
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load dataset
data_path = '/content/drive/MyDrive/ADHD_MainProject/Zigzag_predict.csv'
df = pd.read_csv(data_path)
df = df.set_index('Class')
data = df.to_numpy()

X = data[:, 2:]
y = data[:, 1].astype('int')

# Hyperparameter grid for Decision Tree
param_grid = {
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_leaf': list(range(1, 11)),
    'min_samples_split': [2, 3, 4, 5]
}

# Setup LOOCV and scaler
cv = LeaveOneOut()
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Output dataframe
results_df = pd.DataFrame()
best_acc = 0

# Perform SFFS + GridSearchCV over increasing feature sizes
for i in range(2, 31):
    print(f"Running SFFS with {i} features...")

    sffs = SFS(
        DecisionTreeClassifier(random_state=0),
        k_features=i,
        forward=True,
        floating=True,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    sffs.fit(X_std, y)
    selected_features = list(sffs.k_feature_idx_)
    X_sffs = X_std[:, selected_features]
    X_sffs_std = scaler.fit_transform(X_sffs)

    clf = GridSearchCV(
        DecisionTreeClassifier(random_state=0),
        param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    clf.fit(X_sffs_std, y)

    score = clf.best_score_
    print(f"Accuracy: {score}, Params: {clf.best_params_}")

    if score < best_acc - 0.05:
        break
    if score > best_acc:
        best_acc = score

    result_row = pd.DataFrame([[selected_features, score, clf.best_params_]],
                              columns=['Features', 'Accuracy', 'Best_Params'])
    results_df = pd.concat([results_df, result_row])

# Save results
output_path = '/content/drive/MyDrive/ADHD_MainProject/result/DT_Task2_SFFS.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")