"""
name = 
EMAN AHMED             ID=0663/16
MEKLIT MULUGETA        ID=1286/16
SALSAWIT RISTU         ID=1655/16
"""
"""
PUMP CONDITION CLASSIFICATION
Supervised Machine Learning Classification

Algorithms:
1. Decision Tree
2. Random Forest
3. Support Vector Machine (SVM)
4. Gaussian Naive Bayes
5. AdaBoost

Classes:
- Normal
- Warning
- Fault

Dataset:
50 manually defined observations
"""

# --------------------------------------------------
# IMPORT LIBRARIES
# --------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. DATASET (50 OBSERVATIONS INCLUDED IN THE CODE)
# --------------------------------------------------

print("=" * 65)
print("PUMP CONDITION CLASSIFICATION")
print("=" * 65)

data = [
    # NORMAL CONDITION (20 observations)
    [2.1, 1490, 54, 5.1, 1200, "Normal"],
    [1.8, 1520, 52, 5.3, 1500, "Normal"],
    [2.3, 1480, 56, 4.9, 1800, "Normal"],
    [1.9, 1510, 53, 5.2, 900, "Normal"],
    [2.0, 1500, 55, 5.0, 2000, "Normal"],
    [2.2, 1470, 57, 4.8, 2200, "Normal"],
    [1.7, 1530, 51, 5.4, 1100, "Normal"],
    [2.4, 1460, 58, 4.7, 2500, "Normal"],
    [1.9, 1505, 54, 5.1, 1700, "Normal"],
    [2.0, 1495, 55, 5.0, 1300, "Normal"],
    [2.1, 1515, 53, 5.2, 1600, "Normal"],
    [1.8, 1485, 52, 5.3, 800, "Normal"],
    [2.3, 1500, 56, 4.9, 1900, "Normal"],
    [1.9, 1525, 54, 5.1, 1400, "Normal"],
    [2.2, 1475, 57, 4.8, 2100, "Normal"],
    [1.7, 1510, 51, 5.4, 1000, "Normal"],
    [2.4, 1490, 58, 4.7, 2300, "Normal"],
    [2.0, 1505, 55, 5.0, 1750, "Normal"],
    [1.8, 1535, 52, 5.3, 950, "Normal"],
    [2.1, 1480, 54, 5.1, 2800, "Normal"],

    # WARNING CONDITION (15 observations)
    [4.1, 1350, 72, 4.0, 3500, "Warning"],
    [3.8, 1380, 70, 4.2, 3200, "Warning"],
    [4.3, 1320, 75, 3.9, 4000, "Warning"],
    [3.9, 1370, 71, 4.1, 3000, "Warning"],
    [4.5, 1300, 76, 3.8, 4500, "Warning"],
    [4.0, 1360, 73, 4.0, 3800, "Warning"],
    [3.7, 1390, 69, 4.3, 2800, "Warning"],
    [4.2, 1340, 74, 3.9, 4200, "Warning"],
    [4.4, 1310, 77, 3.7, 4800, "Warning"],
    [3.9, 1375, 71, 4.1, 3300, "Warning"],
    [4.1, 1330, 73, 4.0, 3600, "Warning"],
    [3.8, 1385, 70, 4.2, 3100, "Warning"],
    [4.3, 1325, 75, 3.8, 4400, "Warning"],
    [4.0, 1355, 72, 4.1, 3900, "Warning"],
    [4.5, 1290, 78, 3.7, 5000, "Warning"],

    # FAULT CONDITION (15 observations)
    [7.1, 1100, 91, 2.7, 6500, "Fault"],
    [6.8, 1150, 88, 3.0, 7000, "Fault"],
    [7.5, 1050, 95, 2.5, 8000, "Fault"],
    [6.9, 1120, 90, 2.8, 6200, "Fault"],
    [7.3, 1080, 93, 2.6, 8500, "Fault"],
    [7.8, 1020, 98, 2.3, 9000, "Fault"],
    [6.7, 1160, 87, 3.1, 5800, "Fault"],
    [7.2, 1090, 92, 2.7, 7500, "Fault"],
    [7.6, 1040, 96, 2.4, 9500, "Fault"],
    [6.9, 1130, 89, 2.9, 6800, "Fault"],
    [7.4, 1070, 94, 2.6, 8200, "Fault"],
    [6.8, 1140, 88, 3.0, 6000, "Fault"],
    [7.7, 1030, 97, 2.3, 9800, "Fault"],
    [7.0, 1110, 90, 2.8, 7200, "Fault"],
    [7.5, 1060, 95, 2.5, 8800, "Fault"]
]

# Convert the data into a Pandas DataFrame.
df = pd.DataFrame(
    data,
    columns=[
        "Vibration",
        "RPM",
        "Temperature",
        "Pressure",
        "Operating_Hours",
        "Condition"
    ]
)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nComplete Dataset:")
print(df.to_string(index=False))

print("\nClass distribution:")
print(df["Condition"].value_counts())


# --------------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# --------------------------------------------------

# Input features.
X = df[
    [
        "Vibration",
        "RPM",
        "Temperature",
        "Pressure",
        "Operating_Hours"
    ]
]

# Target variable.
y = df["Condition"]

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget column:")
print("Condition")


# --------------------------------------------------
# 3. SPLIT DATASET
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))


# --------------------------------------------------
# 4. DEFINE CLASSIFICATION MODELS
# --------------------------------------------------

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            random_state=42
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    )
}


# --------------------------------------------------
# 5. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = []
confusion_matrices = {}

valid_classes = ["Normal", "Warning", "Fault"]

for model_name, model in models.items():

    print("\n" + "-" * 65)
    print("MODEL:", model_name)
    print("-" * 65)

    # Train the model.
    model.fit(X_train, y_train)

    # Predict test observations.
    y_pred = model.predict(X_test)

    # Calculate accuracy.
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate Fault precision.
    fault_precision = precision_score(
        y_test,
        y_pred,
        labels=["Fault"],
        average="macro",
        zero_division=0
    )

    # Calculate Fault recall.
    fault_recall = recall_score(
        y_test,
        y_pred,
        labels=["Fault"],
        average="macro",
        zero_division=0
    )

    # Calculate macro F1 score.
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Generate confusion matrix.
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=valid_classes
    )

    confusion_matrices[model_name] = cm

    # Display evaluation results.
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Fault Precision: {fault_precision:.4f}")
    print(f"Fault Recall: {fault_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=valid_classes,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(cm)

    # Save results for comparison.
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Fault Precision": fault_precision,
        "Fault Recall": fault_recall,
        "Macro F1": macro_f1
    })


# --------------------------------------------------
# 6. COMPARE ALL MODELS
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.sort_values(
        by="Macro F1",
        ascending=False
    ).to_string(index=False)
)

best_model = results_df.loc[
    results_df["Macro F1"].idxmax()
]

print("\nBest model based on Macro F1:")
print(best_model["Model"])

print("\nBest model Fault Recall:")
print(f"{best_model['Fault Recall']:.4f}")


# --------------------------------------------------
# 7. VISUALIZE CONFUSION MATRICES
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

axes = axes.flatten()

for index, (model_name, cm) in enumerate(
    confusion_matrices.items()
):

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=valid_classes,
        yticklabels=valid_classes,
        ax=axes[index]
    )

    axes[index].set_title(model_name)
    axes[index].set_xlabel("Predicted Condition")
    axes[index].set_ylabel("Actual Condition")

# Hide the unused sixth subplot.
axes[5].axis("off")

plt.suptitle(
    "Pump Condition Classification - Confusion Matrices",
    fontsize=16
)

plt.tight_layout()

# Save the visualization in the current working directory.
plt.savefig(
    "confusion_matrices.png",
    dpi=300
)

print("\nConfusion matrices saved as confusion_matrices.png")

plt.show()


# --------------------------------------------------
# 8. FINAL CONCLUSION
# --------------------------------------------------

print("\n" + "=" * 65)
print("CLASSIFICATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    "\nImportant: Model predictions support engineering "
    "inspection and safety procedures. They do not replace them."
)

print(
    "\nNote: This project uses synthetic practice data, "
    "not measurements from real industrial pumps."
)
