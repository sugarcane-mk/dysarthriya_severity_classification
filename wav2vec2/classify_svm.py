import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import accuracy_score
import pandas as pd
import sys

# CONFIGURATION
DATA_DIR = sys.argv[1]
CLASS_NAMES = ['verylow', 'low', 'medium']
SAVE_DIR = sys.argv[2]
os.makedirs(SAVE_DIR, exist_ok=True)

# STEP 1: Load Data
data = defaultdict(lambda: defaultdict(list))  # class -> speaker -> list of embeddings

for class_label in CLASS_NAMES:
    class_path = os.path.join(DATA_DIR, class_label)
    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            emb_path = os.path.join(class_path, file)
            #speaker_id = file.split('_')[0][:3] # ssn_tdsc
            #speaker_id = file.split('_')[0] # ua speech
            speaker_id = file.split('_')[2][:3]  # extract the first three characters from the third field  wav_headMic_M01S01_0005_embedding.npy torgo
            embedding = np.load(emb_path)
            data[class_label][speaker_id].append(embedding)

# STEP 2: Prepare Speaker IDs
class_speakers = {cls: list(speakers.keys()) for cls, speakers in data.items()}

# STEP 3: Generate all combinations (1 speaker per class as test)
all_combinations = list(itertools.product(*[class_speakers[cls] for cls in CLASS_NAMES]))

# Encode labels to integers
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)

avg_conf_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
all_reports = []
fold_accuracies = []

# STEP 4: Cross-validation Loop
for i, combo in enumerate(all_combinations):
    X_train, y_train, X_test, y_test = [], [], [], []

    # Build train and test sets
    for cls in CLASS_NAMES:
        for spk in data[cls]:
            for emb in data[cls][spk]:
                if spk in combo and cls == CLASS_NAMES[combo.index(spk)]:
                    X_test.append(emb)
                    y_test.append(cls)
                else:
                    X_train.append(emb)
                    y_train.append(cls)

    # Train classifier
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    # Evaluation
    cm = confusion_matrix(y_test, preds, labels=CLASS_NAMES)
    avg_conf_matrix += cm

    report = classification_report(y_test, preds, labels=CLASS_NAMES, output_dict=True)
    all_reports.append(report)
    
    acc = accuracy_score(y_test, preds)
    fold_accuracies.append(acc)


    # Save individual fold confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.title(f"Confusion Matrix - Fold {i+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"confusion_matrix_fold_{i+1}.png"))
    plt.close()

# STEP 5: Average Results
avg_conf_matrix /= len(all_combinations)

# Average Classification Report
avg_report = defaultdict(lambda: defaultdict(float))
for rep in all_reports:
    for cls, metrics in rep.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                avg_report[cls][metric] += value
for cls in avg_report:
    for metric in avg_report[cls]:
        avg_report[cls][metric] /= len(all_reports)

# Display and Save Average Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(avg_conf_matrix, annot=True, fmt=".2f", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
plt.title("Average Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "average_confusion_matrix.png"))
#plt.show()

# Print Average Classification Report
print("\n=== Average Classification Report ===")
for cls in CLASS_NAMES:
    print(f"\nClass: {cls}")
    for metric, value in avg_report[cls].items():
        print(f"{metric}: {value:.4f}")

# STEP 6: Save and Display Accuracy
acc_df = pd.DataFrame({
    'Fold': list(range(1, len(fold_accuracies)+1)),
    'Accuracy': fold_accuracies
})
acc_df.to_csv(os.path.join(SAVE_DIR, 'fold_accuracies.csv'), index=False)

avg_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"\n=== Accuracy Summary ===")
print(f"Average Accuracy: {avg_acc:.4f}")
print(f"Standard Deviation: {std_acc:.4f}")

