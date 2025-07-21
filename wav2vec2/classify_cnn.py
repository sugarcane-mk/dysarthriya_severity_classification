import os
import numpy as np
import itertools
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Configuration
# ----------------------------
DATA_DIR = sys.argv[1]  # your embeddings directory
SAVE_DIR = sys.argv[2]
os.makedirs(SAVE_DIR, exist_ok=True)

############# change class folder name accordingly

CLASS_NAMES = ['verylow', 'low', 'medium']

batch_size = 64
num_epochs = 100
patience = 20
learning_rate = 0.001

# ----------------------------
# STEP 1: Load Data
# ----------------------------
data = defaultdict(lambda: defaultdict(list))

for class_label in CLASS_NAMES:
    class_path = os.path.join(DATA_DIR, class_label)
    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            emb_path = os.path.join(class_path, file)
            #speaker_id = file.split('_')[0][:3] # ssn_tdsc spk_id 
            #speaker_id = file.split('_')[0] # ua speech
            speaker_id = file.split('_')[2][:3]  # e.g., M01
            embedding = np.load(emb_path)
            data[class_label][speaker_id].append(embedding)

# ----------------------------
# STEP 2: Speaker IDs
# ----------------------------
class_speakers = {cls: list(speakers.keys()) for cls, speakers in data.items()}

# ----------------------------
# STEP 3: Combinations
# ----------------------------
all_combinations = list(itertools.product(*[class_speakers[cls] for cls in CLASS_NAMES]))

label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)

# ----------------------------
# Dataset Class
# ----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = self.embeddings[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ----------------------------
# CNN Model
# ----------------------------
class CNNClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        flatten_dim = 64 * input_shape[0] * input_shape[1]
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dim
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------
# Train Function
# ----------------------------
def train_model(model, dataloader, criterion, optimizer, num_epochs=100, patience=20):
    best_acc = 0
    best_model_state = None
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        all_preds, all_labels = [], []

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_acc = accuracy_score(all_labels, all_preds)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

# ----------------------------
# Cross-Validation Loop
# ----------------------------
avg_conf_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
all_reports = []
fold_accuracies = []

for fold_idx, combination in enumerate(all_combinations):
    test_speakers = {cls: spk for cls, spk in zip(CLASS_NAMES, combination)}
    
    train_embeddings, train_labels = [], []
    test_embeddings, test_labels = [], []

    for cls in CLASS_NAMES:
        for speaker, emb_list in data[cls].items():
            for emb in emb_list:
                # wav2vec2 embeddings are 1D, reshape to (H, W)
                if len(emb.shape) == 1:
                    side = int(np.ceil(np.sqrt(len(emb))))
                    padded = np.zeros((side * side,))
                    padded[:len(emb)] = emb
                    emb_reshaped = padded.reshape(side, side)
                else:
                    emb_reshaped = emb

                if speaker == test_speakers[cls]:
                    test_embeddings.append(emb_reshaped)
                    test_labels.append(label_encoder.transform([cls])[0])
                else:
                    train_embeddings.append(emb_reshaped)
                    train_labels.append(label_encoder.transform([cls])[0])

    if len(train_embeddings) == 0 or len(test_embeddings) == 0:
        continue

    input_shape = train_embeddings[0].shape
    model = CNNClassifier(input_shape, num_classes=len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = train_model(model, train_loader, criterion, optimizer, num_epochs, patience)

    # Evaluate
    model.eval()
    preds = []
    for emb in test_embeddings:
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        output = model(emb_tensor)
        pred = output.argmax(dim=1).item()
        preds.append(pred)

    acc = accuracy_score(test_labels, preds)
    cm = confusion_matrix(test_labels, preds, labels=list(range(len(CLASS_NAMES))))
    avg_conf_matrix += cm

    report = classification_report(test_labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    all_reports.append(report)
    fold_accuracies.append(acc)

    # Save individual confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.title(f"Confusion Matrix - Fold {fold_idx + 1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"confusion_matrix_fold_{fold_idx + 1}.png"))
    plt.close()

# ----------------------------
# Average Results
# ----------------------------
avg_conf_matrix /= len(all_combinations)

# Average classification report
avg_report = defaultdict(lambda: defaultdict(float))
for rep in all_reports:
    for cls, metrics in rep.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                avg_report[cls][metric] += value

for cls in avg_report:
    for metric in avg_report[cls]:
        avg_report[cls][metric] /= len(all_reports)

# Save average confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(avg_conf_matrix, annot=True, fmt=".2f", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
plt.title("Average Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "average_confusion_matrix.png"))
plt.close()

# Print average classification report
print("\n=== Average Classification Report ===")
for cls in CLASS_NAMES:
    print(f"\nClass: {cls}")
    for metric, value in avg_report[cls].items():
        print(f"{metric}: {value:.4f}")

# Save accuracy summary
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

