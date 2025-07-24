import os
import re

def parse_float(value):
    try:
        return float(value)
    except:
        return 0.0

def extract_class_metrics(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    class_metrics = {}
    current_class = None

    for line in lines:
        line = line.strip()
        if line.startswith("Class:"):
            current_class = line.split(":")[1].strip()
            class_metrics[current_class] = {}
        elif any(metric in line for metric in ['precision', 'recall', 'support']):
            if current_class:
                key, val = line.split(":")
                class_metrics[current_class][key.strip()] = parse_float(val.strip())

    return class_metrics

def compute_classwise_accuracy(class_metrics):
    classwise_accuracy = {}
    for cls, metrics in class_metrics.items():
        recall = metrics.get('recall', 0.0)
        classwise_accuracy[cls] = recall
    return classwise_accuracy

def append_to_file(filepath, classwise_accuracy):
    with open(filepath, 'a') as f:
        f.write("\n=== Computed Class-wise Accuracy (based on recall) ===\n")
        for cls, acc in classwise_accuracy.items():
            f.write(f"{cls}: {acc:.4f}\n")

def process_all_results(folder_path):
    count = 0
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                full_path = os.path.join(root, filename)
                try:
                    class_metrics = extract_class_metrics(full_path)
                    classwise_acc = compute_classwise_accuracy(class_metrics)
                    append_to_file(full_path, classwise_acc)
                    print(f"✔ Processed: {full_path}")
                    count += 1
                except Exception as e:
                    print(f"⚠️ Skipped {full_path} due to error: {e}")
    print(f"\n✅ Done. Processed {count} result file(s).")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python batch_classwise_accuracy.py <results_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory.")
        sys.exit(1)

    process_all_results(folder)