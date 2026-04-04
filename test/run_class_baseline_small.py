import random
import numpy as np
import os
import json
import csv
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scripts.run_classification_baseline import load_image_paths, load_images

OUT = "outputs/analysis"
os.makedirs(OUT, exist_ok=True)

train_paths, train_labels = load_image_paths("data/processed/train")
test_paths_gt, test_labels_gt = load_image_paths("data/processed/test")
test_paths_recon, test_labels_recon = load_image_paths("outputs/inference_test")

random.seed(42)
train_sample = random.sample(
    list(zip(train_paths, train_labels)), min(400, len(train_paths))
)
train_paths_s, train_labels_s = zip(*train_sample)

test_sample_gt = random.sample(
    list(zip(test_paths_gt, test_labels_gt)), min(200, len(test_paths_gt))
)
test_paths_gt_s, test_labels_gt_s = zip(*test_sample_gt)

test_sample_recon = random.sample(
    list(zip(test_paths_recon, test_labels_recon)), min(200, len(test_paths_recon))
)
test_paths_recon_s, test_labels_recon_s = zip(*test_sample_recon)

le = LabelEncoder()
_ = le.fit_transform(train_labels_s)
X_train = load_images(list(train_paths_s), size=(64, 64))
X_test_gt = load_images(list(test_paths_gt_s), size=(64, 64))
X_test_recon = load_images(list(test_paths_recon_s), size=(64, 64))

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, le.transform(train_labels_s))

y_pred_gt = clf.predict(X_test_gt)
y_test_gt = le.transform(test_labels_gt_s)
acc_gt = accuracy_score(y_test_gt, y_pred_gt)
report_gt = classification_report(
    y_test_gt, y_pred_gt, target_names=le.classes_, output_dict=True
)
cm_gt = confusion_matrix(y_test_gt, y_pred_gt)

y_pred_recon = clf.predict(X_test_recon)
y_test_recon = le.transform(test_labels_recon_s)
acc_recon = accuracy_score(y_test_recon, y_pred_recon)
report_recon = classification_report(
    y_test_recon, y_pred_recon, target_names=le.classes_, output_dict=True
)
cm_recon = confusion_matrix(y_test_recon, y_pred_recon)

with open(os.path.join(OUT, "report_gt_baseline_small.json"), "w") as f:
    json.dump(report_gt, f, indent=2)
with open(os.path.join(OUT, "report_recon_baseline_small.json"), "w") as f:
    json.dump(report_recon, f, indent=2)

print("GT acc", acc_gt)
print("Recon acc", acc_recon)
