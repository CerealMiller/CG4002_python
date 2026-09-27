import glob
import os
from collections import Counter
import numpy as np
import pandas as pd

# Dataset parameters matching PyTorch notebook
FEATURE_COLS = [
    "ax_m/s/s",
    "ay_m/s/s",
    "az_m/s/s",
    "gx_deg/s",
    "gy_deg/s",
    "gz_deg/s",
    "mx_microT",
    "my_microT",
    "mz_microT",
]
LABEL_COL = "condition"
WINDOW_SIZE = 50
STRIDE = 25
PURITY_THRESHOLD = 0.8
NUM_CLASSES = 6


def create_windows_from_file(filepath):
  df = pd.read_csv(filepath)
  df = df.dropna().reset_index(drop=True)
  if LABEL_COL not in df.columns or len(df) < WINDOW_SIZE:
    return [], []

  df[LABEL_COL] = df[LABEL_COL].astype(int)
  X = df[FEATURE_COLS].values.astype(np.float32)
  y = df[LABEL_COL].values.astype(np.int64)

  windows, labels = [], []
  for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    x_win = X[start:end]
    y_win = y[start:end]

    counts = np.bincount(y_win, minlength=NUM_CLASSES)
    majority_label = counts.argmax()
    purity = counts[majority_label] / WINDOW_SIZE

    if purity >= PURITY_THRESHOLD:
      windows.append(x_win)
      labels.append(majority_label)

  return windows, labels


def main():
  # Priority 1: Check if a explicit test file list CSV exists
  if os.path.exists("selected_test_files.csv"):
    test_split_df = pd.read_csv("selected_test_files.csv")
    csv_files = test_split_df["test_file_path"].tolist()
    print(
        f"Found 'selected_test_files.csv'. Processing {len(csv_files)} test"
        " split files..."
    )
  elif os.path.exists("cg4002_selected_test_files.csv"):
    test_split_df = pd.read_csv("cg4002_selected_test_files.csv")
    csv_files = test_split_df["test_file_path"].tolist()
    print(
        f"Found 'cg4002_selected_test_files.csv'. Processing {len(csv_files)}"
        " test split files..."
    )
  else:
    # Priority 2: Process ALL CSV files in directory (excluding output/generated metadata CSVs)
    csv_files = sorted(glob.glob("*.csv"))
    ignored_files = {
        "selected_test_files.csv",
        "cg4002_selected_test_files.csv",
        "selected_test_dataset.csv",
        "cg4002_selected_test_dataset.csv",
    }
    csv_files = [f for f in csv_files if f not in ignored_files]
    print(f"Processing ALL {len(csv_files)} raw CSV files in directory...")

  all_windows, all_labels = [], []
  for f in csv_files:
    if not os.path.exists(f):
      print(f"Warning: File {f} not found, skipping.")
      continue
    wins, lbls = create_windows_from_file(f)
    all_windows.extend(wins)
    all_labels.extend(lbls)

  all_windows = np.array(all_windows, dtype=np.float32)
  all_labels = np.array(all_labels, dtype=np.int32)
  print(
      f"Extracted {len(all_windows)} window samples across {len(csv_files)}"
      " files."
  )

  # Export Flattened Inputs (450 floats per sample: time-interleaved t*9 + c)
  with open("test_inputs.txt", "w") as f_in:
    for sample in all_windows:
      # sample shape is (50, 9) -> flatten produces [t0_c0..t0_c8, t1_c0..t1_c8, ..., t49_c8]
      flat = sample.reshape(-1)
      f_in.write(" ".join(f"{val:.6f}" for val in flat) + "\n")

  # Export Ground Truth Labels
  with open("test_labels.txt", "w") as f_lbl:
    for label in all_labels:
      f_lbl.write(f"{label}\n")

  print(
      "Successfully exported all samples to 'test_inputs.txt' and"
      " 'test_labels.txt'!"
  )


if __name__ == "__main__":
  main()