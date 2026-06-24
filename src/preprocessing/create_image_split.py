import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ==========================================================
# IMPORT CONFIG
# ==========================================================
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

import config


RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE_IN_TRAINVAL = 0.20


def label_counts(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name} samples: {len(df)}")
    print(df["final_label"].value_counts())
    print(f"Unique videos: {df['video_id'].nunique()}")


def check_group_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_video_ids = set(train_df["video_id"].astype(str))
    val_video_ids = set(val_df["video_id"].astype(str))
    test_video_ids = set(test_df["video_id"].astype(str))

    train_val_overlap = train_video_ids & val_video_ids
    train_test_overlap = train_video_ids & test_video_ids
    val_test_overlap = val_video_ids & test_video_ids

    print("\n===== LEAKAGE CHECK BY video_id =====")
    print(f"train ∩ val : {len(train_val_overlap)}")
    print(f"train ∩ test: {len(train_test_overlap)}")
    print(f"val ∩ test  : {len(val_test_overlap)}")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError("Leakage detected: the same video_id appears in more than one split.")


def main() -> None:
    processed_dir = Path(config.CNN_PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(config.MASTER_DATASET_DIR) / "metadata_master.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found: {csv_path}")

    print("Loading metadata...")
    df = pd.read_csv(csv_path)

    required_cols = {"status", "final_label", "image_path", "video_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")

    df = df[
        (df["status"] == "success")
        & (df["final_label"].isin(["open", "closed"]))
        & (df["video_id"].notna())
    ].copy()

    # Keep only existing images to guarantee all downstream pipelines use the same rows.
    df = df[df["image_path"].apply(lambda p: Path(str(p)).exists())].copy()
    df = df.reset_index(drop=True)

    print(f"Total valid samples: {len(df)}")
    print("\nClass distribution:")
    print(df["final_label"].value_counts())
    print(f"Unique videos: {df['video_id'].nunique()}")

    # ======================================================
    # STEP 1: TrainVal/Test split by video_id
    # ======================================================
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    trainval_idx, test_idx = next(
        gss_test.split(
            df,
            y=df["final_label"],
            groups=df["video_id"],
        )
    )

    trainval_df = df.iloc[trainval_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # ======================================================
    # STEP 2: Train/Val split by video_id
    # ======================================================
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE_IN_TRAINVAL,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        gss_val.split(
            trainval_df,
            y=trainval_df["final_label"],
            groups=trainval_df["video_id"],
        )
    )

    train_df = trainval_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].copy().reset_index(drop=True)

    check_group_leakage(train_df, val_df, test_df)

    train_df.to_csv(processed_dir / "train_split.csv", index=False)
    val_df.to_csv(processed_dir / "val_split.csv", index=False)
    test_df.to_csv(processed_dir / "test_split.csv", index=False)

    label_counts(train_df, "Train")
    label_counts(val_df, "Val")
    label_counts(test_df, "Test")

    print("\nSaved:")
    print(processed_dir / "train_split.csv")
    print(processed_dir / "val_split.csv")
    print(processed_dir / "test_split.csv")


if __name__ == "__main__":
    main()
