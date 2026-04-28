import numpy as np
import pandas as pd
import torch

INPUT_TABLE_PATH = "./DATA-Table_S2__MPRA_dataset.txt"
OUTPUT_SPLITS_PATH = "./x_top_splits.pt"
TOP_QUANTILE = 0.9
SEQUENCE_LENGTH = 200
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42


def sequence_to_one_hot_numpy(sequence: str) -> np.ndarray:
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    one_hot_array = np.full((len(sequence), 4), -1.0, dtype=np.float32)

    for index, base in enumerate(sequence.upper()):
        if base in base_to_idx:
            one_hot_array[index, base_to_idx[base]] = 1.0

    return one_hot_array


def dataframe_to_tensor(dataframe: pd.DataFrame) -> torch.Tensor:
    encoded_sequences = [sequence_to_one_hot_numpy(sequence) for sequence in dataframe["sequence"]]
    sequence_array = np.stack(encoded_sequences).transpose(0, 2, 1)
    return torch.from_numpy(sequence_array)


def main() -> None:
    if not np.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0.")

    dataframe = pd.read_csv(INPUT_TABLE_PATH, sep="\t", engine="python")
    dataframe = dataframe[dataframe["sequence"].str.len() == SEQUENCE_LENGTH]
    dataframe = dataframe[["chr", "sequence", "HepG2_log2FC"]]
    dataframe["chr"] = dataframe["chr"].astype(str)

    total_count = len(dataframe)
    chr1_count = int((dataframe["chr"] == "1").sum())
    chr2_count = int((dataframe["chr"] == "2").sum())
    non_held_out_dataframe = dataframe[~dataframe["chr"].isin(["1", "2"])].copy()
    non_held_out_count = len(non_held_out_dataframe)

    if non_held_out_count < 3:
        raise ValueError("Not enough samples remain after excluding chr1/chr2 for train/val/test splitting.")

    activity_threshold = float(non_held_out_dataframe["HepG2_log2FC"].quantile(TOP_QUANTILE))
    top_dataframe = non_held_out_dataframe[non_held_out_dataframe["HepG2_log2FC"] >= activity_threshold].copy()
    top_count = len(top_dataframe)

    if top_count < 3:
        raise ValueError("Not enough top-activity non-held-out samples remain for train/val/test splitting.")

    shuffled_dataframe = top_dataframe.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    val_size = int(top_count * VAL_RATIO)
    test_size = int(top_count * TEST_RATIO)
    train_size = top_count - val_size - test_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("Computed split sizes must all be positive after top-activity non-held-out filtering.")

    train_dataframe = shuffled_dataframe.iloc[:train_size].copy()
    val_dataframe = shuffled_dataframe.iloc[train_size : train_size + val_size].copy()
    test_dataframe = shuffled_dataframe.iloc[train_size + val_size :].copy()

    print(f"total 200bp samples: {total_count}")
    print(f"excluded chr1 samples before top-{1 - TOP_QUANTILE:.0%} selection: {chr1_count}")
    print(f"excluded chr2 samples before top-{1 - TOP_QUANTILE:.0%} selection: {chr2_count}")
    print(f"non-held-out samples after exclusion: {non_held_out_count}")
    print(f"top-activity threshold on non-held-out samples: {activity_threshold:.6f}")
    print(f"selected top-{1 - TOP_QUANTILE:.0%} non-held-out samples: {top_count}")
    print(f"split ratios: train={TRAIN_RATIO:.1%}, val={VAL_RATIO:.1%}, test={TEST_RATIO:.1%}")
    print(f"train samples: {len(train_dataframe)}")
    print(f"validation samples: {len(val_dataframe)}")
    print(f"test samples: {len(test_dataframe)}")

    train_tensor = dataframe_to_tensor(train_dataframe)
    val_tensor = dataframe_to_tensor(val_dataframe)
    test_tensor = dataframe_to_tensor(test_dataframe)

    print("\nFinal tensor shapes:")
    print(f"train: {tuple(train_tensor.shape)}, dtype={train_tensor.dtype}")
    print(f"validation: {tuple(val_tensor.shape)}, dtype={val_tensor.dtype}")
    print(f"test: {tuple(test_tensor.shape)}, dtype={test_tensor.dtype}")

    split_dict = {
        "train": train_tensor,
        "val": val_tensor,
        "test": test_tensor,
        "metadata": {
            "excluded_chromosomes": ["1", "2"],
            "total_count": total_count,
            "excluded_chr1_count": chr1_count,
            "excluded_chr2_count": chr2_count,
            "non_held_out_count": non_held_out_count,
            "selected_top_count": top_count,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "random_seed": RANDOM_SEED,
            "top_quantile": TOP_QUANTILE,
            "activity_threshold": activity_threshold,
        },
    }
    torch.save(split_dict, OUTPUT_SPLITS_PATH)
    print(f"\nSaved train/val/test splits to {OUTPUT_SPLITS_PATH}")


if __name__ == "__main__":
    main()
