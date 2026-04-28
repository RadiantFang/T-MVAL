import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from enformer_regressor import EnformerModel

DEFAULT_INPUT_TABLE_PATH = "./DATA-Table_S2__MPRA_dataset.txt"
DEFAULT_OUTPUT_DIR = "./oracle_training"
DEFAULT_ACTIVITY_COLUMN = "HepG2_log2FC"
DEFAULT_CHROMOSOME_COLUMN = "chr"
DEFAULT_SEQUENCE_COLUMN = "sequence"
DEFAULT_SEQUENCE_LENGTH = 200
DEFAULT_EVAL_HOLDOUT_CHROMOSOME = "1"
DEFAULT_FINETUNE_HOLDOUT_CHROMOSOME = "2"
DEFAULT_FINETUNE_TRAIN_FRACTION = 0.5
DEFAULT_ORACLE_VALIDATION_FRACTION = 0.1
DEFAULT_RANDOM_SEED = 42
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_LOSS = "mse"
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_WORKERS = 1
DEFAULT_MAX_EPOCHS = 10
DEFAULT_DEVICE_INDEX = 0
DEFAULT_FINETUNE_CHECKPOINT_OUTPUT = "./checkpoints/oracle_finetune.ckpt"
DEFAULT_EVAL_CHECKPOINT_OUTPUT = "./checkpoints/oracle_eval.ckpt"


class SequenceActivityDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, sequence_column: str, activity_column: str):
        self.sequences = dataframe[sequence_column].astype(str).tolist()
        self.targets = torch.tensor(
            dataframe[activity_column].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.targets[index]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process MPRA data and train separate fine-tuning/evaluation Enformer oracles."
    )
    parser.add_argument("--input-table-path", default=DEFAULT_INPUT_TABLE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--activity-column", default=DEFAULT_ACTIVITY_COLUMN)
    parser.add_argument("--chromosome-column", default=DEFAULT_CHROMOSOME_COLUMN)
    parser.add_argument("--sequence-column", default=DEFAULT_SEQUENCE_COLUMN)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--eval-holdout-chromosome", default=DEFAULT_EVAL_HOLDOUT_CHROMOSOME)
    parser.add_argument("--finetune-holdout-chromosome", default=DEFAULT_FINETUNE_HOLDOUT_CHROMOSOME)
    parser.add_argument("--finetune-train-fraction", type=float, default=DEFAULT_FINETUNE_TRAIN_FRACTION)
    parser.add_argument("--oracle-validation-fraction", type=float, default=DEFAULT_ORACLE_VALIDATION_FRACTION)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--loss", choices=["mse", "poisson"], default=DEFAULT_LOSS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--device-index", type=int, default=DEFAULT_DEVICE_INDEX)
    parser.add_argument("--finetune-checkpoint-output", default=DEFAULT_FINETUNE_CHECKPOINT_OUTPUT)
    parser.add_argument("--eval-checkpoint-output", default=DEFAULT_EVAL_CHECKPOINT_OUTPUT)
    return parser.parse_args()


def validate_args(args):
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive.")
    if not 0.0 < args.finetune_train_fraction < 1.0:
        raise ValueError("--finetune-train-fraction must be between 0 and 1.")
    if not 0.0 < args.oracle_validation_fraction < 1.0:
        raise ValueError("--oracle-validation-fraction must be between 0 and 1.")
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_epochs <= 0:
        raise ValueError("Batch size, num workers, and max epochs must be valid positive values.")
    if args.eval_holdout_chromosome == args.finetune_holdout_chromosome:
        raise ValueError("Evaluation and fine-tuning holdout chromosomes must be different.")


def load_and_filter_dataframe(args):
    dataframe = pd.read_csv(args.input_table_path, sep="\t", engine="python")
    original_count = len(dataframe)

    dataframe = dataframe.dropna(
        subset=[args.chromosome_column, args.sequence_column, args.activity_column]
    ).copy()
    dataframe[args.sequence_column] = dataframe[args.sequence_column].astype(str).str.upper()
    dataframe = dataframe[dataframe[args.sequence_column].str.len() == args.sequence_length].copy()
    dataframe = dataframe[
        [args.chromosome_column, args.sequence_column, args.activity_column]
    ].copy()
    dataframe[args.chromosome_column] = dataframe[args.chromosome_column].astype(str)

    print(f"[data] loaded rows={original_count}")
    print(f"[data] retained length=={args.sequence_length} rows={len(dataframe)}")
    return dataframe


def build_data_splits(dataframe: pd.DataFrame, args):
    chromosome_column = args.chromosome_column

    eval_test_df = dataframe[dataframe[chromosome_column] == args.eval_holdout_chromosome].copy()
    finetune_test_df = dataframe[dataframe[chromosome_column] == args.finetune_holdout_chromosome].copy()
    non_holdout_df = dataframe[
        ~dataframe[chromosome_column].isin(
            [args.eval_holdout_chromosome, args.finetune_holdout_chromosome]
        )
    ].copy()

    if eval_test_df.empty:
        raise ValueError(f"No samples found for eval holdout chromosome {args.eval_holdout_chromosome}.")
    if finetune_test_df.empty:
        raise ValueError(
            f"No samples found for fine-tuning holdout chromosome {args.finetune_holdout_chromosome}."
        )
    if non_holdout_df.empty:
        raise ValueError("No non-held-out samples remain after chromosome filtering.")

    shuffled_non_holdout_df = non_holdout_df.sample(
        frac=1.0,
        random_state=args.random_seed,
    ).reset_index(drop=True)
    finetune_train_size = int(len(shuffled_non_holdout_df) * args.finetune_train_fraction)

    if finetune_train_size <= 0 or finetune_train_size >= len(shuffled_non_holdout_df):
        raise ValueError("Fine-tuning oracle split produced an empty subset.")

    finetune_pool_df = shuffled_non_holdout_df.iloc[:finetune_train_size].copy()
    eval_pool_df = shuffled_non_holdout_df.iloc[finetune_train_size:].copy()

    finetune_val_size = int(len(finetune_pool_df) * args.oracle_validation_fraction)
    eval_val_size = int(len(eval_pool_df) * args.oracle_validation_fraction)

    if finetune_val_size <= 0 or finetune_val_size >= len(finetune_pool_df):
        raise ValueError("Fine-tuning oracle validation split produced an empty subset.")
    if eval_val_size <= 0 or eval_val_size >= len(eval_pool_df):
        raise ValueError("Evaluation oracle validation split produced an empty subset.")

    finetune_val_df = finetune_pool_df.iloc[:finetune_val_size].copy()
    finetune_train_df = finetune_pool_df.iloc[finetune_val_size:].copy()
    eval_val_df = eval_pool_df.iloc[:eval_val_size].copy()
    eval_train_df = eval_pool_df.iloc[eval_val_size:].copy()

    metadata = {
        "original_filtered_rows": int(len(dataframe)),
        "non_holdout_rows": int(len(non_holdout_df)),
        "finetune_train_rows": int(len(finetune_train_df)),
        "finetune_val_rows": int(len(finetune_val_df)),
        "finetune_test_rows": int(len(finetune_test_df)),
        "eval_train_rows": int(len(eval_train_df)),
        "eval_val_rows": int(len(eval_val_df)),
        "eval_test_rows": int(len(eval_test_df)),
        "eval_test_chromosome": args.eval_holdout_chromosome,
        "finetune_test_chromosome": args.finetune_holdout_chromosome,
        "activity_column": args.activity_column,
        "random_seed": int(args.random_seed),
        "finetune_train_fraction": float(args.finetune_train_fraction),
        "oracle_validation_fraction": float(args.oracle_validation_fraction),
    }
    return {
        "finetune_train_df": finetune_train_df,
        "finetune_val_df": finetune_val_df,
        "finetune_test_df": finetune_test_df,
        "eval_train_df": eval_train_df,
        "eval_val_df": eval_val_df,
        "eval_test_df": eval_test_df,
        "metadata": metadata,
    }


def save_processing_artifacts(splits, args):
    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = processed_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(splits["metadata"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[data] saved metadata to {metadata_path}")


def build_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, args):
    train_dataset = SequenceActivityDataset(train_df, args.sequence_column, args.activity_column)
    val_dataset = SequenceActivityDataset(val_df, args.sequence_column, args.activity_column)
    return train_dataset, val_dataset


def evaluate_oracle_model(model: EnformerModel, test_df: pd.DataFrame, args) -> float:
    test_dataset = SequenceActivityDataset(test_df, args.sequence_column, args.activity_column)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if torch.cuda.is_available():
        eval_device = torch.device(f"cuda:{args.device_index}")
    else:
        eval_device = torch.device("cpu")

    model = model.to(eval_device)
    model.eval()

    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for sequences, targets in test_loader:
            targets = targets.to(eval_device)
            loss = model._compute_loss((sequences, targets))
            total_loss += loss.item()
            batch_count += 1

    if batch_count == 0:
        raise ValueError("Test split is empty.")

    return total_loss / batch_count


def train_oracle_model(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    checkpoint_output: str,
    args,
):
    save_dir = Path(args.output_dir) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[train:{name}] train_rows={len(train_df)} "
        f"val_rows={len(val_df)} "
        f"test_rows={len(test_df)} "
        f"activity={args.activity_column}"
    )

    train_dataset, val_dataset = build_dataloaders(train_df, val_df, args)
    model = EnformerModel(lr=args.learning_rate, loss=args.loss, pretrained=True)
    trainer = model.train_on_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=args.device_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_dir=str(save_dir),
        max_epochs=args.max_epochs,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError(f"No checkpoint was produced for {name}.")

    checkpoint_output_path = Path(checkpoint_output)
    checkpoint_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model_path, checkpoint_output_path)
    best_model = EnformerModel.load_from_checkpoint(str(checkpoint_output_path), map_location="cpu")
    test_loss = evaluate_oracle_model(best_model, test_df, args)
    print(f"[train:{name}] best_checkpoint={best_model_path}")
    print(f"[train:{name}] copied_best_checkpoint_to={checkpoint_output_path}")
    print(f"[train:{name}] test_loss={test_loss:.6f}")


def main():
    args = parse_args()
    validate_args(args)
    pl.seed_everything(args.random_seed, workers=True)

    dataframe = load_and_filter_dataframe(args)
    splits = build_data_splits(dataframe, args)
    save_processing_artifacts(splits, args)

    print(
        "[data] test strategy: "
        f"eval_test=chr{args.eval_holdout_chromosome}, "
        f"finetune_test=chr{args.finetune_holdout_chromosome}"
    )
    print(
        "[data] oracle splits: "
        f"finetune(train={len(splits['finetune_train_df'])}, val={len(splits['finetune_val_df'])}, test={len(splits['finetune_test_df'])}) "
        f"eval(train={len(splits['eval_train_df'])}, val={len(splits['eval_val_df'])}, test={len(splits['eval_test_df'])})"
    )

    train_oracle_model(
        name="oracle_finetune",
        train_df=splits["finetune_train_df"],
        val_df=splits["finetune_val_df"],
        test_df=splits["finetune_test_df"],
        checkpoint_output=args.finetune_checkpoint_output,
        args=args,
    )
    train_oracle_model(
        name="oracle_eval",
        train_df=splits["eval_train_df"],
        val_df=splits["eval_val_df"],
        test_df=splits["eval_test_df"],
        checkpoint_output=args.eval_checkpoint_output,
        args=args,
    )


if __name__ == "__main__":
    main()
