import argparse
from pathlib import Path

import numpy as np
import torch

from Diffusion.diffusion import Diffusion
from Diffusion.unet import UNet

DEFAULT_BATCH_SIZE = 640
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DNA_MAPPING = np.array(["A", "C", "G", "T"])


def tensor_to_dna_numpy(tensor, mapping_array=DNA_MAPPING):
    numpy_tensor = tensor.cpu().numpy()
    char_array = mapping_array[numpy_tensor]
    return ["".join(row) for row in char_array]


def load_diffusion_model(checkpoint_path: str, device: str) -> Diffusion:
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    model = UNet(dim=200, channels=1, dim_mults=[1, 2, 4], resnet_block_groups=4)
    diffusion = Diffusion(model, timesteps=50, beta_start=0.0001, beta_end=0.2)
    diffusion.load_state_dict(checkpoint_dict)
    diffusion.to(device)
    diffusion.eval()
    return diffusion


def sample_sequences(diffusion: Diffusion, batch_size: int) -> torch.Tensor:
    shape = (batch_size, 1, 4, 200)
    sample = diffusion.p_sample_loop(classes=None, image_size=shape, cond_weight=1)[-1]
    return torch.from_numpy(np.array(sample)).squeeze(1)


def write_outputs(output_dir: Path, sampled_sequences: torch.Tensor) -> None:
    tokenized = torch.argmax(sampled_sequences, dim=1)
    sequences = tensor_to_dna_numpy(tokenized)

    with open(output_dir / "samples.txt", "w") as f:
        for seq in sequences:
            f.write(seq + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sample sequences from one or more fine-tuned diffusion checkpoints.")
    parser.add_argument(
        "--fine",
        "-f",
        nargs="+",
        required=True,
        help="One or more fine-tuned checkpoint names without the .pt suffix.",
    )
    parser.add_argument("--device", "-d", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    for fine_name in args.fine:
        output_dir = Path(f"./{fine_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = f"./checkpoints/{fine_name}.pt"
        diffusion = load_diffusion_model(checkpoint_path, args.device)
        print(f"Diffusion model loaded successfully: {fine_name}")

        with torch.no_grad():
            sampled_sequences = sample_sequences(diffusion, args.batch_size)

        write_outputs(output_dir, sampled_sequences)
        print(f"Sampling finished: {fine_name}")


if __name__ == "__main__":
    main()
