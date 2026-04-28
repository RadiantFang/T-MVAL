"""Value Network Warm-up: generate supervision data and pretrain the value model."""

import math
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from Diffusion.diffusion import Diffusion
from Diffusion.unet import UNet
from enformer_regressor import EnformerModel
from MHVN.mhvn import EnsembleModelMotified

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

DEVICE_INDEX = 0
ENFORMER_CHECKPOINT = "./checkpoints/oracle_finetune.ckpt"
DIFFUSION_CHECKPOINT = "./checkpoints/pretrain.pt"
VALUE_MODEL_WEIGHTS_PATH = "./checkpoints/best_ensemble_model.pt"

NUM_TRAJECTORIES = 128000
GENERATION_BATCH_SIZE = 512
TIMESTEPS = 50
TIMESTEPS_TO_OPTIMIZE = 50
IMAGE_SIZE = (GENERATION_BATCH_SIZE, 1, 4, 200)

VALUE_BATCH_SIZE = 512
VALUE_EPOCHS = 42
VALUE_LEARNING_RATE = 1e-5
VAL_RATIO = 0.2
VALUE_REPLAY_SEED_PATH = "./checkpoints/value_replay_seed.pt"
VALUE_REPLAY_SEED_SAMPLE_SIZE = 256_000

def load_reward_model():
    print("Loading reward model...")
    model = EnformerModel.load_from_checkpoint(
        ENFORMER_CHECKPOINT,
        map_location=DEVICE,
    )
    model = model.to(DEVICE)
    model.eval()
    return model


def load_diffusion_model():
    print("Loading diffusion model...")
    checkpoint_dict = torch.load(DIFFUSION_CHECKPOINT, map_location=DEVICE)
    model = UNet(dim=200, channels=1, dim_mults=[1, 2, 4], resnet_block_groups=4)
    diffusion = Diffusion(model, timesteps=TIMESTEPS, beta_start=0.0001, beta_end=0.2)
    diffusion.load_state_dict(checkpoint_dict["model"])
    diffusion.to(DEVICE)
    diffusion.eval()
    return diffusion


def get_rewards(model, x_batch_neg1_1):
    x_batch_0_1 = torch.argmax(x_batch_neg1_1, dim=1)
    input_tensor = F.one_hot(x_batch_0_1, num_classes=4).float()
    with torch.no_grad():
        predictions = model(input_tensor.to(DEVICE))
    return predictions.detach().cpu().float().view(-1)


def generate_training_tensors(reward_model, diffusion):
    num_batches = math.ceil(NUM_TRAJECTORIES / GENERATION_BATCH_SIZE)
    training_data_list = []

    print("Starting data generation using reverse denoising sampling...")
    for batch_idx in tqdm(range(num_batches), desc="Generating Trajectories"):
        current_batch_size = min(
            GENERATION_BATCH_SIZE,
            NUM_TRAJECTORIES - batch_idx * GENERATION_BATCH_SIZE,
        )
        with torch.no_grad():
            trajectory_numpy_list = diffusion.p_sample_loop(
                classes=None,
                image_size=(current_batch_size, 1, 4, 200),
                cond_weight=1,
            )

        x_0_tensor = torch.from_numpy(trajectory_numpy_list[-1]).to(DEVICE).squeeze(1)
        final_rewards = get_rewards(reward_model, x_0_tensor).cpu()

        for t_index, x_t_numpy in enumerate(trajectory_numpy_list[-TIMESTEPS_TO_OPTIMIZE:]):
            t_actual = TIMESTEPS_TO_OPTIMIZE - 1 - t_index
            t_tensor = torch.full(
                (current_batch_size,),
                t_actual,
                device="cpu",
                dtype=torch.long,
            )
            x_t_tensor = torch.from_numpy(x_t_numpy).squeeze(1)
            x_t_for_value_net = x_t_tensor.permute(0, 2, 1)
            training_data_list.append((x_t_for_value_net.cpu(), t_tensor, final_rewards))

    all_x_t = torch.cat([data[0] for data in training_data_list], dim=0).float()
    all_t = torch.cat([data[1] for data in training_data_list], dim=0)
    all_rewards = torch.cat([data[2] for data in training_data_list], dim=0).float().view(-1)

    print("Data generation complete.")
    print("Final dataset shapes:")
    print("Noisy Samples (x_t):", all_x_t.shape)
    print("Timesteps (t):", all_t.shape)
    print("Final Rewards (R):", all_rewards.shape)
    return all_x_t, all_t, all_rewards


def build_value_dataloaders(x_t, t, rewards):
    dataset = TensorDataset(x_t, t, rewards)
    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=VALUE_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VALUE_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader


def save_value_replay_seed(x_t, t, rewards):
    seed_sample_size = min(VALUE_REPLAY_SEED_SAMPLE_SIZE, len(x_t))
    if seed_sample_size <= 0:
        print("Skipped saving value replay seed: dataset is empty.")
        return

    sample_generator = torch.Generator().manual_seed(42)
    sample_indices = torch.randperm(len(x_t), generator=sample_generator)[:seed_sample_size]

    seed_payload = {
        "x_t": x_t[sample_indices].cpu().float(),
        "t": t[sample_indices].cpu().long(),
        "rewards": rewards[sample_indices].cpu().float(),
        "metadata": {
            "source_num_samples": len(x_t),
            "seed_num_samples": seed_sample_size,
            "timesteps": TIMESTEPS,
            "timesteps_to_optimize": TIMESTEPS_TO_OPTIMIZE,
        },
    }
    os.makedirs(os.path.dirname(VALUE_REPLAY_SEED_PATH), exist_ok=True)
    torch.save(seed_payload, VALUE_REPLAY_SEED_PATH)
    print(
        f"Saved value replay seed to {VALUE_REPLAY_SEED_PATH} "
        f"with {seed_sample_size} samples."
    )


def compute_loss(model, x_t, t, rewards):
    predictions_tensor = torch.stack(model(x_t, t), dim=1)
    targets = rewards.view(-1, 1).expand_as(predictions_tensor)
    return F.mse_loss(predictions_tensor, targets)


def train_value_model(train_loader, val_loader):
    torch.set_float32_matmul_precision("medium")

    model = EnsembleModelMotified().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=VALUE_LEARNING_RATE)
    best_val_loss = float("inf")

    os.makedirs(os.path.dirname(VALUE_MODEL_WEIGHTS_PATH), exist_ok=True)
    print("\n--- Starting value model training ---")

    for epoch in range(VALUE_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for x_t, t, rewards in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{VALUE_EPOCHS} [train]",
            leave=False,
        ):
            x_t = x_t.to(DEVICE)
            t = t.to(DEVICE)
            rewards = rewards.to(DEVICE)

            loss = compute_loss(model, x_t, t, rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_t, t, rewards in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{VALUE_EPOCHS} [val]",
                leave=False,
            ):
                x_t = x_t.to(DEVICE)
                t = t.to(DEVICE)
                rewards = rewards.to(DEVICE)
                total_val_loss += compute_loss(model, x_t, t, rewards).item()

        avg_val_loss = total_val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), VALUE_MODEL_WEIGHTS_PATH)
            print(f"Saved best model, val loss: {best_val_loss:.6f}")

        print(
            f"Epoch [{epoch + 1}/{VALUE_EPOCHS}], "
            f"train loss: {avg_train_loss:.6f}, "
            f"val loss: {avg_val_loss:.6f}"
        )

    print(f"Training complete. Best weights saved to {VALUE_MODEL_WEIGHTS_PATH}")


def main():
    reward_model = load_reward_model()
    diffusion = load_diffusion_model()

    all_x_t, all_t, all_rewards = generate_training_tensors(reward_model, diffusion)
    save_value_replay_seed(all_x_t, all_t, all_rewards)

    del reward_model
    del diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loader, val_loader = build_value_dataloaders(all_x_t, all_t, all_rewards)
    train_value_model(train_loader, val_loader)


if __name__ == "__main__":
    main()
