import argparse
from copy import deepcopy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from grelu.lightning import LightningModel
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from enformer_regressor import EnformerModel
from MHVN.mhvn import EnsembleModelMotified
from utils.finetuning_utils import (
    calculate_kmer_similarity,
    count_kmers,
    freeze_module,
    get_enformer,
    load_diffusion_model,
    p_sample_guided,
    tensor_to_dna_numpy,
)

LEARNING_RATE_DIFFUSION = 1e-5
LEARNING_RATE_VALUE = 5e-5
BATCH_SIZE = 320
TIMESTEPS = 50
VALUE_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 640
TIMESTEPS_TO_OPTIMIZE = 50
SAVE_ACTIVITY_THRESHOLD = 7.0
NUM_FINE_TUNING_STEPS = 300
KL_PENALTY_WEIGHT = 0.5
VALUE_REPLAY_BUFFER_CAPACITY = 256_000
VALUE_REPLAY_SAMPLE_SIZE = VALUE_BATCH_SIZE * TIMESTEPS_TO_OPTIMIZE
VALUE_REPLAY_SEED_PATH = "./checkpoints/value_replay_seed.pt"
PRETRAINED_DIFFUSION_CHECKPOINT = "./checkpoints/pretrain.pt"
VALUE_MODEL_CHECKPOINT = "./checkpoints/best_ensemble_model.pt"
MHVN_TRAINING_ENFORMER_CHECKPOINT = "./checkpoints/oracle_finetune.ckpt"
ENFORMER_CHECKPOINT = "./checkpoints/oracle_eval.ckpt"
ATAC_CHECKPOINT = "./checkpoints/epoch=3-step=3204.ckpt"
NATURAL_SEQUENCE_PATH = "./top_0_1pct_reference_sequences.txt"

device = None
trainable_diffusion = None
value_model = None
mhvn_training_enformermodel = None
enformermodel = None
reference_diffusion = None
atac_model = None
reference_kmer_counts = None
diffusion_optimizer = None
value_optimizer = None
value_replay_buffer = None
value_replay_enabled = True
value_replay_seed_path = VALUE_REPLAY_SEED_PATH


def format_metric(value):
    return f"{float(value):.4f}"


def parse_bool_arg(value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value == "true":
        return True
    if normalized_value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


class ValueReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.state_buffer = None
        self.timestep_buffer = None
        self.reward_buffer = None
        self.size = 0
        self.write_index = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state_tensors: torch.Tensor,
        timestep_tensors: torch.Tensor,
        reward_tensors: torch.Tensor,
    ) -> None:
        states = state_tensors.detach().cpu().float()
        timesteps = timestep_tensors.detach().cpu().long()
        rewards = reward_tensors.detach().cpu().float().view(-1)

        if not (len(states) == len(timesteps) == len(rewards)):
            raise ValueError("Replay buffer inputs must have matching batch sizes.")

        batch_size = len(states)
        if batch_size == 0:
            return

        if self.state_buffer is None:
            self.state_buffer = torch.empty((self.capacity, *states.shape[1:]), dtype=states.dtype)
            self.timestep_buffer = torch.empty((self.capacity,), dtype=timesteps.dtype)
            self.reward_buffer = torch.empty((self.capacity,), dtype=rewards.dtype)

        if batch_size >= self.capacity:
            states = states[-self.capacity:]
            timesteps = timesteps[-self.capacity:]
            rewards = rewards[-self.capacity:]
            batch_size = self.capacity

        first_chunk_size = min(batch_size, self.capacity - self.write_index)
        second_chunk_size = batch_size - first_chunk_size

        self.state_buffer[self.write_index:self.write_index + first_chunk_size] = states[:first_chunk_size]
        self.timestep_buffer[self.write_index:self.write_index + first_chunk_size] = timesteps[:first_chunk_size]
        self.reward_buffer[self.write_index:self.write_index + first_chunk_size] = rewards[:first_chunk_size]

        if second_chunk_size > 0:
            self.state_buffer[:second_chunk_size] = states[first_chunk_size:]
            self.timestep_buffer[:second_chunk_size] = timesteps[first_chunk_size:]
            self.reward_buffer[:second_chunk_size] = rewards[first_chunk_size:]

        self.write_index = (self.write_index + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, sample_size: int):
        if self.size == 0 or sample_size <= 0:
            return None

        actual_sample_size = min(sample_size, self.size)
        sample_indices = torch.randint(0, self.size, (actual_sample_size,), dtype=torch.long)
        return (
            self.state_buffer[sample_indices],
            self.timestep_buffer[sample_indices],
            self.reward_buffer[sample_indices],
        )


def print_setup_log(enable_kl_penalty: bool, enable_value_replay: bool):
    print("[setup] device=", device, sep="")
    print(
        "[setup] diffusion_lr="
        f"{LEARNING_RATE_DIFFUSION} value_lr={LEARNING_RATE_VALUE} "
        f"batch_size={BATCH_SIZE} value_batch_size={VALUE_BATCH_SIZE} "
        f"eval_batch_size={EVAL_BATCH_SIZE} guided_steps={TIMESTEPS_TO_OPTIMIZE}"
    )
    print(
        "[setup] checkpoints: "
        f"diffusion={PRETRAINED_DIFFUSION_CHECKPOINT}, "
        f"value={VALUE_MODEL_CHECKPOINT}, "
        f"mhvn_activity={MHVN_TRAINING_ENFORMER_CHECKPOINT}, "
        f"eval_activity={ENFORMER_CHECKPOINT}, "
        f"atac={ATAC_CHECKPOINT}"
    )
    print(
        "[setup] kl_penalty="
        f"{'enabled' if enable_kl_penalty else 'disabled'} "
        f"(weight={format_metric(KL_PENALTY_WEIGHT)})"
    )
    if enable_value_replay:
        print(
            "[setup] value_replay=enabled "
            f"(capacity={VALUE_REPLAY_BUFFER_CAPACITY} "
            f"sample_size={VALUE_REPLAY_SAMPLE_SIZE} "
            f"seed_path={value_replay_seed_path})"
        )
    else:
        print("[setup] value_replay=disabled")
    print(f"[setup] save_threshold.activity_mean>{format_metric(SAVE_ACTIVITY_THRESHOLD)}")


def print_step_log(step_idx):
    print(f"\n[step {step_idx + 1}/{NUM_FINE_TUNING_STEPS}] start")


def print_guided_update_log(step_idx, guided_step, value_objective, kl_penalty, total_loss):
    print(
        f"[diffusion][step {step_idx + 1:03d}][t={guided_step:02d}] "
        f"value_objective={format_metric(value_objective.mean())} "
        f"kl={format_metric(kl_penalty)} "
        f"total_loss={format_metric(total_loss)}"
    )


def print_value_warmup_log(step_idx, guided_step, dataset_size, mean_loss):
    print(
        f"[value][step {step_idx + 1:03d}][t={guided_step:02d}] "
        f"dataset_size={dataset_size} "
        f"mean_loss={format_metric(mean_loss)}"
    )


def print_value_replay_log(step_idx, guided_step, fresh_size, replay_size, replay_buffer_size):
    print(
        f"[value][step {step_idx + 1:03d}][t={guided_step:02d}] "
        f"fresh_size={fresh_size} "
        f"replay_size={replay_size} "
        f"replay_buffer_size={replay_buffer_size}"
    )


def print_value_replay_seed_log(seed_path, loaded_size=None):
    if loaded_size is None:
        print(f"[setup] value_replay_seed_missing path={seed_path}; starting from empty replay buffer")
        return

    print(f"[setup] value_replay_seed_loaded path={seed_path} loaded_size={loaded_size}")


def print_evaluation_log(step_idx, mean_activity_value, atac_positive_rate, kmer_similarity):
    print(
        f"[eval][step {step_idx + 1:03d}] "
        f"activity_mean={format_metric(mean_activity_value)} "
        f"atac_positive_rate={format_metric(atac_positive_rate)} "
        f"kmer_corr={format_metric(kmer_similarity)}"
    )


def print_checkpoint_log(step_idx, mean_activity_value, checkpoint_path=None):
    if checkpoint_path is None:
        print(
            f"[checkpoint][step {step_idx + 1:03d}] skipped "
            f"(activity_mean={format_metric(mean_activity_value)} <= "
            f"{format_metric(SAVE_ACTIVITY_THRESHOLD)})"
        )
        return

    print(
        f"[checkpoint][step {step_idx + 1:03d}] saved "
        f"{checkpoint_path} "
        f"(activity_mean={format_metric(mean_activity_value)})"
    )


def build_value_training_dataset():
    value_training_batches = []
    was_training = trainable_diffusion.training
    trainable_diffusion.eval()
    with torch.no_grad():
        sample_shape = (VALUE_BATCH_SIZE, 1, 4, 200)
        for _ in range(1):
            sample_trajectory = torch.from_numpy(
                np.array(
                    trainable_diffusion.p_sample_loop(
                        classes=None,
                        image_size=sample_shape,
                        cond_weight=1,
                    )
                )
            ).squeeze(2)
            reward_scores = get_enformer(mhvn_training_enformermodel, sample_trajectory[-1].to(device))

            for reverse_step_index, state_tensor in enumerate(sample_trajectory[-TIMESTEPS_TO_OPTIMIZE:]):
                timestep_tensor = torch.full(
                    (VALUE_BATCH_SIZE,),
                    TIMESTEPS_TO_OPTIMIZE - 1 - reverse_step_index,
                    device=device,
                    dtype=torch.long,
                )
                value_training_batches.append(
                    (state_tensor.permute(0, 2, 1), timestep_tensor, reward_scores)
                )

        fresh_state_tensors = torch.cat([batch[0] for batch in value_training_batches], dim=0).detach().cpu().float()
        fresh_timestep_tensors = torch.cat([batch[1] for batch in value_training_batches], dim=0).detach().cpu().long()
        fresh_reward_scores = torch.cat([batch[2] for batch in value_training_batches], dim=0).detach().cpu().float()
    trainable_diffusion.train(was_training)

    state_tensor_groups = [fresh_state_tensors]
    timestep_tensor_groups = [fresh_timestep_tensors]
    reward_tensor_groups = [fresh_reward_scores]
    replay_sample_size = 0
    replay_buffer_size = 0

    if value_replay_enabled and value_replay_buffer is not None:
        replay_sample = value_replay_buffer.sample(VALUE_REPLAY_SAMPLE_SIZE)
        value_replay_buffer.add(fresh_state_tensors, fresh_timestep_tensors, fresh_reward_scores)
        replay_buffer_size = len(value_replay_buffer)

        if replay_sample is not None:
            replay_state_tensors, replay_timestep_tensors, replay_reward_scores = replay_sample
            state_tensor_groups.append(replay_state_tensors)
            timestep_tensor_groups.append(replay_timestep_tensors)
            reward_tensor_groups.append(replay_reward_scores)
            replay_sample_size = len(replay_state_tensors)

    all_state_tensors = torch.cat(state_tensor_groups, dim=0)
    all_timestep_tensors = torch.cat(timestep_tensor_groups, dim=0)
    all_reward_scores = torch.cat(reward_tensor_groups, dim=0)

    return (
        TensorDataset(all_state_tensors, all_timestep_tensors, all_reward_scores),
        len(fresh_state_tensors),
        replay_sample_size,
        replay_buffer_size,
    )


def preload_value_replay_buffer_from_seed():
    if value_replay_buffer is None:
        return

    if not os.path.exists(value_replay_seed_path):
        print_value_replay_seed_log(value_replay_seed_path)
        return

    seed_payload = torch.load(value_replay_seed_path, map_location="cpu")
    seed_states = seed_payload["x_t"].float()
    seed_timesteps = seed_payload["t"].long()
    seed_rewards = seed_payload["rewards"].float()
    value_replay_buffer.add(seed_states, seed_timesteps, seed_rewards)
    print_value_replay_seed_log(value_replay_seed_path, len(value_replay_buffer))


def evaluate_current_model(step_idx):
    was_training = trainable_diffusion.training
    trainable_diffusion.eval()
    with torch.no_grad():
        sampled_trajectory = torch.from_numpy(
            np.array(
                trainable_diffusion.p_sample_loop(
                    classes=None,
                    image_size=(EVAL_BATCH_SIZE, 1, 4, 200),
                    cond_weight=1,
                )
            )
        ).squeeze(2)
        final_step_logits = sampled_trajectory[-1].to(device)
        predicted_tokens = torch.argmax(final_step_logits, dim=1)
        one_hot_inputs = F.one_hot(predicted_tokens, num_classes=4).float()
        mean_activity = enformermodel(one_hot_inputs).mean()
        mean_activity_value = mean_activity.item()
        atac_predictions = atac_model(one_hot_inputs.permute(0, 2, 1)).detach().cpu().numpy()
        atac_positive_rate = (atac_predictions.squeeze()[:, 1] > 0.5).sum() / EVAL_BATCH_SIZE

        generated_kmer_counts = count_kmers(tensor_to_dna_numpy(predicted_tokens), 3)
        kmer_similarity = calculate_kmer_similarity(reference_kmer_counts, generated_kmer_counts)
        print_evaluation_log(step_idx, mean_activity_value, atac_positive_rate, kmer_similarity)
    trainable_diffusion.train(was_training)
    return mean_activity_value


def fine_tune_lora_model(
        lora_value_model: PeftModel,
        value_optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        step_idx: int,
        guided_step: int,
        num_epochs: int = 3,
) -> PeftModel:
    _ = val_loader
    lora_value_model.to(device)
    loss_fn = nn.MSELoss()

    for epoch_idx in range(num_epochs):
        lora_value_model.train()
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        train_bar = tqdm(
            train_loader,
            desc=f"[value][step {step_idx + 1:03d}][t={guided_step:02d}] epoch {epoch_idx + 1}/{num_epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_data in train_bar:
            *model_inputs, target_rewards = batch_data
            model_inputs = [item.to(device) for item in model_inputs]
            target_rewards = target_rewards.to(device)
            value_optimizer.zero_grad()
            head_predictions = lora_value_model(*model_inputs)
            head_predictions = torch.stack(head_predictions, dim=1)
            expanded_targets = target_rewards.unsqueeze(1)
            broadcast_targets = expanded_targets.expand_as(head_predictions)
            loss = loss_fn(head_predictions, broadcast_targets)
            loss.backward()
            value_optimizer.step()
            loss_value = loss.item()
            epoch_loss_sum += loss_value
            epoch_batch_count += 1
            train_bar.set_postfix({"batch_loss": format_metric(loss_value)})
        mean_epoch_loss = epoch_loss_sum / max(epoch_batch_count, 1)
        print_value_warmup_log(step_idx, guided_step, len(train_loader.dataset), mean_epoch_loss)

    return lora_value_model


def main():
    global device
    global LEARNING_RATE_DIFFUSION
    global LEARNING_RATE_VALUE
    global BATCH_SIZE
    global TIMESTEPS
    global VALUE_BATCH_SIZE
    global EVAL_BATCH_SIZE
    global TIMESTEPS_TO_OPTIMIZE
    global SAVE_ACTIVITY_THRESHOLD
    global NUM_FINE_TUNING_STEPS
    global KL_PENALTY_WEIGHT
    global VALUE_REPLAY_BUFFER_CAPACITY
    global VALUE_REPLAY_SAMPLE_SIZE
    global VALUE_REPLAY_SEED_PATH
    global PRETRAINED_DIFFUSION_CHECKPOINT
    global VALUE_MODEL_CHECKPOINT
    global MHVN_TRAINING_ENFORMER_CHECKPOINT
    global ENFORMER_CHECKPOINT
    global ATAC_CHECKPOINT
    global NATURAL_SEQUENCE_PATH
    global trainable_diffusion
    global value_model
    global mhvn_training_enformermodel
    global enformermodel
    global reference_diffusion
    global atac_model
    global reference_kmer_counts
    global diffusion_optimizer
    global value_optimizer
    global value_replay_buffer
    global value_replay_enabled
    global value_replay_seed_path

    parser = argparse.ArgumentParser(
        description="Fine-tune diffusion with optional KL-style regularization."
    )
    parser.add_argument("--learning-rate-diffusion", type=float, default=LEARNING_RATE_DIFFUSION)
    parser.add_argument("--learning-rate-value", type=float, default=LEARNING_RATE_VALUE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--value-batch-size", type=int, default=VALUE_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--timesteps-to-optimize", type=int, default=TIMESTEPS_TO_OPTIMIZE)
    parser.add_argument("--save-activity-threshold", type=float, default=SAVE_ACTIVITY_THRESHOLD)
    parser.add_argument("--num-fine-tuning-steps", type=int, default=NUM_FINE_TUNING_STEPS)
    parser.add_argument("--kl-penalty-weight", type=float, default=KL_PENALTY_WEIGHT)
    parser.add_argument("--value-replay-buffer-capacity", type=int, default=VALUE_REPLAY_BUFFER_CAPACITY)
    parser.add_argument(
        "--value-replay-sample-size",
        type=int,
        default=None,
        help="Replay sample size for value training. Defaults to value_batch_size * timesteps_to_optimize.",
    )
    parser.add_argument(
        "--value-replay-seed-path",
        type=str,
        default=VALUE_REPLAY_SEED_PATH,
    )
    parser.add_argument(
        "--pretrained-diffusion-checkpoint",
        type=str,
        default=PRETRAINED_DIFFUSION_CHECKPOINT,
    )
    parser.add_argument("--value-model-checkpoint", type=str, default=VALUE_MODEL_CHECKPOINT)
    parser.add_argument(
        "--mhvn-training-enformer-checkpoint",
        type=str,
        default=MHVN_TRAINING_ENFORMER_CHECKPOINT,
    )
    parser.add_argument("--enformer-checkpoint", type=str, default=ENFORMER_CHECKPOINT)
    parser.add_argument("--atac-checkpoint", type=str, default=ATAC_CHECKPOINT)
    parser.add_argument("--natural-sequence-path", type=str, default=NATURAL_SEQUENCE_PATH)
    parser.add_argument(
        "--enable-kl-penalty",
        type=parse_bool_arg,
        default=False,
        metavar="{True,False}",
        help="Whether to enable the penalty against a frozen reference diffusion model.",
    )
    parser.add_argument(
        "--enable-value-replay",
        type=parse_bool_arg,
        default=True,
        metavar="{True,False}",
        help="Whether to enable the replay buffer for value model training.",
    )
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive.")
    if args.timesteps_to_optimize <= 0:
        parser.error("--timesteps-to-optimize must be positive.")
    if args.timesteps_to_optimize > args.timesteps:
        parser.error("--timesteps-to-optimize must be less than or equal to --timesteps.")
    if args.batch_size <= 0 or args.value_batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("Batch size arguments must be positive.")
    if args.num_fine_tuning_steps <= 0:
        parser.error("--num-fine-tuning-steps must be positive.")
    if args.value_replay_buffer_capacity <= 0:
        parser.error("--value-replay-buffer-capacity must be positive.")
    if args.value_replay_sample_size is not None and args.value_replay_sample_size <= 0:
        parser.error("--value-replay-sample-size must be positive when provided.")

    LEARNING_RATE_DIFFUSION = args.learning_rate_diffusion
    LEARNING_RATE_VALUE = args.learning_rate_value
    BATCH_SIZE = args.batch_size
    TIMESTEPS = args.timesteps
    VALUE_BATCH_SIZE = args.value_batch_size
    EVAL_BATCH_SIZE = args.eval_batch_size
    TIMESTEPS_TO_OPTIMIZE = args.timesteps_to_optimize
    SAVE_ACTIVITY_THRESHOLD = args.save_activity_threshold
    NUM_FINE_TUNING_STEPS = args.num_fine_tuning_steps
    KL_PENALTY_WEIGHT = args.kl_penalty_weight
    VALUE_REPLAY_BUFFER_CAPACITY = args.value_replay_buffer_capacity
    VALUE_REPLAY_SAMPLE_SIZE = (
        args.value_replay_sample_size
        if args.value_replay_sample_size is not None
        else VALUE_BATCH_SIZE * TIMESTEPS_TO_OPTIMIZE
    )
    VALUE_REPLAY_SEED_PATH = args.value_replay_seed_path
    value_replay_seed_path = VALUE_REPLAY_SEED_PATH
    PRETRAINED_DIFFUSION_CHECKPOINT = args.pretrained_diffusion_checkpoint
    VALUE_MODEL_CHECKPOINT = args.value_model_checkpoint
    MHVN_TRAINING_ENFORMER_CHECKPOINT = args.mhvn_training_enformer_checkpoint
    ENFORMER_CHECKPOINT = args.enformer_checkpoint
    ATAC_CHECKPOINT = args.atac_checkpoint
    NATURAL_SEQUENCE_PATH = args.natural_sequence_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    value_replay_enabled = args.enable_value_replay
    print_setup_log(args.enable_kl_penalty, value_replay_enabled)

    trainable_diffusion = load_diffusion_model(
        PRETRAINED_DIFFUSION_CHECKPOINT,
        device,
        timesteps=TIMESTEPS,
    )
    print("[setup] loaded pretrained diffusion")

    value_backbone = EnsembleModelMotified().to(device)
    value_backbone.load_state_dict(torch.load(VALUE_MODEL_CHECKPOINT, map_location=device))
    print("[setup] loaded pretrained MHVN model")

    mhvn_training_enformermodel = EnformerModel.load_from_checkpoint(
        MHVN_TRAINING_ENFORMER_CHECKPOINT,
        map_location=device,
    )
    mhvn_training_enformermodel = mhvn_training_enformermodel.to(device)
    freeze_module(mhvn_training_enformermodel)
    print("[setup] loaded finetuning activity model")

    enformermodel = EnformerModel.load_from_checkpoint(ENFORMER_CHECKPOINT, map_location=device)
    enformermodel = enformermodel.to(device)
    freeze_module(enformermodel)
    print("[setup] loaded evaluation activity model")

    if args.enable_kl_penalty:
        reference_diffusion = deepcopy(trainable_diffusion).to(device)
        freeze_module(reference_diffusion)
        print("[setup] prepared frozen reference diffusion")
    else:
        reference_diffusion = None

    with open(NATURAL_SEQUENCE_PATH, "r") as f:
        natural_sequences = [line.strip().upper() for line in f if line.strip()]
    atac_model = LightningModel.load_from_checkpoint(
        ATAC_CHECKPOINT,
        map_location=device,
    )
    freeze_module(atac_model)
    reference_kmer_counts = count_kmers(natural_sequences, 3)
    print(f"[setup] loaded ATAC model and {len(natural_sequences)} reference sequences")

    modules_to_save = [f"mlp_heads.{i}" for i in range(7)]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["attn_qkv", "attn_out", "mlp.0", "mlp.2"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=modules_to_save,
    )

    value_model = get_peft_model(value_backbone, lora_config)

    diffusion_optimizer = torch.optim.AdamW(
        trainable_diffusion.model.parameters(),
        lr=LEARNING_RATE_DIFFUSION,
    )
    value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=LEARNING_RATE_VALUE)
    if value_replay_enabled:
        value_replay_buffer = ValueReplayBuffer(VALUE_REPLAY_BUFFER_CAPACITY)
        preload_value_replay_buffer_from_seed()
    else:
        value_replay_buffer = None

    for step_idx in range(NUM_FINE_TUNING_STEPS):
        print_step_log(step_idx)
        trainable_diffusion.train()

        current_states = torch.randn((BATCH_SIZE, 1, 4, 200), device=device)
        trainable_diffusion.eval()
        for timestep_idx in reversed(range(TIMESTEPS_TO_OPTIMIZE, TIMESTEPS)):
            with torch.no_grad():
                current_states, _ = p_sample_guided(
                    trainable_diffusion,
                    current_states,
                    torch.full((BATCH_SIZE,), timestep_idx, device=device, dtype=torch.long),
                    timestep_idx,
                    device,
                )
        current_states = current_states.detach()
        trainable_diffusion.train()

        for guided_step in reversed(range(TIMESTEPS_TO_OPTIMIZE)):
            value_model.eval()
            value_parameter_states = [parameter.requires_grad for parameter in value_model.parameters()]
            for parameter in value_model.parameters():
                parameter.requires_grad_(False)
            timestep_tensor = torch.full((BATCH_SIZE,), guided_step, device=device, dtype=torch.long)
            predicted_states, predicted_noise = p_sample_guided(
                trainable_diffusion,
                current_states,
                timestep_tensor,
                guided_step,
                device,
            )

            ensemble_predictions = value_model(
                predicted_states.squeeze(1).permute(0, 2, 1),
                torch.full((BATCH_SIZE,), guided_step, dtype=torch.long, device=device),
            )
            ensemble_prediction_tensor = torch.stack(ensemble_predictions, dim=1)
            value_objective = torch.mean(ensemble_prediction_tensor, dim=1) - torch.std(
                ensemble_prediction_tensor,
                dim=1,
            )
            if args.enable_kl_penalty:
                with torch.no_grad():
                    _, reference_noise = p_sample_guided(
                        reference_diffusion,
                        current_states,
                        timestep_tensor,
                        guided_step,
                        device,
                    )
                kl_penalty = F.mse_loss(predicted_noise, reference_noise)
                total_loss = -value_objective.mean() + KL_PENALTY_WEIGHT * kl_penalty
            else:
                kl_penalty = torch.zeros((), device=device)
                total_loss = -value_objective.mean()
            print_guided_update_log(step_idx, guided_step, value_objective, kl_penalty, total_loss)

            diffusion_optimizer.zero_grad()
            total_loss.backward()
            diffusion_optimizer.step()
            for parameter, was_trainable in zip(value_model.parameters(), value_parameter_states):
                parameter.requires_grad_(was_trainable)
            current_states = predicted_states.detach()

            value_dataset, fresh_size, replay_size, replay_buffer_size = build_value_training_dataset()
            print_value_replay_log(step_idx, guided_step, fresh_size, replay_size, replay_buffer_size)
            value_train_loader = DataLoader(value_dataset, batch_size=VALUE_BATCH_SIZE, shuffle=True)

            value_model = fine_tune_lora_model(
                lora_value_model=value_model,
                value_optimizer=value_optimizer,
                train_loader=value_train_loader,
                val_loader=None,
                step_idx=step_idx,
                guided_step=guided_step,
                num_epochs=1,
            )

        mean_activity_value = evaluate_current_model(step_idx)
        if mean_activity_value > SAVE_ACTIVITY_THRESHOLD:
            checkpoint_path = f"./checkpoints/{step_idx}.pt"
            torch.save(trainable_diffusion.state_dict(), checkpoint_path)
            print_checkpoint_log(step_idx, mean_activity_value, checkpoint_path)
        else:
            print_checkpoint_log(step_idx, mean_activity_value)


if __name__ == "__main__":
    main()
