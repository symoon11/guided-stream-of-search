from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset


class Sample(NamedTuple):
    observations: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    ref_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    masks: torch.Tensor
    rollin_masks: torch.Tensor
    target_ids: torch.Tensor
    target_masks: torch.Tensor
    sep_masks: torch.Tensor


class Batch(NamedTuple):
    observations: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    ref_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    masks: torch.Tensor
    rollin_masks: torch.Tensor
    target_ids: torch.Tensor
    target_masks: torch.Tensor
    sep_masks: torch.Tensor


class OnlineBuffer(Dataset):
    """
    Combined Dataset and Buffer
    """

    def __init__(
        self,
        accelerator,
        trajectories_in_buffer: int,
        trajectories_per_sample: int,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        use_whitening: bool = False,
        max_prompt_len: int = 500,
        max_gen_len: int = 50,
        sep_token_id: int = None,
    ):
        self.total_num_traj = trajectories_in_buffer
        self.num_traj_per_sample = trajectories_per_sample
        self.obs_shape = (max_prompt_len + max_gen_len,)
        self.max_gen_len = max_gen_len

        # Buffer Pointers
        self.pos = 0
        self.full = False
        self.sep_token_id = sep_token_id

        # GAE Params
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.use_whitening = use_whitening
        self.reset()

        self.num_processes = accelerator.num_processes

    def is_full(self):
        return self.full

    def reset(self):
        # Base Buffer
        self.pos = 0
        self.full = False

        # Rollout Buffer
        self.observations = torch.zeros(
            (self.total_num_traj, *self.obs_shape), dtype=torch.float32
        )
        self.rewards = torch.zeros(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.returns = torch.zeros(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.values = torch.zeros(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.log_probs = torch.zeros(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.ref_log_probs = torch.zeros(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.advantages = torch.zeros(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.masks = torch.zeros(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.rollin_masks = torch.ones(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.target_ids = torch.zeros(
            (self.total_num_traj, *self.obs_shape), dtype=torch.float32
        )
        self.target_masks = torch.zeros(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.sep_masks = torch.ones(
            (self.total_num_traj, self.max_gen_len), dtype=torch.float32
        )
        self.generator_ready = False

    def gather_buffer(self, accelerator):
        tensor_names = [
            "observations",
            "rewards",
            "returns",
            "values",
            "log_probs",
            "ref_log_probs",
            "advantages",
            "masks",
            "rollin_masks",
            "target_ids",
            "target_masks",
            "sep_masks",
        ]
        for tensor in tensor_names:
            self.__dict__[tensor] = accelerator.gather(
                self.__dict__[tensor].to(accelerator.device)
            ).cpu()

    def batch_add(
        self,
        obs,
        value,
        log_prob,
        ref_log_prob,
        reward,
        mask,
        target_id,
        target_mask,
        rollin_mask=None,
    ) -> None:
        pointer = min((self.pos + 1) * self.num_traj_per_sample, self.total_num_traj)
        start = pointer - self.num_traj_per_sample

        # Base Values
        self.observations[start:pointer] = obs
        self.values[start:pointer] = value
        self.log_probs[start:pointer] = log_prob
        self.ref_log_probs[start:pointer] = ref_log_prob
        self.rewards[start:pointer] = reward
        self.masks[start:pointer] = mask

        # Dataset References
        self.target_ids[start:pointer] = target_id
        self.target_masks[start:pointer] = target_mask

        # Rollin Masks
        if rollin_mask is not None:
            self.rollin_masks[start:pointer] = ~rollin_mask

        # Seperator Masks
        if self.sep_token_id is not None:
            curr_token = obs[:, -self.max_gen_len - 1 : -1]
            sep_mask = (curr_token == self.sep_token_id).float()
            self.sep_masks[start:pointer] = sep_mask

        # Buffer Pointer update
        self.pos += 1
        if pointer == self.total_num_traj:
            self.full = True

    def compute_returns_and_advantage(self) -> None:
        last_gae_lam = 0.0
        values = self.values * self.masks
        rewards = self.rewards * self.masks
        sep_masks = self.sep_masks * self.masks
        cumsum = torch.zeros_like(rewards[:, -1])
        next_value = torch.zeros_like(values[:, -1])
        for step in reversed(range(self.max_gen_len)):
            sep_mask = sep_masks[:, step]
            cumsum += rewards[:, step]
            delta = cumsum + self.gamma * next_value - values[:, step]
            new_last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
            last_gae_lam = new_last_gae_lam * sep_mask + last_gae_lam * (1 - sep_mask)
            self.advantages[:, step] = last_gae_lam
            cumsum *= 1 - sep_mask
            next_value = values[:, step] * sep_mask + next_value * (1 - sep_mask)
        self.returns = self.advantages + values
        if self.use_whitening:
            mean = (self.advantages * sep_masks).sum() / sep_masks.sum()
            mean_sq = (self.advantages.square() * sep_masks).sum() / sep_masks.sum()
            var = mean_sq - mean.square()
            self.advantages = (self.advantages - mean) * (var + 1e-8).rsqrt()

    def __len__(self):
        # Note: Dataloader uses Dataset length to determine iterations. Mulitply length by num processes # noqa
        # to account for the dataset POST gather across all buffers
        return self.total_num_traj * self.num_processes

    def __getitem__(self, idx):
        # Note pretty sure squeeze is not necesssary
        return Sample(
            observations=self.observations[idx],
            old_values=self.values[idx],
            old_log_prob=self.log_probs[idx],
            ref_log_prob=self.ref_log_probs[idx],
            advantages=self.advantages[idx],
            returns=self.returns[idx],
            masks=self.masks[idx],
            rollin_masks=self.rollin_masks[idx],
            target_ids=self.target_ids[idx],
            target_masks=self.target_masks[idx],
            sep_masks=self.sep_masks[idx],
        )

    def create_dataloader(self, batch_size, shuffle):
        def stack_samples(elems):
            # If dictionary stack along keys
            if isinstance(elems[0], dict):
                return {
                    k: torch.stack([elem[k] for elem in elems]) for k in elems[0].keys()
                }
            # Otherwise stack tensors
            return torch.stack(elems)

        # Place inside batch NamedTuple
        def batch_collator(samples):
            return Batch(*map(stack_samples, zip(*samples)))

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=batch_collator)
