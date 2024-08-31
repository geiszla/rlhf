"""Module for training an RL agent."""

import sys
import typing
from os import path
from typing import Union

import matplotlib
import numpy
import torch
from imitation.rewards.reward_function import RewardFn
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from .common import (
    ALGORITHM,
    DEVICE,
    ENSEMBLE_COUNT,
    ENVIRONMENT_NAME,
    USE_SDE,
    checkpoints_path,
    cpu_count,
    get_reward_model_name,
    script_path,
)
from .networks import LightningNetwork
from .types import FeedbackType

SEEDS = [1337, 42, 69, 420]
SEED = SEEDS[0]

# Uncomment line below to use PyPlot with VSCode Tunnels
matplotlib.use("agg")

IS_EXPERT_REWARD = False

if len(sys.argv) < 2:
    raise ValueError("Give the reward model suffixes as the arguments to the program.")

# Set feedback type to None to not use the custom reward model

reward_model_paths: list[str] = []

for arg in sys.argv[1:]:
    feedback_type, suffix = arg.split("-")

    REWARD_MODEL_ID = get_reward_model_name(
        suffix,
        feedback_override=(
            "without" if IS_EXPERT_REWARD else typing.cast(FeedbackType, feedback_type)
        ),
    )

    reward_model_paths.append(
        path.join(
            script_path, "..", "reward_model_checkpoints", f"{REWARD_MODEL_ID}.ckpt"
        )
    )

RUN_NAME = get_reward_model_name(
    f"{'-'.join(sys.argv[1:])}", feedback_override="without"
)

tensorboard_path = path.join(script_path, "..", "rl_logs", RUN_NAME)
output_path = path.join(checkpoints_path, RUN_NAME)

random_generator = numpy.random.default_rng(SEED)

# if TRAINING_FEEDBACK_TYPE == "expert":
# PPO
# expert_model = PPO.load(
#     path.join(
#         script_path,
#         "..",
#         "experts",
#         "ppo",
#         "HalfCheetah-v3_1",
#         "HalfCheetah-v3.zip",
#     ),
#     custom_objects={
#         "learning_rate": 0.0,
#         "lr_schedule": lambda _: 0.0,
#         "clip_range": lambda _: 0.0,
#     },
# )

# SAC
expert_model = SAC.load(
    path.join(
        script_path,
        "..",
        "experts",
        "sac",
        "HalfCheetah-v3_1",
        "HalfCheetah-v3.zip",
    ),
    custom_objects={
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    },
)


class CustomReward(RewardFn):
    """Custom reward based on fine-tuned reward model."""

    def __init__(self):
        """Initialize custom reward."""
        super().__init__()

        self.reward_models: list[LightningNetwork] = []

        if not IS_EXPERT_REWARD:
            for reward_model_path in reward_model_paths:
                # pylint: disable=no-value-for-parameter
                self.reward_models.append(
                    LightningNetwork.load_from_checkpoint(
                        checkpoint_path=reward_model_path
                    )
                )

        self.rewards = []
        self.expert_rewards = []
        self.counter = 0

        # Variables for calculating a running mean
        self.reward_mean: Union[torch.Tensor, None] = None
        self.squared_distance_from_mean: Union[torch.Tensor, None] = None

    def standardize_rewards(self, rewards: torch.Tensor):
        """
        Standradizes the input using the rolling mean and standard deviation of the rewards.

        Input should be a tensor of shape (batch_size, model_count).
        """
        model_count = len(self.reward_models)

        if self.reward_mean is None:
            self.reward_mean = torch.zeros(model_count).to(DEVICE)

        if self.squared_distance_from_mean is None:
            self.squared_distance_from_mean = torch.zeros(model_count).to(DEVICE)

        standard_deviation = torch.ones(model_count).to(DEVICE)

        for reward_index, reward in enumerate(rewards):
            # Welford's algorithm for calculating running mean and variance
            self.counter += 1

            difference = reward - self.reward_mean
            self.reward_mean += difference / self.counter
            new_difference = reward - self.reward_mean
            self.squared_distance_from_mean += difference * new_difference

            if self.counter > 1:
                standard_deviation = (
                    self.squared_distance_from_mean / (self.counter - 1)
                ).sqrt()

            rewards[reward_index] = (reward - self.reward_mean) / standard_deviation

        # Alternatively standardize rewards using the mean and standard deviation of
        # the entire batch (usually performs worse)
        # rewards = (rewards - self.reward_mean) / standard_deviation

        return rewards

    def __call__(
        self,
        state: numpy.ndarray,
        actions: numpy.ndarray,
        next_state: numpy.ndarray,
        _done: numpy.ndarray,
    ) -> list:
        """Return reward given the current state."""
        with torch.no_grad():
            if IS_EXPERT_REWARD:
                # PPO
                # rewards = expert_model.policy.value_net(
                #     expert_model.policy.mlp_extractor(
                #         expert_model.policy.extract_features(
                #             torch.from_numpy(next_state).to(DEVICE)
                #         )
                #     )[0]
                # )

                # SAC
                rewards = (
                    expert_model.policy.critic_target(
                        torch.from_numpy(state).to(DEVICE),
                        torch.from_numpy(actions).to(DEVICE),
                    )
                    .cat(dim=1)
                    .min(dim=1)[0]
                )
            elif ENSEMBLE_COUNT > 1:
                model_input = torch.cat(
                    [
                        torch.Tensor(state).to(DEVICE),
                        torch.Tensor(actions).to(DEVICE),
                    ],
                    dim=1,
                )

                rewards = torch.empty(
                    state.shape[0], len(self.reward_models), ENSEMBLE_COUNT
                ).to(DEVICE)

                # Predict reward for all state-action pairs with each model in the ensemble
                for input_index, single_input in enumerate(model_input):
                    batched_input = single_input.tile((ENSEMBLE_COUNT, 1))

                    for model_index, reward_model in enumerate(self.reward_models):
                        rewards[input_index][model_index] = reward_model(
                            batched_input
                        ).squeeze(1)

                # Standardize rewards using rolling mean and standard deviation calculated for
                # a model across state-action pairs and ensemble members
                rewards = self.standardize_rewards(
                    rewards.transpose(1, 2).flatten(start_dim=0, end_dim=1)
                )

                # Reshape rewards back to (batch_size, model_count, ensemble_count)
                rewards = rewards.reshape(
                    (state.shape[0], ENSEMBLE_COUNT, len(self.reward_models)),
                ).transpose(1, 2)

                # Select the least uncertain reward model for each state-action pair
                # least_uncertain_index = rewards.std(dim=2).argmin(dim=1)

                # rewards = rewards[
                #     torch.arange(rewards.size(0)), least_uncertain_index
                # ].mean(dim=1)

                # Alternatively, weight the reward predictions by the inverse of the
                # standard deviation of the models
                inverse_standard_deviations = 1 / rewards.std(dim=2)

                rewards = (inverse_standard_deviations * rewards.mean(dim=2)).sum(
                    dim=1
                ) / inverse_standard_deviations.sum(dim=1)
            else:
                model_input = torch.cat(
                    [
                        torch.Tensor(state).to(DEVICE),
                        torch.Tensor(actions).to(DEVICE),
                    ],
                    dim=1,
                )

                rewards = torch.empty(len(self.reward_models), state.shape[0]).to(
                    DEVICE
                )

                for model_index, reward_model in enumerate(self.reward_models):
                    rewards[model_index] = reward_model(model_input).squeeze(1)

                rewards = (
                    self.standardize_rewards(rewards.transpose(0, 1))
                    .transpose(0, 1)
                    .mean(dim=0)
                )

                # rewards = rewards.mean(dim=0)

        # if self.counter > 0:
        #     with torch.no_grad():
        #         expert_rewards = (
        #             expert_model.policy.critic_target(
        #                 torch.from_numpy(state).to(DEVICE),
        #                 torch.from_numpy(actions).to(DEVICE),
        #             )
        #             .cat(dim=1)
        #             .min(dim=1)[0]
        #         )

        #     self.rewards.append(rewards[0].cpu().numpy())
        #     self.expert_rewards.append(expert_rewards[0].cpu().numpy())

        #     if len(self.rewards) >= 1000:
        #         steps = range(1000)

        #         pyplot.plot(steps, self.rewards, label="Reward model")
        #         pyplot.plot(steps, self.expert_rewards, label="Expert value")
        #         # pyplot.plot(steps, rewards, label="Ground truth rewards")

        #         pyplot.xlabel("Steps")
        #         pyplot.ylabel("Rewards")
        #         pyplot.legend()

        #         pyplot.savefig(
        #             path.join(script_path, "..", "plots", "agent_training_start_output.png")
        #         )

        #         exit()

        # self.counter += 1

        return rewards.cpu().numpy()


def main():
    """Run RL agent training."""
    # For PPO, the more environments there are, the more `num_timesteps` shifts
    # from `total_timesteps` (TODO: check this again)
    environment = make_vec_env(
        ENVIRONMENT_NAME,
        n_envs=cpu_count if ALGORITHM != "ppo" else 1,
        # n_envs=1,
        rng=random_generator,
    )

    print("Run name:", RUN_NAME)

    if not IS_EXPERT_REWARD:
        print("Reward model ID:", REWARD_MODEL_ID)
        environment = RewardVecEnvWrapper(environment, reward_fn=CustomReward())

    print()

    # Select agent algorithm
    if ALGORITHM == "sac":
        model_class = SAC
    elif ALGORITHM == "ppo":
        model_class = PPO
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    model = model_class(
        "MlpPolicy",
        environment,
        # verbose=1,
        tensorboard_log=tensorboard_path,
        use_sde=USE_SDE,
        # gamma=0,
    )

    iterations = 30
    steps_per_iteration = 125000
    timesteps = 0

    # model.save(f"{output_path}_{timesteps}")

    for iteration_count in range(iterations):
        trained_model = model.learn(
            total_timesteps=steps_per_iteration * (iteration_count + 1) - timesteps,
            reset_num_timesteps=False,
            tb_log_name=str(SEED),
        )

        timesteps = trained_model.num_timesteps
        print(f"{timesteps}/{steps_per_iteration * iterations} steps done")

        # model.save(f"{output_path}_{timesteps}")


if __name__ == "__main__":
    main()
