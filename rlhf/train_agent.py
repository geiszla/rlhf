"""Module for training an RL agent."""

import sys
import typing
from os import path

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
    ENVIRONMENT_NAME,
    USE_SDE,
    checkpoints_path,
    cpu_count,
    get_reward_model_name,
    script_path,
)
from .networks import LightningNetwork
from .types import FeedbackType

# Uncomment line below to use PyPlot with VSCode Tunnels
matplotlib.use("agg")

IS_EXPERT_REWARD = False
tensorboard_path = path.join(script_path, "..", "rl_logs")

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

RUN_NAME = get_reward_model_name("-".join(sys.argv[1:]), feedback_override="without")

output_path = path.join(checkpoints_path, RUN_NAME)

random_generator = numpy.random.default_rng(0)

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
                rewards = torch.min(
                    torch.cat(
                        expert_model.policy.critic_target(
                            torch.from_numpy(state).to(DEVICE),
                            torch.from_numpy(actions).to(DEVICE),
                        ),
                        dim=1,
                    ),
                    dim=1,
                )[0]
            else:
                rewards = torch.zeros(state.shape[0]).to(DEVICE)

                for reward_model in self.reward_models:
                    rewards += reward_model(
                        torch.cat(
                            [
                                torch.Tensor(state).to(DEVICE),
                                torch.Tensor(actions).to(DEVICE),
                            ],
                            dim=1,
                        )
                    ).squeeze(1)

                rewards /= len(self.reward_models)

        # if self.counter > 0:
        #     with torch.no_grad():
        #         expert_rewards = torch.min(
        #             torch.cat(
        #                 expert_model.policy.critic_target(
        #                     torch.from_numpy(state).to(DEVICE),
        #                     torch.from_numpy(actions).to(DEVICE),
        #                 ),
        #                 dim=1,
        #             ),
        #             dim=1,
        #         )[0]

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
            tb_log_name=RUN_NAME,
        )

        timesteps = trained_model.num_timesteps
        print(f"{timesteps}/{steps_per_iteration * iterations} steps done")

        model.save(f"{output_path}_{timesteps}")


if __name__ == "__main__":
    main()
