import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Type, Union
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from minigrid.wrappers import FlatObsWrapper 
from procgen import ProcgenGym3Env
from rl_zoo3.wrappers import Gym3ToGymnasium
import ale_py
import minigrid
minigrid.register_minigrid_envs()
import numpy as np
import torch
from captum.attr import IntegratedGradients
from numpy.typing import NDArray
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecExtractDictObs
from stable_baselines3.common.atari_wrappers import AtariWrapper
from torch import Tensor
import random
import itertools

from rlhf.datatypes import FeedbackDataset
from rlhf.save_reset_wrapper import SaveResetEnvWrapper

def get_attributions(
    observation: Tensor,
    actions: NDArray,
    explainer: IntegratedGradients,
    algorithm: str = "sac",
    device: str = "cuda",
) -> NDArray[np.float64]:
    """
    Compute attributions for a given observation and actions using the provided explainer.
    """
    obs_baselines = torch.zeros_like(observation)
    if algorithm == "sac":
        actions_tensor = torch.from_numpy(actions).unsqueeze(0).to(device)
        actions_baselines = torch.zeros_like(actions_tensor)
    
        attributions = explainer.attribute(
            (observation, actions_tensor),
            target=0,
            baselines=(obs_baselines, actions_baselines),
            internal_batch_size=64,
        )
    else:
        attributions = explainer.attribute(
            observation,
            target=0,
            baselines=obs_baselines,
            internal_batch_size=64,
        )        

    return attributions.squeeze().cpu().numpy()

def predict_expert_value(
    expert_model: Union[PPO, SAC], 
    observation: np.ndarray, 
    actions: Tensor = None
) -> Tensor:
    """Return the value from the expert's value function for a given observation and actions."""

    observation = expert_model.policy.obs_to_tensor(observation)[0]
    with torch.no_grad():
        return torch.min(
            torch.cat(expert_model.policy.critic_target(observation, actions), dim=1) if isinstance(expert_model, SAC) else expert_model.policy.predict_values(observation),
            dim=1,
            keepdim=True,
        )[0]

def get_model_logits(
    expert_model: Union[PPO, SAC],
    observation: Tensor,
    actions: Tensor = None,
) -> Tensor:
    if isinstance(expert_model, SAC):
        return torch.min(
            torch.cat(expert_model.policy.critic_target(observation, actions), dim=1) if isinstance(expert_model, SAC) else expert_model.policy.predict_values(observation),
            dim=1,
            keepdim=True,
        )[0]
    else:
        return expert_model.policy.predict_values(observation)

def create_segments(arr, start_indices, segment_length):
    segments = []
    for start in start_indices:
        segment = arr[start:start + segment_length]
        segments.append(segment)
    return segments

def discounted_sum_numpy(rewards, discount_factor):
    rewards = np.array(rewards)
    n = len(rewards)
    discount_factors = discount_factor ** np.arange(n)
    return np.sum(rewards * discount_factors)

def equal_depth_binning_with_indices(data, num_bins):
    # Sort the data and get the original indices
    sorted_indices = np.argsort(data)
    sorted_data = np.sort(data)
    
    # Determine the number of elements per bin
    bin_size = len(data) // num_bins
    remainder = len(data) % num_bins
    
    bins = []
    bin_indices = np.zeros(len(data), dtype=int)
    start = 0
    
    for i in range(num_bins):
        end = start + bin_size + (1 if i < remainder else 0)
        bin_indices[sorted_indices[start:end]] = i
        bins.append(sorted_data[start:end])
        start = end
    
    return bin_indices, bins

def get_k_random_pairs(data, k):
    all_pairs = list(itertools.combinations(data, 2))
    if k > len(all_pairs):
        raise ValueError("k is too large for the number of possible unique pairs")
    return random.sample(all_pairs, k)

def generate_feedback(
    model_class: Type[Union[PPO, SAC]],
    expert_model: Union[PPO, SAC],
    environment: gym.Env,
    environment_name: str = "HalfCheetah-v3",
    checkpoints_path: str = "rl_checkpoints",
    total_steps: int = 10000,
    n_feedback: int = 100,
    segment_len: int = 50,
    algorithm: str = "sac",
    device: str = "cuda",
) -> FeedbackDataset:
    """Generate agent's observations and feedback in the training environment."""
    feedback = []
    feedback_id = f"{algorithm}_{environment_name.replace("/", "-")}"
    checkpoints_dir = os.path.join(checkpoints_path, algorithm, f"{environment_name.replace("/", "-")}_1")

    print(f"Generating feedback for: {feedback_id}")

    checkpoint_files = [
        file for file in os.listdir(checkpoints_dir) if re.search(r"rl_model_.*\.zip", file)
    ] or [f"{environment_name}.zip"]

    num_checkpoints = len(checkpoint_files)
    steps_per_checkpoint = total_steps // num_checkpoints
    feedback_per_checkpoint = n_feedback // num_checkpoints

    gamma = expert_model.gamma

    explainer_cls = IntegratedGradients

    for model_file in checkpoint_files:
        
        # we already sample the indices for the number of generated feedback instances/segments
        fb_indices = random.choices(range(0, steps_per_checkpoint + 1), k=feedback_per_checkpoint)
        
        model = model_class.load(
            os.path.join(checkpoints_dir, model_file),
            custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
        )

        observation, _ = environment.reset()
        state_copies = []

        # now collect original data
        
        feedback = []
        for step in range(steps_per_checkpoint):
            if step in fb_indices:
                state_copies.append(environment.save_state(observation=observation))

            actions, _ = model.predict(observation, deterministic=True)
            next_observation, reward, terminated, _, _ = environment.step(actions)

            feedback.append(
                (observation, actions, reward, terminated)
            )

            observation = next_observation if not terminated else environment.reset()[0]


        # generate feedback from collected examples, split at given indices and dones
        final_segment_indices = fb_indices #set(fb_indices) | set(np.where([f[3] for f in feedback] == True)[0])
        segments = create_segments(feedback, final_segment_indices, segment_len)

        # start by computing the evaluative fb. (for the comparative one, we just used samples segment pairs)
        opt_gaps = []
        for seg in segments:
            # predict the initial value
            initial_val = predict_expert_value(
                    expert_model, np.array(seg[0][0])
                ).item()

            # sum the discounted rewards
            discounted_rew_sum = discounted_sum_numpy([s[2] for s in seg], gamma)

            # get the final value
            final_val = predict_expert_value(
                    expert_model, np.array(seg[-1][0])).item()

            opt_gap = initial_val - (discounted_rew_sum + gamma ** len(seg) * final_val)
            opt_gaps.append(opt_gap)

        # compute a histogram for rating feedback
        counts, bin_edges = np.histogram(opt_gaps, bins=10)

        plt.hist(opt_gaps, bins=10)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.savefig("debug_hist.png")

        # bin indices, which we interpret as rating feedback
        ratings = equal_depth_binning_with_indices(opt_gaps, 10)[0]

        # generate pair preferences, with tolerance 1.0
        tolerance = 1.0
        pairs = get_k_random_pairs(np.arange(len(segments)), n_feedback)
        preferences = [(a,b,-1) if (opt_gaps[a] - opt_gaps[b] > tolerance) else (a,b,1) if (opt_gaps[b] - opt_gaps[a] > tolerance) else (a,b,0) for a, b in pairs]
        
        # instructive feedback, reset env and run expert model for demos for each segment
        demos = []
        corrections = []
        for i, state in enumerate(state_copies):
            _, _ = environment.reset()
            obs = environment.load_state(state)

            demo = []
            for _ in range(segment_len):
                action, _ = expert_model.predict(obs, deterministic=True)
                obs, rew, terminated, _, _ = environment.step(action)
                demo.append((obs, action, rew, terminated))

                if terminated:
                    break
            demos.append(demo)
            corrections.append((segments[i], demo))

        # descriptive feedback, for now, compute attributions and the average over features
        if algorithm == "sac":
            explainer = explainer_cls(lambda obs, acts: get_model_logits(expert_model, obs, acts))
        else:
            explainer = explainer_cls(lambda obs: get_model_logits(expert_model, obs))


        descriptions = []
        for i, seg in enumerate(segments):
            attributions = get_attributions(observation = expert_model.policy.obs_to_tensor(np.array([s[0] for s in seg]))[0], actions=None, explainer=explainer, algorithm=algorithm)
            saliency = np.std(attributions) / np.mean(attributions) * np.abs(np.mean(attributions))
            descriptions.append((saliency, opt_gaps[i]))

        
        descr_preferences = [(a,b,-1) if (descriptions[a][1] - descriptions[b][1] > tolerance) else (a,b,1) if (descriptions[b][1] - descriptions[a][1] > tolerance) else (a,b,0) for a, b in pairs]        

    return {
        "segments": segments,
        "ratings": ratings,
        "preferences": preferences,
        "demos": demos,
        "corrections": corrections,
        "description": descriptions,
        "description_preference": descr_preferences,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=0, help="Experiment number")
    parser.add_argument("--algorithm", type=str, default="sac", help="RL algorithm")
    parser.add_argument("--environment", type=str, default="HalfCheetah-v5", help="Environment")
    parser.add_argument("--n-steps", type=int, default=int(1e4), help="Number of steps to generate feedback for")
    parser.add_argument("--n-feedback", type=int, default=int(1000), help="How many feedback instances should be generated")
    parser.add_argument("--seed", type=int, default=1337, help="TODO: Seed for env and stuff")
    parser.add_argument("--segment-len", type=int, default=50, help="How long is the segment we generate feedback for")
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feedback_id = f"{args.algorithm}_{args.environment}"
    feedback_path = Path(__file__).parents[1].resolve() / "feedback" / f"{feedback_id}.pkl"
    checkpoints_path = "../main/logs"

    expert_model = (PPO if args.algorithm == "ppo" else SAC).load(
        os.path.join(checkpoints_path, args.algorithm, f"{args.environment.replace("/", "-")}_1", f"{args.environment.replace("/", "-")}.zip"),
        # is this really necessary?
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0},
    )

    if "procgen" in args.environment:
        _, short_name, _ = args.environment.split("-")
        environment = Gym3ToGymnasium(ProcgenGym3Env(num=1, env_name=short_name))
        environment = SaveResetEnvWrapper(TransformObservation(environment, lambda obs: obs["rgb"], environment.observation_space))
    elif "ALE/" in args.environment:
        environment = FrameStackObservation(AtariWrapper(gym.make(args.environment)), 4)
        environment = SaveResetEnvWrapper(TransformObservation(environment, lambda obs: obs.squeeze(-1), environment.observation_space))
    elif "MiniGrid" in args.environment:
        environment = SaveResetEnvWrapper(FlatObsWrapper(gym.make(args.environment)))
    else:
        environment = SaveResetEnvWrapper(gym.make(args.environment))
    
    model_class = PPO if args.algorithm == "ppo" else SAC

    feedback = generate_feedback(
        model_class,
        expert_model,
        environment,
        environment_name=args.environment,
        total_steps=args.n_steps,
        n_feedback=args.n_feedback,
        segment_len=args.segment_len,
        checkpoints_path=checkpoints_path,
        algorithm=args.algorithm,
        device=device,
    )

    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "wb") as feedback_file:
        pickle.dump(feedback, feedback_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
