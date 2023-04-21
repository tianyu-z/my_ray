##########
# Contribution by the Center on Long-Term Risk:
# https://github.com/longtermrisk/marltoolbox
##########
import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.pg import PG
from ray.rllib.algorithms.ppo import PPO

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument("--stop-iters", type=int, default=10)
parser.add_argument("--algo", type=str, default="PPO", help="PG or PPO")
parser.add_argument("--ismessaged", type=int, default=1, help="0 or 1")


from ray.rllib.algorithms.callbacks import DefaultCallbacks
import csv


class CustomLoggingCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.episode_number = 0

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_sub_environments()[0]
        info = env._get_episode_info()
        self.episode_number += 1
        for k, v in info.items():
            episode.custom_metrics[k] = v
        # Save custom metrics to a CSV file
        episode_id = episode.episode_id
        custom_metrics = episode.custom_metrics

        with open("custom_metrics.csv", mode="a") as csv_file:
            fieldnames = ["episode_number", "episode_id"] + list(custom_metrics.keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if csv_file.tell() == 0:
                writer.writeheader()

            row = {"episode_number": self.episode_number, "episode_id": episode_id}
            row.update(custom_metrics)
            writer.writerow(row)


def main(debug, args):
    # stop_iters, framework, algo_name = args.stop_iters, args.framework, args.algo
    algo_name = args.algo
    algo = globals()[algo_name]
    train_n_replicates = 1 if debug else 1
    seeds = list(range(train_n_replicates))

    ray.init(num_cpus=os.cpu_count(), num_gpus=1, local_mode=debug)

    rllib_config, stop_config = get_rllib_config(seeds, debug, args)
    tuner = tune.Tuner(
        algo,
        param_space=rllib_config,
        run_config=air.RunConfig(
            name=f"{algo_name}_IPD",
            stop=stop_config,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=0,
                checkpoint_at_end=True,
            ),
        ),
    )
    tuner.fit()
    ray.shutdown()


def get_rllib_config(seeds, debug=False, args=None):
    assert args is not None
    stop_config = {
        "training_iteration": 2 if debug else args.stop_iters,
    }

    if args.ismessaged:
        from ray.rllib.examples.env.matrix_sequential_social_dilemma_messaged import (
            MessageIteratedPrisonersDilemma as IPD,
        )
    else:
        from ray.rllib.examples.env.matrix_sequential_social_dilemma import (
            IteratedPrisonersDilemma as IPD,
        )

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": 20,
        "get_additional_info": True,
    }

    rllib_config = {
        "env": IPD,
        "seed": 42,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    None,
                    IPD.OBSERVATION_SPACE,
                    IPD.ACTION_SPACE,
                    {},
                ),
                env_config["players_ids"][1]: (
                    None,
                    IPD.OBSERVATION_SPACE,
                    IPD.ACTION_SPACE,
                    {},
                ),
            },
            "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: agent_id,
        },
        "seed": tune.grid_search(seeds),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": args.framework,
        # Add buffer settings for the PPO algorithm
        "rollout_fragment_length": 200,
        "train_batch_size": 4000,
        "callbacks": CustomLoggingCallback,
    }

    return rllib_config, stop_config


if __name__ == "__main__":
    debug_mode = False
    args = parser.parse_args()
    main(debug_mode, args)
