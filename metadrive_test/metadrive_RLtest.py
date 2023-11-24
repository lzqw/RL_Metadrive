import torch
import argparse
import numpy as np
import os
from metadrive import TopDownMetaDrive,TopDownMetaDriveEnvV2,MetaDriveEnv
import random
from result_plotter.logger import SafeLogger
from RL_algo import TD3,utils


def eval_policy(policy, eval_env, eval_episodes=1):
    _rewards = [0.] * eval_episodes
    _costs = [0.] * eval_episodes

    for ep_num in range(eval_episodes):
        state, done = eval_env.reset(), False
        state=state[0]
        while not done:
            action = policy.select_action(np.array(state))
            print(action)
            state, reward, done, info,_ = eval_env.step(action)
        # _rewards[ep_num] = info['route_completion']
        _rewards[ep_num] = reward
        _costs[ep_num] = float(eval_env.vehicle.crash_vehicle)
    avg_reward_normal = sum(_rewards) / eval_episodes
    avg_cost_normal = sum(_costs) / eval_episodes

    _rewards = [0.] * eval_episodes
    _costs = [0.] * eval_episodes
    avg_reward_adv = sum(_rewards) / eval_episodes
    avg_cost_adv = sum(_costs) / eval_episodes

    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: Reward_normal {avg_reward_normal:.3f} Cost_normal {avg_cost_normal: .3f} Reward_adv {avg_reward_adv:.3f} Cost_adv {avg_cost_adv: .3f}")
    print("---------------------------------------")
    return avg_reward_normal, avg_cost_normal, avg_reward_adv, avg_cost_adv


if __name__ == '__main__':
    parser  = argparse.ArgumentParser ()
    parser.add_argument("--env", default="MDWaymo")
    parser.add_argument("--seed", default=1,    type=int)
    parser.add_argument("--start_timesteps", default=10000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=25000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="default")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--batch_size", default=1024, type=int)  # Batch size for both actor and critic
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    args = parser.parse_args()
    file_name = 'args.mode'
    logger = SafeLogger(exp_name='test', env_name=args.env, seed=args.seed,
                        fieldnames=['route_completion_normal', 'crash_rate_normal', 'route_completion_adv',
                                    'crash_rate_adv'])
    config_test =dict(
            map="SSSSSSSS",
            traffic_density=0.1,
            num_scenarios=400,
            use_render=True,
            start_seed=random.randint(0, 1000),
            # manual_control= True,
            show_coordinates=True,
            # show_policy_mark=True,
            image_observation=False,
            vehicle_config=dict(
                lidar=dict(
                    add_others_navi=False,
                    num_others=4,
                    distance=50,
                    num_lasers=30,
                ),
                side_detector=dict(num_lasers=30),
                lane_line_detector=dict(num_lasers=12),
            ))
    env_test = MetaDriveEnv(
        config_test
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env_test.observation_space.shape[0]
    action_dim = env_test.action_space.shape[0]
    max_action = float(env_test.action_space.high[0])
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")


    state, done = env_test.reset(), False
    state=state[0]
    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0

    last_eval_step = 0

    evalRC_normal, evalCrash_normal, evalRC_adv, evalCrash_adv = eval_policy(policy, env_test)
    env_test.close()