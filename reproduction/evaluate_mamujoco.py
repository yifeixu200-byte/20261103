"""Actor-only real-environment evaluation with optional trajectory export."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from configs.EnvConfigs import MAMujocoConfig
from networks.dreamer.action import StochasticPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--env_name', default='HalfCheetah-v2')
    parser.add_argument('--agent_conf', default='2x3')
    parser.add_argument('--seed', type=int, default=100123)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--policy_class', choices=['gaussian', 'beta'], default='gaussian')
    parser.add_argument('--output', type=Path, default=Path('actor_evaluation'))
    parser.add_argument('--save-trajectories', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Load only locally produced/trusted checkpoints: upstream includes an RMS object.
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    returns = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = MAMujocoConfig(args.env_name, seed, args.agent_conf).create_env()
        try:
            actor = StochasticPolicy(env.n_obs, env.n_actions, 128, 3,
                                     continuous_action=True,
                                     continuous_action_space=env.individual_action_space,
                                     policy_class=args.policy_class).eval()
            actor.load_state_dict(ckpt['actor'])
            observations, shared, _ = env.reset()
            records = {key: [] for key in ['states', 'observations', 'actions', 'rewards', 'dones', 'next_states', 'next_observations']}
            total_return = 0.
            for step in range(env.episode_limit):
                obs = np.stack([observations[i] for i in range(env.n_agents)])
                state = np.stack([shared[i] for i in range(env.n_agents)])
                with torch.no_grad():
                    action, _ = actor(torch.tensor(obs, dtype=torch.float32), deterministic=True)
                nxt, nxt_shared, reward, done, _, _ = env.step(action)
                team_reward = float(np.mean([reward[i] for i in range(env.n_agents)]))
                total_return += team_reward
                for key, value in [('states', state.mean(0)), ('observations', obs),
                                   ('actions', action.numpy()), ('rewards', team_reward),
                                   ('dones', all(bool(done[i]) for i in range(env.n_agents))),
                                   ('next_states', np.stack([nxt_shared[i] for i in range(env.n_agents)]).mean(0)),
                                   ('next_observations', np.stack([nxt[i] for i in range(env.n_agents)]))]:
                    records[key].append(value)
                observations, shared = nxt, nxt_shared
                if all(bool(done[i]) for i in range(env.n_agents)):
                    break
            returns.append(total_return)
            if args.save_trajectories:
                np.savez_compressed(args.output / ('episode_%02d.npz' % episode),
                                    **{key: np.asarray(value) for key, value in records.items()})
        finally:
            env.close()
    report = {'episode_returns': returns, 'mean_return': float(np.mean(returns)),
              'episode_std_ddof1': float(np.std(returns, ddof=1)) if len(returns)>1 else None,
              'note': 'Episode variance within one policy; not across-training-seed variance.'}
    (args.output/'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
