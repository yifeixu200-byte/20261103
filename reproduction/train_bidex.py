"""Experimental reconstruction, not an author-released Bi-DexHands entry point."""
import argparse
import ast
import csv
import json
import os
from pathlib import Path
import sys

TASKS = ('ShadowHandPen', 'ShadowHandDoorOpenOutward',
         'ShadowHandDoorOpenInward', 'ShadowHandBottleCap')


def make_env(root, task_name, seed, device, headless):
    # Isaac Gym MUST precede torch imports.
    import isaacgym  # noqa: F401
    sys.path.insert(0, str(root))
    from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_seed
    from bidexhands.utils.parse_task import parse_task
    argv, cwd = sys.argv, Path.cwd()
    try:
        config_root = root if (root / 'cfg').is_dir() else root / 'bidexhands'
        if not (config_root / 'cfg').is_dir():
            raise FileNotFoundError('Cannot find upstream cfg directory; complete the DexterousHands checkout first.')
        os.chdir(config_root)
        sys.argv = ['bidex', '--task', task_name, '--algo', 'mappo',
                    '--task_type', 'MultiAgent', '--num_envs', '1',
                    '--seed', str(seed), '--rl_device', device,
                    '--sim_device', device, '--pipeline', 'gpu' if device.startswith('cuda') else 'cpu']
        if headless:
            sys.argv.append('--headless')
        args = get_args()
        cfg, cfg_train, _ = load_cfg(args)
        set_seed(seed)
        hand = ast.literal_eval(cfg['env']['handAgentIndex'])
        if hand != [[0, 1, 2, 3, 4, 5]]:
            raise ValueError('Paper reconstruction requires two whole-hand agents.')
        cfg['env']['observationType'] = 'full_state'
        params = parse_sim_params(args, cfg, cfg_train)
        _, env = parse_task(args, cfg, cfg_train, params, [hand, hand])
        return env, cfg
    finally:
        sys.argv = argv
        os.chdir(cwd)


def make_configs(adapter, args, run_dir):
    from configs.dreamer.mamujoco.mamujocoLearnerConfig import MAMujocoDreamerLearnerConfig
    from configs.dreamer.mamujoco.mamujocoControllerConfig import MAMujocoDreamerControllerConfig
    from environments import Env
    learner = MAMujocoDreamerLearnerConfig()
    controller = MAMujocoDreamerControllerConfig()
    for cfg in (learner, controller):
        # Internal continuous-action branch reuse; this does not instantiate MuJoCo.
        cfg.ENV_TYPE = Env.MAMUJOCO
        cfg.IN_DIM = adapter.n_obs
        cfg.STATE_DIM = adapter.state_dim
        cfg.ACTION_SIZE = adapter.n_actions
        cfg.NUM_AGENTS = adapter.n_agents
        cfg.CONTINUOUS_ACTION = True
        cfg.ACTION_SPACE = adapter.individual_action_space
        cfg.policy_class = args.policy
        cfg.vq_type = 'fsq'
        cfg.critic_state_only = True
        cfg.state_decoder_type = 's + id'
        cfg.GAMMA = .95
    learner.ENTROPY = .01
    learner.N_SAMPLES = 500
    learner.EPOCHS = learner.ac_steps_first_epoch = 4
    learner.fsq_reconstruction_loss = 'mse'
    learner.critic_loss = 'paper_mse'
    learner.use_valuenorm = False
    learner.use_ce_for_cont = learner.compute_end_in_TD = True
    learner.diffusion_sampler_cfg.num_steps_denoising = 4
    learner.DEVICE = args.device
    learner.seed = args.seed
    learner.env_name = learner.map_name = args.task
    learner.RUN_DIR = str(run_dir)
    learner.LOG_FOLDER = str(run_dir / 'logs')
    learner.load_pretrained = False
    learner.load_path = None
    learner.sample_temperature = 20.
    return learner, controller


def append_csv(path, row):
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bidex-root', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=300000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--policy', choices=['gaussian', 'beta'], default='gaussian',
                        help='Gaussian is an explicit reconstruction choice, not verified author BiDex config.')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--eval-interval', type=int, default=5000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    args = parser.parse_args()
    if args.steps < 2 or args.eval_interval < 1 or args.eval_episodes < 1:
        parser.error('steps >= 2 and positive evaluation settings required')
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'ckpt').mkdir()
    env, env_cfg = make_env(args.bidex_root.resolve(), args.task, args.seed, args.device, not args.render)
    from bidex_adapter import BiDexAdapter, collect_episode
    import wandb
    adapter = BiDexAdapter(env)
    lc, cc = make_configs(adapter, args, run_dir)
    (run_dir / 'configuration.json').write_text(json.dumps({
        'status': 'reconstructed; not simulator-validated',
        'arguments': vars(args), 'environment': env_cfg, 'learner': lc.to_dict(),
    }, default=str, indent=2), encoding='utf-8')
    wandb.init(project='dima-bidex-reconstruction', mode='disabled')
    learner, controller = lc.create_learner(), cc.create_controller()
    training_calls = training_data = evaluation_calls = 0
    last_eval = 0
    while training_calls < args.steps - 1:
        controller.receive_params(learner.params())
        before = adapter.simulator_steps
        rollout, count = collect_episode(adapter, controller, args.steps - training_calls - 1)
        training_calls += adapter.simulator_steps - before
        training_data += count
        learner.step(rollout)
        append_csv(run_dir / 'training.csv', dict(
            simulator_steps=training_calls, policy_transitions=training_data,
            seed=args.seed, episode_return=float(rollout['reward'].sum(0).mean()),
            episode_length=count))
        print(f'train simulator calls={training_calls}, replay transitions={training_data}')
        if training_calls - last_eval >= args.eval_interval or training_calls >= args.steps - 1:
            controller.receive_params(learner.params())
            controller.use_deterministic_action = True
            returns = []
            before = adapter.simulator_steps
            for _ in range(args.eval_episodes):
                evaluation, _ = collect_episode(adapter, controller)
                returns.append(float(evaluation['reward'].sum(0).mean()))
            evaluation_calls += adapter.simulator_steps - before
            controller.use_deterministic_action = False
            append_csv(run_dir / 'evaluations.csv', dict(
                steps=training_calls, policy_transitions=training_data, seed=args.seed,
                eval_returns=sum(returns)/len(returns), eval_episodes=len(returns),
                evaluation_simulator_steps=evaluation_calls))
            learner.save(str(run_dir / 'ckpt' / f'model_{training_calls}.pth'))
            last_eval = training_calls
    learner.save(str(run_dir / 'ckpt' / 'final.pth'))
    wandb.finish()


if __name__ == '__main__':
    main()
