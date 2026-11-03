"""Reconstructed adapter for PKU-MARL/DexterousHands MultiVecTaskPython.

This is NOT code released by DIMA's authors. Imports only torch so its tensor
contract can be tested without installing Isaac Gym. One environment, two hands.
"""
import torch


class BiDexAdapter:
    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError('Use num_envs=1: never flatten vector environments into a trajectory.')
        self.env = env
        self.n_agents = env.num_agents
        shapes = [space.shape for space in env.action_space]
        if self.n_agents != 2 or len(set(shapes)) != 1:
            raise ValueError('Only two whole-hand agents with equal action dimensions are supported.')
        self.n_actions = shapes[0][0]
        self.n_obs = env.observation_space[0].shape[0]
        self.state_dim = env.share_observation_space[0].shape[0]
        self.individual_action_space = env.action_space[0]
        self.policy_steps = 0
        self.reset_steps = 0
        self.needs_reset = True

    def _copy(self, value):
        return value.detach().to(device='cpu', dtype=torch.float32).clone()

    def _validate(self, obs, state):
        if tuple(obs.shape) != (1, self.n_agents, self.n_obs):
            raise ValueError(f'Unexpected local observation shape: {obs.shape}')
        if tuple(state.shape) != (1, self.n_agents, self.state_dim):
            raise ValueError(f'Unexpected global state shape: {state.shape}')
        if not torch.equal(state[:, 0], state[:, 1]):
            raise ValueError('Global state must be the same for both agents, not concatenated local observations.')

    def reset(self):
        # Upstream reset() advances simulation; it does NOT request a reset.
        # pre_physics_step handles reset_buf before applying the warm-up action.
        self.env.task.reset_buf.fill_(1)
        obs, state, _ = self.env.reset()
        self.reset_steps += 1
        self._validate(obs, state)
        self.needs_reset = False
        return self._copy(obs), self._copy(state)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError('Reset before stepping; terminal states may not cross episode boundaries.')
        if tuple(action.shape) != (self.n_agents, self.n_actions):
            raise ValueError(f'Expected joint action [{self.n_agents},{self.n_actions}].')
        # Store precisely the clipped action actually sent to the physics task.
        action = action.clamp(-self.env.clip_actions, self.env.clip_actions)
        action_list = [a.unsqueeze(0).to(self.env.rl_device) for a in action]
        obs, state, reward, done, info, _ = self.env.step(action_list)
        self.policy_steps += 1
        self._validate(obs, state)
        done = done.reshape(1, self.n_agents, 1)
        if not torch.equal(done[:, 0], done[:, 1]):
            raise ValueError('Asynchronous agent terminations are unsupported.')
        self.needs_reset = bool(done.all())
        # Copies are essential: Isaac Gym reuses these buffers on reset/step.
        return tuple(self._copy(x) for x in (obs, state, reward, done)), self._copy(action), info

    @property
    def simulator_steps(self):
        return self.policy_steps + self.reset_steps


def collect_episode(adapter, controller, max_steps=None):
    """DIMA rollout contract; done=true only for simulator termination.

    Budget truncation is represented by last=1, done=0. No absorbing padding.
    Reset warm-up calls are counted by the adapter, but cannot be used as data
    because upstream does not expose their pre-reset state/action pair.
    """
    if max_steps is not None and max_steps < 1:
        raise ValueError('max_steps must be positive')
    controller.init_rnns()
    controller.init_buffer()
    obs, state = adapter.reset()
    steps = 0
    while True:
        actions, entropy = controller.step(obs, state, None, None)
        (next_obs, next_state, reward, done), applied, _ = adapter.step(actions)
        controller.update_buffer(dict(
            action=applied.unsqueeze(0), observation=obs, shared_obs=state,
            next_shared_obs=next_state, reward=reward, done=done,
            fake=torch.zeros_like(done), entropy=entropy.unsqueeze(0)))
        steps += 1
        obs, state = next_obs, next_state
        if done.all() or (max_steps is not None and steps >= max_steps):
            break
    return controller.dispatch_buffer(), steps
