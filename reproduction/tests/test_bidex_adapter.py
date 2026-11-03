"""Contract tests; no claim to test Isaac Gym physics or convergence."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bidex_adapter import BiDexAdapter


class FakeUpstream:
    num_envs = 1
    num_agents = 2
    clip_actions = 1.
    rl_device = 'cpu'
    action_space = [SimpleNamespace(shape=(3,))] * 2
    observation_space = [SimpleNamespace(shape=(4,))] * 2
    share_observation_space = [SimpleNamespace(shape=(6,))] * 2

    def __init__(self):
        self.task = SimpleNamespace(reset_buf=torch.zeros(1))
        self.obs = torch.zeros(1, 2, 4)
        self.state = torch.zeros(1, 2, 6)

    def reset(self):
        assert self.task.reset_buf.item() == 1
        self.task.reset_buf.zero_()
        self.obs.zero_()
        self.state.zero_()
        return self.obs, self.state, None

    def step(self, actions):
        assert len(actions) == 2 and actions[0].shape == (1, 3)
        self.received = torch.cat(actions, dim=1)
        self.obs.fill_(3)
        self.state.fill_(7)
        return self.obs, self.state, torch.ones(1, 2, 1), torch.ones(1, 2), None, None


class AdapterTests(unittest.TestCase):
    def test_joint_actions_terminal_copy_and_transition_count(self):
        env = FakeUpstream()
        adapter = BiDexAdapter(env)
        adapter.reset()
        (obs, state, reward, done), applied, _ = adapter.step(torch.full((2, 3), 2.))
        self.assertTrue(torch.equal(env.received, torch.ones(1, 6)))
        self.assertTrue(torch.equal(applied, torch.ones(2, 3)))
        self.assertEqual(adapter.policy_steps, 1)  # NOT 2 agent rows
        self.assertEqual(adapter.simulator_steps, 2)  # includes reset advance
        with self.assertRaises(RuntimeError):
            adapter.step(torch.zeros(2, 3))
        adapter.reset()
        self.assertTrue(torch.equal(state, torch.full((1, 2, 6), 7.)))
        self.assertEqual(done.shape, (1, 2, 1))

    def test_reject_parallel_envs(self):
        env = FakeUpstream()
        env.num_envs = 8
        with self.assertRaises(ValueError):
            BiDexAdapter(env)

    def test_reject_local_states_disguised_as_global(self):
        env = FakeUpstream()
        adapter = BiDexAdapter(env)
        env.state[:, 1] = 1
        with self.assertRaises(ValueError):
            adapter._validate(env.obs, env.state)


if __name__ == '__main__':
    unittest.main()
