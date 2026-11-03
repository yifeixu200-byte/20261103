"""Synthetic model checks; NOT simulator evaluation or reproduced returns."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import torch
import torch.nn.functional as F
from agent.world_models.diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from agent.world_models.diffusion.inner_model import StateInnerModelConfig
from agent.world_models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from agent.world_models.perceiver import PerceiverConfig
from agent.world_models.vq import SimpleFSQAutoEncoder
from agent.world_models.rew_end_model import TransRewEndModel
from agent.world_models.transformer import TransformerConfig
from networks.dreamer.action import StochasticPolicy
from networks.dreamer.critic import VNet
import numpy as np


def run(device: str) -> dict:
    torch.manual_seed(123)
    torch.set_num_threads(2)
    state_dim, obs_dim, action_dim, agents, batch = 17, 12, 3, 2, 2
    pc = PerceiverConfig(256, 512, 32, 2, 1, 64, 8, 64, 0., 0., 256, True)
    inner = StateInnerModelConfig(state_dim, 3, 256, [2]*3, [64]*3, [0]*3, action_dim)
    model = Denoiser(DenoiserConfig(inner, pc, .5, 0.), agents, is_continuous_act=True).to(device)
    model.setup_training(SigmaDistributionConfig(-.4, 1.2, .002, 20.))
    states = torch.randn(batch, 6, state_dim, device=device) * .5
    actions = torch.randn(batch, 6, agents, action_dim, device=device)
    data = SimpleNamespace(shared_obs=states, act=actions,
                           mask_padding=torch.ones(batch, 6, dtype=torch.bool, device=device))
    loss, _ = model(data)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.inner_model.conv_out.weight.grad.abs().sum() > 0

    # All historical actions stay visible; each current-step mask selects one agent.
    seen = []
    original = model.denoise
    def record(x, sigma, obs, act, mask):
        assert mask[:, :-1].eq(1).all()
        assert mask[:, -1].sum(-1).eq(1).all()
        seen.append(int(mask[0, -1].argmax()))
        return original(x, sigma, obs, act, mask)
    model.denoise = record
    sampler = DiffusionSampler(model, DiffusionSamplerConfig(4, agent_order=[1, 0, 0, 1]))
    prediction, trajectory = sampler.sample(states[:, :3], actions[:, :3])
    assert seen == [1, 0, 0, 1]
    assert prediction.shape == (batch, 1, state_dim) and torch.isfinite(prediction).all()
    assert len(trajectory) == 5
    model.denoise = original
    # Exercise repaired non-default Heun and churn path, plus >2 steps/agent.
    for cfg in [DiffusionSamplerConfig(4, order=2, s_churn=.1), DiffusionSamplerConfig(6)]:
        result, _ = DiffusionSampler(model, cfg).sample(states[:, :3], actions[:, :3])
        assert torch.isfinite(result).all()

    decoder = SimpleFSQAutoEncoder(state_dim, 12, [8, 6, 5], agents*obs_dim).to(device)
    target = torch.randn(batch, agents*obs_dim, device=device)
    reconstructed, indices = decoder(states[:, 0])
    decoder_loss = F.mse_loss(reconstructed, target)
    decoder_loss.backward()
    assert decoder.encoder[0].weight.grad.abs().sum() > 0
    assert int(indices.min()) >= 0 and int(indices.max()) < 240
    assert torch.allclose(decoder.encode_decode(states[:, 0]), reconstructed, atol=1e-6)

    tc = TransformerConfig(2, 15, 'causal', 6, 4, 256, .1, .1, .1)
    reward_model = TransRewEndModel(state_dim, action_dim, agents, tc, action_dim,
                                  False, use_ce_for_end=True).to(device)
    reward_batch = {'shared_obs': states[:, :3], 'action': actions[:, :3],
                    'reward': torch.randn(batch, 3, agents, 1, device=device),
                    'done': torch.zeros(batch, 3, agents, 1, device=device)}
    reward_loss, metrics = reward_model.compute_loss(reward_batch)
    reward_loss.backward()
    assert torch.isfinite(reward_loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in reward_model.parameters())
    reward_model.eval()
    tokens = torch.zeros(batch, 6, state_dim, device=device)
    tokens[:, ::2] = states[:, :3]
    cond = reward_model.get_act_emb(actions[:, :3].reshape(batch*3, agents, action_dim)).reshape(batch, 3, -1)
    with torch.no_grad():
        before = reward_model(tokens, perattn_out=cond).pred_rewards
        tokens[:, 4] += 100
        after = reward_model(tokens, perattn_out=cond).pred_rewards
    assert torch.allclose(before[:, :2], after[:, :2], atol=1e-5), 'Causal future leakage'
    space = SimpleNamespace(shape=(action_dim,), high=np.ones(action_dim, dtype=np.float32),
                            low=-np.ones(action_dim, dtype=np.float32))
    actor = StochasticPolicy(obs_dim, action_dim, 128, 3, continuous_action=True,
                             continuous_action_space=space, policy_class='gaussian').to(device)
    critic = VNet(state_dim, 256, 2).to(device)
    local = reconstructed.detach().reshape(batch, agents, obs_dim)
    with torch.no_grad():
        act, old_logp = actor(local)
    logp, entropy, _ = actor.evaluate_actions(local, act)
    ratio = (logp-old_logp).exp()
    advantages = torch.randn(batch, 1, 1, device=device)
    policy_loss = -torch.minimum(ratio*advantages, ratio.clamp(.9, 1.1)*advantages).mean() - .001*entropy
    value_loss = F.mse_loss(critic(states[:, 0]), torch.ones(batch, 1, device=device))
    (policy_loss+value_loss).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in critic.parameters())
    return {'status': 'PASS_SYNTHETIC_ONLY', 'device': device, 'torch': torch.__version__,
            'denoising_loss': float(loss), 'decoder_mse': float(decoder_loss),
            'reward_terminal_loss': float(reward_loss), 'checked': [
                'diffusion backward', 'current-action masking', 'E.5 explicit order',
                'Euler/Heun/churn finite', 'generic denoising repeat count',
                'FSQ gradient and encode/decode consistency', 'reward/terminal backward',
                'transformer causality', 'actor PPO ratio/backward and centralized critic backward'], 'simulator_training': 'NOT_RUN'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='smoke_result.json')
    args = parser.parse_args()
    report = run(args.device)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
