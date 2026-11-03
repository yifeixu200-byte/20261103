# Explicit reproduction changes

The adjacent official source snapshot remains untouched. These changes are local reproducibility adaptations, not author-provided updates or evidence of reproduced scores.

## Profiles

`train.py --profile paper` (default) selects FSQ (implemented levels `[8, 6, 5]`), four actor-critic training iterations per epoch including the first epoch, cross-entropy termination prediction, and a state-only centralized critic. These choices follow Appendix G Tables 9–10 and the paper's critic description. Both learner and controller receive the FSQ selection. The critic input choice is propagated into normal and terminal imagined states.

`--profile official` retains the original experiment configuration choices (including VQ, five actor-critic iterations, and state plus last joint action critic input); `--ce_for_cont` remains available. The official profile still includes the runtime repairs below and is therefore not a byte-identical upstream run. Use the separate official snapshot when that is required.

## Runtime repairs

- Environment dependencies are imported only when their environment is instantiated. A MuJoCo run no longer needs unrelated StarCraft, football, SMACv2, or PettingZoo packages merely to import configuration classes.
- The diffusion sampler repeats agent IDs by `num_steps_denoising // num_agents`, rather than hardcoding two repetitions. An explicit zero-based schedule can be supplied as `DiffusionSamplerConfig(agent_order=[1, 0, 0, 1], ...)` for the random order `(2, 1, 1, 2)` in Appendix E.5. Ascending and descending schedules are `[0, 0, 1, 1]` and `[1, 1, 0, 0]`. The existing `random` string retains the original random-agent-then-repeat behavior.
- Heun's second denoiser call now receives the action mask. The first call receives the churn-adjusted noise level. These repairs do not alter the paper's Euler sampler with zero churn.

## Remaining audit limitations

Upstream `load_pretrained` is not a training-resume implementation: it does not restore normalizer/critic/optimizer/RNG state and switches into a debug evaluation path when actor weights exist. Use purpose-built evaluation loading and do not claim exact resumed training from this routine.

Upstream `log_compounding_errors` samples only 20 segments and feeds unnormalized states into models trained on normalized states. It must not be used unchanged for the paper's 100-segment Appendix E.5 result.

Training budget checks occur at episode boundaries; the final collected transition count may exceed the requested budget. Report actual collected steps, and keep evaluation transitions separate from training transitions.

The upstream evaluation workers receive the same environment seed, so deterministic policies can produce duplicated trajectories across workers. The local runner now deep-copies the nested curriculum for each evaluation worker, adding `100000 + worker_index` to leaf seeds and MuJoCo `env_args.random_seed`. Training seeds are unchanged. This deliberate evaluation protocol correction applies to both profiles. Evaluation means are appended and flushed to `RUN_DIR/evaluations.csv` after every evaluation; the upstream end-of-run pickle remains available.

## Equation-aligned loss choices

The paper profile uses FSQ mean squared reconstruction error (Appendix G Eq. 22), whereas upstream uses L1. It uses raw lambda-return mean squared critic error (Appendix C Eq. 20), disabling upstream value normalization, Huber loss and clipped value objective for this branch. The official profile retains its original losses. These are explicit paper-versus-code resolutions, not claims about which settings generated the authors' plots.

The upstream EDM `sigma_offset_noise=0.3` remains unchanged: the paper does not specify an offset-noise value, so choosing zero would be another unsupported assumption. This is an unresolved reproduction detail requiring author confirmation or a separately labeled ablation.

`python -m unittest discover -s tests -p test_reproduction_support.py` verifies unique deterministic evaluation seeds without mutating the training curriculum, and immediate CSV persistence with a single header. It requires no simulator packages.

No simulation return or statistical reproduction is asserted by these source changes. Full simulator training and multi-seed evaluation are required.
