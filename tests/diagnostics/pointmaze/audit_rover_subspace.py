"""Audit a saved ROVER subspace fit without changing training artifacts.

Run from the repository root with the dist_matching Python environment.
Outputs default to tests/outputs/rover_subspace_audit. Full model reconstruction
uses all saved actor transitions and landmarks, on four CPU threads.
"""

from pathlib import Path
import argparse
import gc
import hashlib
import json
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigs, onenormest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from omegaconf import OmegaConf

from agent.rover_positive_kernel import PositiveKernelTransition, positive_kernel_logits


def policy_stats(pi):
    p = pi.detach().numpy()
    return dict(entropy_nats=float(-(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum(1).mean()),
                mean_tv_uniform=float(np.abs(p - 1 / p.shape[1]).sum(1).mean() / 2),
                mean_pmax=float(p.max(1).mean()),
                fraction_pmax_gt_099=float((p.max(1) > .99).mean()),
                zero_entries=int((p == 0).sum()))


def region_stats(states):
    xy = np.asarray(states)[:, :2]
    return dict(n=len(xy), right_half_fraction=float((xy[:, 0] > 0).mean()),
                far_right_fraction=float((xy[:, 0] > 2).mean()),
                bottom_right_fraction=float(((xy[:, 0] > 2) & (xy[:, 1] < -2)).mean()))


def unsupported_action_control():
    """A valid model can prefer leaving known space when sink coverage costs zero."""
    H = torch.ones(1000, 1, dtype=torch.float64)
    E = torch.zeros(1000, 2, dtype=torch.float64)
    E[:, 0] = 1
    gram = torch.diag(torch.tensor([1., 0.], dtype=torch.float64))
    model = PositiveKernelTransition(H, E, torch.zeros(1000, dtype=torch.long),
                                     gram, H[:1], .99, 1.)
    supported = torch.tensor([[1., 0.]], dtype=torch.float64)
    unsupported = 1-supported
    result = dict(supported_action_loss=float(model.loss(supported)),
                  unsupported_action_loss=float(model.loss(unsupported)))
    assert np.isclose(result["unsupported_action_loss"], .01**2)
    assert result["unsupported_action_loss"] < result["supported_action_loss"]
    return result


@torch.no_grad()
def checkpoint_stats(run, filename, out):
    payload = torch.load(run / "models" / filename, map_location="cpu", weights_only=False)
    a = payload["agent"]
    a.device = a.distribution_matcher.device = "cpu"
    support = a._pmd_previous_support
    prior = torch.softmax(-a._pmd_action_logits_offset, 1)
    ids = torch.linspace(0, len(support) - 1, 128).long()
    queried = torch.from_numpy(a._compute_action_probs_batch(support[ids].numpy()))
    query_error = float((queried - a.pi[ids]).abs().max())
    assert query_error < 1e-7, query_error
    assert a.gradient_coeff.shape == (len(a._phi_all_obs) + 1, 1)
    assert torch.equal(a._coverage_sub_next_features, support[:, :2])
    assert a._sub_alpha[0] == 1 and a._sub_alpha.sum() == 1
    assert torch.equal(support[0], support.new_tensor([-4, 3, 0, 0]))
    result = dict(snapshot=filename, step=int(payload["_global_step"]),
                  agent_class=f"{type(a).__module__}.{type(a).__name__}",
                  n=len(a._phi_all_obs), m=len(support),
                  coefficient_shape=list(a.gradient_coeff.shape),
                  prior=policy_stats(prior), policy=policy_stats(a.pi),
                  behavior=policy_stats(.7 * a.pi + .3 / a.n_actions),
                  query_vs_training_max_error=query_error,
                  prior_max_log_odds=float(a._pmd_action_logits_offset.max(1).values.sub(
                      a._pmd_action_logits_offset.min(1).values).max()),
                  prior_to_final_mean_tv=float((a.pi-prior).abs().sum(1).mean()/2),
                  dynamics_bandwidth=float(a.kernel_fn.bandwidth),
                  coverage_bandwidth=float(a.coverage_kernel_fn.bandwidth),
                  actor_support=region_stats(support),
                  fifo=region_stats(a._encoded_actor_fifo._data["proprio_state"]))
    fifo = a._encoded_actor_fifo
    temporal = []
    for lo in range(0, 260000, 20000):
        mask = (fifo._ids >= lo) & (fifo._ids < lo + 20000)
        if mask.any():
            temporal.append(dict(start=lo, **region_stats(fifo._data["proprio_state"][mask])))
    result["replay_windows"] = temporal
    np.savez_compressed(out / (filename + ".npz"), support=support.numpy(),
                        pi=a.pi.numpy(), prior=prior.numpy())
    return result, a


@torch.no_grad()
def reconstruct(a, expected_loss):
    print("Reconstructing full regression transition", flush=True)
    gamma, m = float(a.discount), len(a.pi)
    H = a.kernel_fn(a._phi_all_obs, a._phi_sub_next)
    Knm = a.distribution_matcher.state_action_kernel(
        a._phi_all_obs, a._phi_sub_obs, a._all_actions, a._sub_actions)
    Kmm = a.distribution_matcher.state_action_kernel(
        a._phi_sub_obs, a._phi_sub_obs, a._sub_actions, a._sub_actions)
    normal = Knm.T @ Knm + a.lambda_reg * Kmm
    normal.diagonal().add_(1e-6)
    del Kmm
    lu, piv = torch.linalg.lu_factor(normal)
    del normal
    F = torch.linalg.lu_solve(lu, piv, Knm.T @ (H * (a.E @ a.pi.T)))
    S = torch.eye(m, dtype=H.dtype) - gamma * F
    slu, spiv = torch.linalg.lu_factor(S)
    z = torch.linalg.lu_solve(slu, spiv, a._sub_alpha)
    G = a.coverage_kernel_fn(a._coverage_sub_next_features, a._coverage_sub_next_features)
    loss = float((1-gamma)**2 * (z.T @ G @ z).squeeze())
    assert np.isclose(loss, expected_loss, rtol=2e-4, atol=1e-7), (loss, expected_loss)
    adjoint = torch.linalg.lu_solve(slu, spiv, G @ z, adjoint=True)
    beta = 2 * gamma * (1-gamma)**2
    coeff = beta * Knm @ torch.linalg.lu_solve(lu, piv, adjoint, adjoint=True)
    Q = H.T @ (coeff * a.E)
    variance = (a.pi * (Q - (a.pi*Q).sum(1, keepdim=True)).square()).sum(1)
    direction = float(-(z[:, 0] * variance).sum())
    dpi = -a.pi * (Q-(a.pi*Q).sum(1, keepdim=True))
    dF = torch.linalg.lu_solve(lu, piv, Knm.T @ (H * (a.E @ dpi.T)))
    differences = []
    for step_size in (1e-2, 1e-3, 1e-4):
        # Perturb the same reconstructed system. Rebuilding F from scratch
        # otherwise swamps the small loss difference with normal-solve error.
        plus = torch.linalg.solve(S-gamma*step_size*dF, a._sub_alpha)
        minus = torch.linalg.solve(S+gamma*step_size*dF, a._sub_alpha)
        derivative = float((1-gamma)**2 * (plus.T@G@plus-minus.T@G@minus)/(2*step_size))
        differences.append(dict(step_size=step_size, derivative=derivative))
    damped = torch.linalg.solve((S + 1e-6*torch.eye(m, dtype=H.dtype)).T, G @ z)
    damped_coeff = beta * Knm @ torch.linalg.lu_solve(lu, piv, damped, adjoint=True)
    radius = float(abs(eigs(F.numpy(), k=1, which="LM", return_eigenvectors=False,
                            tol=1e-6, maxiter=5000)[0]))
    def solve(x, adjoint=False):
        rhs = torch.as_tensor(np.asarray(x).reshape(m, -1), dtype=H.dtype)
        return torch.linalg.lu_solve(slu, spiv, rhs, adjoint=adjoint).numpy().reshape(np.asarray(x).shape)
    inv = LinearOperator((m, m), matvec=solve, rmatvec=lambda x: solve(x, True), dtype=np.float64)
    d = (1-gamma)*z[:, 0]
    # Transform residual coordinates to actual landmark/sink probability coordinates.
    P = H.new_zeros(m+1, m+1)
    P[:-1, :-1] = F
    P[-1, :-1] = 1-F.sum(0)
    P[-1, -1] = 1
    result = dict(loss_reconstructed=loss, expected_log_loss=expected_loss,
                  negative_transition_fraction=float((F < 0).double().mean()),
                  transition_min=float(F.min()), transition_column_mass_min=float(F.sum(0).min()),
                  transition_column_mass_max=float(F.sum(0).max()),
                  residual_sink_transition_min=float(P[-1].min()),
                  discounted_spectral_radius=gamma*radius,
                  condition_estimate_1=float(torch.linalg.matrix_norm(S, ord=1))*float(onenormest(inv)),
                  landmark_occupancy_mass=float(d.sum()), sink_occupancy=float(1-d.sum()),
                  min_landmark_occupancy=float(d.min()),
                  negative_occupancy_l1=float(-d[d<0].sum()),
                  negative_occupancy_fraction=float((d<0).double().mean()),
                  max_action_score_gap=float((Q.max(1).values-Q.min(1).values).max()),
                  pmd_exact_directional_derivative=direction,
                  pmd_central_finite_differences=differences,
                  adjoint_damping_relative_error=float(torch.linalg.norm(coeff-damped_coeff)/torch.linalg.norm(coeff)))
    del Knm, F, S, slu, spiv, lu, piv, P, dF, G, coeff, damped_coeff
    gc.collect()
    print("Checking existing positive estimator on the same full support", flush=True)
    nearest = a._pmd_nearest_rows(a._phi_all_next, a._phi_sub_next)
    gram = H.new_zeros(m+1, m+1)
    gram[:-1, :-1] = a.coverage_kernel_fn(a._coverage_sub_next_features, a._coverage_sub_next_features)
    positive = []
    for sink_norm in (0., 1.):
        gram[-1, -1] = sink_norm**2
        model = PositiveKernelTransition(H, a.E, nearest, gram, a._sub_alpha, gamma, 1.)
        P = model.transition_matrix(a.pi)
        zp, Sp = model._resolvent(a.pi)
        dp = (1-gamma)*zp
        initial = float(model.loss(a.pi))
        c = model.gradient_coefficients(a.pi)
        qp = positive_kernel_logits(H.T, a.E, c, 1.)
        trial = torch.softmax(a.pi.clamp_min(torch.finfo(H.dtype).tiny).log() - 25*qp, 1)
        after = float(model.loss(trial))
        assert after <= initial
        positive.append(dict(sink_norm=sink_norm, loss=initial, loss_after_eta25=after,
                             min_transition=float(P.min()), column_mass_error=float((P.sum(0)-1).abs().max()),
                             min_occupancy=float(dp.min()), occupancy_mass=float(dp.sum()),
                             sink_occupancy=float(dp[-1]),
                             condition_upper_bound_1=float(torch.linalg.matrix_norm(Sp, ord=1))/(1-gamma)))
        # Fixed-data control: same normalized model, warm versus uniform prior.
        # These objective changes say nothing by themselves about rollout coverage.
        controls = []
        for name, prior in (("warm", a.pi), ("uniform", torch.full_like(a.pi, 1/a.n_actions))):
            pi, accumulated = prior.clone(), torch.zeros_like(c)
            initial_control = float(model.loss(pi))
            history = [initial_control]
            for _ in range(5):
                direction = model.gradient_coefficients(pi)
                eta = 25.
                for trial in range(13):
                    candidate_c = accumulated + eta*direction
                    candidate = torch.softmax(prior.clamp_min(torch.finfo(H.dtype).tiny).log()
                                              - positive_kernel_logits(H.T, a.E, candidate_c, 1.), 1)
                    loss = float(model.loss(candidate))
                    if loss <= history[-1]:
                        pi, accumulated = candidate, candidate_c
                        history.append(loss)
                        break
                    eta *= .25
                else:
                    break
            assert all(y <= x for x, y in zip(history, history[1:]))
            controls.append(dict(prior=name, losses=history, final_policy=policy_stats(pi)))
        positive[-1]["fixed_model_controls"] = controls
    return dict(regression=result, positive=positive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "exp_local_parallel/2026.10.19/213717_868576_PointMaze_LargeDense-v3_nenv4")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/outputs/rover_subspace_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    events = EventAccumulator(str(args.run / "tb"), size_guidance={"scalars": 0})
    events.Reload()
    scalars = {tag: [dict(step=e.step, value=e.value) for e in events.Scalars(tag)]
               for tag in ("eval/coverage_pct", "train/policy_deviation_from_uniform", "train/sink_norm")}
    updates, step = [], None
    for line in (args.run / "pretrain.log").read_text().splitlines():
        match = re.search(r"step_(\d+)_nystrom_kernels", line)
        if match:
            step = int(match[1])
        match = re.search(r"Actor loss \(squared norm of occupancy measure\): ([\deE.+-]+)", line)
        if match:
            updates.append(dict(step=step, initial=float(match[1]), final=float(match[1])))
        match = re.search(r"PMD Iteration \d+,.*Actor loss: ([\deE.+-]+), eta:", line)
        if match:
            updates[-1]["final"] = float(match[1])
    result = dict(run=str(args.run), scalars=scalars, actor_updates=updates, checkpoints=[])
    for name in ("best_snapshot.pt", "final_snapshot.pt"):
        metrics, agent = checkpoint_stats(args.run, name, args.output)
        result["checkpoints"].append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        if name == "final_snapshot.pt":
            result["model"] = reconstruct(agent, updates[-1]["final"])
        del agent
        gc.collect()
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (ROOT / "agent").glob("rover*.py")}
    result["unsupported_action_control"] = unsupported_action_control()
    (args.output / "audit.json").write_text(json.dumps(result, indent=2))
    overrides = list(OmegaConf.load(args.run / ".hydra/overrides.yaml"))
    command = ["python", "pretrain_parallel.py",
               "--config-name=pretrain_parallel/pointmaze_subspace_positive_states", *overrides]
    (args.output / "rerun_command.txt").write_text(shlex.join(command) + "\n")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for tag, axis, label in (("eval/coverage_pct", ax[0], "Evaluation coverage (%)"),
                             ("train/policy_deviation_from_uniform", ax[1], "Mean visited-state TV from uniform")):
        axis.plot([e["step"] for e in scalars[tag]], [e["value"] for e in scalars[tag]], "o-")
        axis.set(xlabel="Aggregate environment transitions", ylabel=label)
    ax[1].axhline(.75, color="gray", linestyle="--")
    ax[2].plot([u["step"] for u in updates], [u["initial"] for u in updates], "o-", label="Before PMD")
    ax[2].plot([u["step"] for u in updates], [u["final"] for u in updates], "o-", label="After PMD")
    ax[2].axhline(1, color="red", linestyle="--", label="Probability-model upper bound")
    ax[2].set(yscale="log", xlabel="Actor update step", ylabel="Model occupancy objective")
    ax[2].legend()
    fig.savefig(args.output / "coverage_and_model.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    windows = result["checkpoints"][-1]["replay_windows"]
    ax[0].bar([w["start"] for w in windows], [100*w["far_right_fraction"] for w in windows],
              width=18000, align="edge")
    ax[0].set(xlabel="Replay transition ID (20k windows)", ylabel="Far-right replay states x > 2 (%)")
    for name in ("best_snapshot.pt", "final_snapshot.pt"):
        data = np.load(args.output / (name + ".npz"))
        ax[1].hist(data["pi"].max(1), bins=np.linspace(.25, 1, 31), alpha=.6, label=name)
    ax[1].set(xlabel="Largest action probability on landmarks", ylabel="Landmark count")
    ax[1].legend()
    sc = ax[2].scatter(data["support"][:, 0], data["support"][:, 1],
                       c=data["pi"].max(1), s=3, vmin=.25, vmax=1)
    ax[2].set(xlabel="x", ylabel="y", aspect="equal", title="Final landmarks still span far branch")
    fig.colorbar(sc, ax=ax[2], label="Largest action probability")
    fig.savefig(args.output / "replay_and_support.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result["model"], indent=2), flush=True)


if __name__ == "__main__":
    main()
