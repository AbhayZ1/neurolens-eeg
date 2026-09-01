"""Jansen-Rit neural mass model (JR-NMM) synthetic EEG generator.

Provides a mathematically grounded ground-truth benchmark for the NeuroLens
XAI evaluation: a continuous 6-state Jansen & Rit (1995) cortical column
model, driven by a mean input p(t) that is ramped through a numerically
located Hopf bifurcation of the model's fixed point. Below the bifurcation
the column sits at a damped stable focus that produces noise-driven
alpha-band background activity; above it the focus loses stability and the
sigmoid saturation of the population firing-rate nonlinearity folds the
trajectory onto a limit cycle, producing sustained epileptiform spike-wave
discharges. Because the bifurcation point p_crit is found by linear
stability analysis (not assumed from the literature), the reported
transition timestamps are exact with respect to the simulated dynamics
rather than visually/empirically defined.

Numerical backends:
    - scipy: equilibrium solving and root-finding (brentq) for the offline
      bifurcation analysis (fixed points, Jacobian eigenvalues, Hopf
      crossings).
    - torch: vectorized 4th-order Runge-Kutta (RK4) time integration of the
      multi-channel ODE system, so simulated traces are returned as
      dataset-ready tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.signal import decimate


# --------------------------------------------------------------------------
# Model parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JansenRitParams:
    """Standard Jansen & Rit (1995) cortical-column parameters."""

    A: float = 3.25       # mV, excitatory post-synaptic potential (PSP) gain
    B: float = 22.0        # mV, inhibitory PSP gain
    a: float = 100.0       # 1/s, excitatory synaptic rate constant
    b: float = 50.0        # 1/s, inhibitory synaptic rate constant
    C: float = 135.0       # dimensionless, average number of synaptic contacts
    e0: float = 2.5        # 1/s, half of the maximum firing rate
    r: float = 0.56        # 1/mV, sigmoid slope at v0
    v0: float = 6.0        # mV, PSP at half-maximal firing rate

    @property
    def C1(self) -> float:
        return self.C

    @property
    def C2(self) -> float:
        return 0.8 * self.C

    @property
    def C3(self) -> float:
        return 0.25 * self.C

    @property
    def C4(self) -> float:
        return 0.25 * self.C


def _sigmoid_np(v: np.ndarray, params: JansenRitParams) -> np.ndarray:
    """Sigmoid firing-rate nonlinearity S(v), numpy path (used offline)."""
    return 2.0 * params.e0 / (1.0 + np.exp(params.r * (params.v0 - v)))


def _sigmoid_prime_np(v: np.ndarray, params: JansenRitParams) -> np.ndarray:
    """Analytic derivative dS/dv, used to build the Jacobian."""
    ex = np.exp(params.r * (params.v0 - v))
    return 2.0 * params.e0 * params.r * ex / (1.0 + ex) ** 2


# --------------------------------------------------------------------------
# Offline bifurcation analysis (equilibria, Jacobian, Hopf points)
# --------------------------------------------------------------------------


def _y1_y2_of_y0(y0: np.ndarray, p: float, params: JansenRitParams) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form y1, y2 as functions of y0 at equilibrium (y3=y4=y5=0)."""
    s_exc = _sigmoid_np(params.C1 * y0, params)
    s_inh = _sigmoid_np(params.C3 * y0, params)
    y1 = (params.A / params.a) * (p + params.C2 * s_exc)
    y2 = (params.B / params.b) * params.C4 * s_inh
    return y1, y2


def _y0_residual(y0: np.ndarray, p: float, params: JansenRitParams) -> np.ndarray:
    """Scalar fixed-point residual g(y0) = (A/a) S(y1(y0) - y2(y0)) - y0.

    The three algebraic equilibrium equations of the JR system reduce
    exactly to this single-variable equation, which is solved by bracketing
    sign changes on a dense grid and refining with Brent's method. This is
    far more robust than a multivariate Newton solve (fsolve) across the
    model's known fold/bistability region (Grimbert & Faugeras, 2006).
    """
    y1, y2 = _y1_y2_of_y0(y0, p, params)
    return (params.A / params.a) * _sigmoid_np(y1 - y2, params) - y0


def _all_equilibrium_roots(
    p: float,
    params: JansenRitParams,
    y0_bounds: Tuple[float, float] = (-5.0, 15.0),
    grid_size: int = 4001,
) -> np.ndarray:
    """All y0 roots of the scalar equilibrium equation at input p, ascending.

    The Jansen-Rit sigmoid nonlinearity gives the equilibrium curve an
    S-shape in p over part of its range (Grimbert & Faugeras, 2006): up to
    three coexisting fixed points (low/middle/high branches) can exist
    simultaneously. Returning all of them lets callers distinguish a
    genuine Hopf bifurcation (a complex eigenvalue pair crossing the
    imaginary axis while continuously following one branch) from a
    saddle-node/fold bifurcation (a branch terminating where two roots
    merge and annihilate) -- only the former is a Hopf bifurcation.
    """
    grid = np.linspace(y0_bounds[0], y0_bounds[1], grid_size)
    vals = _y0_residual(grid, p, params)
    signs = np.sign(vals)

    roots = grid[vals == 0.0].tolist()
    change_idx = np.where((signs[:-1] != signs[1:]) & (signs[:-1] != 0) & (signs[1:] != 0))[0]
    for i in change_idx:
        roots.append(brentq(_y0_residual, grid[i], grid[i + 1], args=(p, params), xtol=1e-12))
    return np.array(sorted(roots))


def find_equilibrium(
    p: float,
    params: JansenRitParams,
    x0: Optional[np.ndarray] = None,
    y0_bounds: Tuple[float, float] = (-5.0, 15.0),
    grid_size: int = 4001,
) -> np.ndarray:
    """Solve for the fixed point (y0, y1, y2) of the JR system at input p.

    When the system is bistable, ``x0`` (a previous equilibrium, typically
    from a neighboring p on the same continuation branch) selects the root
    whose y0 is closest to ``x0[0]``; otherwise the lowest-y0 root is
    returned. Note this local nearest-neighbor rule is only branch-safe for
    small steps in p -- for bifurcation analysis across the full range,
    use :func:`trace_equilibrium_branches` instead, which tracks each
    branch explicitly and stops at folds rather than jumping across them.
    """
    roots = _all_equilibrium_roots(p, params, y0_bounds, grid_size)
    if roots.size == 0:
        raise RuntimeError(
            f"No equilibrium found for p={p} within y0 in {y0_bounds}; widen y0_bounds."
        )

    if x0 is not None:
        y0_guess = float(np.asarray(x0).reshape(-1)[0])
        y0_star = float(roots[np.argmin(np.abs(roots - y0_guess))])
    else:
        y0_star = float(roots[0])

    y1_star, y2_star = _y1_y2_of_y0(np.array(y0_star), p, params)
    return np.array([y0_star, float(y1_star), float(y2_star)])


def jacobian(y0: float, y1: float, y2: float, p: float, params: JansenRitParams) -> np.ndarray:
    """Analytic 6x6 Jacobian of the JR vector field at state (y0..y2, 0, 0, 0).

    State ordering is [y0, y1, y2, y3, y4, y5] with y3 = dy0/dt, y4 = dy1/dt,
    y5 = dy2/dt (standard second-order-to-first-order reduction).
    """
    s_prime_pyr = _sigmoid_prime_np(y1 - y2, params)
    s_prime_exc = _sigmoid_prime_np(params.C1 * y0, params)
    s_prime_inh = _sigmoid_prime_np(params.C3 * y0, params)
    a, b, A, B = params.a, params.b, params.A, params.B

    jac = np.zeros((6, 6))
    jac[0, 3] = 1.0
    jac[1, 4] = 1.0
    jac[2, 5] = 1.0

    jac[3, 0] = -a ** 2
    jac[3, 1] = A * a * s_prime_pyr
    jac[3, 2] = -A * a * s_prime_pyr
    jac[3, 3] = -2.0 * a

    jac[4, 0] = A * a * params.C2 * s_prime_exc * params.C1
    jac[4, 1] = -a ** 2
    jac[4, 4] = -2.0 * a

    jac[5, 0] = B * b * params.C4 * s_prime_inh * params.C3
    jac[5, 2] = -b ** 2
    jac[5, 5] = -2.0 * b
    return jac


def stability_at(
    p: float, params: JansenRitParams, x0: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Max real eigenvalue part, full eigenvalue spectrum, and fixed point at p."""
    eq = find_equilibrium(p, params, x0)
    eig = np.linalg.eigvals(jacobian(*eq, p, params))
    return float(np.max(eig.real)), eig, eq


@dataclass
class EquilibriumBranch:
    """One continuously-existing equilibrium branch over a range of p.

    Boundaries where a branch appears/disappears are saddle-node (fold)
    points, not tracked across into a different branch.
    """

    p: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    y2: np.ndarray
    max_real: np.ndarray   # leading eigenvalue's real part along the branch
    max_imag: np.ndarray   # that eigenvalue's imaginary part (0 => real eigenvalue)


def trace_equilibrium_branches(
    params: JansenRitParams,
    p_min: float = 0.0,
    p_max: float = 400.0,
    n_points: int = 1601,
    match_tol: float = 0.01,
) -> List[EquilibriumBranch]:
    """Sweep p and track every equilibrium branch without jumping at folds.

    At each grid point every coexisting root is found (see
    :func:`_all_equilibrium_roots`); each active branch is continued onto
    the nearest new root within ``match_tol`` in y0. A branch with no match
    ends there (a fold); an unmatched new root starts a new branch there
    (also a fold, or the domain edge). Because a branch by construction
    never jumps to a disjoint root, a sign change of the leading
    eigenvalue's real part *within* a branch is a true local bifurcation of
    that fixed point, not a discontinuity artifact.
    """
    p_grid = np.linspace(p_min, p_max, n_points)
    active: List[dict] = []
    finished: List[dict] = []

    for p in p_grid:
        roots = _all_equilibrium_roots(p, params)
        used = np.zeros(len(roots), dtype=bool)
        still_active = []

        for br in active:
            last_y0 = br["y0"][-1]
            if roots.size:
                idx = int(np.argmin(np.abs(roots - last_y0)))
            else:
                idx = -1
            if idx >= 0 and not used[idx] and abs(roots[idx] - last_y0) < match_tol:
                used[idx] = True
                y0 = float(roots[idx])
                y1, y2 = _y1_y2_of_y0(np.array(y0), p, params)
                eig = np.linalg.eigvals(jacobian(y0, float(y1), float(y2), p, params))
                k = int(np.argmax(eig.real))
                br["p"].append(p)
                br["y0"].append(y0)
                br["y1"].append(float(y1))
                br["y2"].append(float(y2))
                br["mr"].append(float(eig.real[k]))
                br["mi"].append(float(eig.imag[k]))
                still_active.append(br)
            else:
                finished.append(br)

        for j, y0 in enumerate(roots):
            if used[j]:
                continue
            y0 = float(y0)
            y1, y2 = _y1_y2_of_y0(np.array(y0), p, params)
            eig = np.linalg.eigvals(jacobian(y0, float(y1), float(y2), p, params))
            k = int(np.argmax(eig.real))
            still_active.append(
                {
                    "p": [p], "y0": [y0], "y1": [float(y1)], "y2": [float(y2)],
                    "mr": [float(eig.real[k])], "mi": [float(eig.imag[k])],
                }
            )
        active = still_active

    finished.extend(active)
    branches = [
        EquilibriumBranch(
            p=np.array(br["p"]), y0=np.array(br["y0"]), y1=np.array(br["y1"]),
            y2=np.array(br["y2"]), max_real=np.array(br["mr"]), max_imag=np.array(br["mi"]),
        )
        for br in finished
        if len(br["p"]) >= 2
    ]
    return branches


@dataclass
class HopfPoint:
    """A genuine Hopf bifurcation: a complex eigenvalue pair crossing Re=0."""

    p_crit: float
    omega_rad_s: float  # angular frequency of the crossing eigenvalue pair
    y0: float

    @property
    def frequency_hz(self) -> float:
        return self.omega_rad_s / (2.0 * np.pi)


def find_hopf_points(
    params: JansenRitParams,
    p_min: float = 0.0,
    p_max: float = 400.0,
    n_points: int = 1601,
    imag_threshold: float = 1.0,
) -> List[HopfPoint]:
    """Locate genuine Hopf bifurcations of the Jansen-Rit fixed point.

    Traces every equilibrium branch (see :func:`trace_equilibrium_branches`)
    and, within each branch, brackets sign changes of the leading
    eigenvalue's real part. A bracket is accepted as a Hopf point only if
    the crossing eigenvalue has a non-negligible imaginary part
    (``|Im| > imag_threshold``, in rad/s) -- i.e. it is one of a complex-
    conjugate pair, which is the defining signature of a Hopf bifurcation.
    Sign changes with a near-zero imaginary part are saddle-node-type
    crossings of a real eigenvalue and are excluded.
    """
    branches = trace_equilibrium_branches(params, p_min, p_max, n_points)
    hopf_points: List[HopfPoint] = []

    for br in branches:
        for i in range(len(br.p) - 1):
            lo_val, hi_val = br.max_real[i], br.max_real[i + 1]
            if lo_val == 0.0 or hi_val == 0.0 or np.sign(lo_val) == np.sign(hi_val):
                continue

            p_lo, p_hi = br.p[i], br.p[i + 1]
            y0_lo, y0_hi = br.y0[i], br.y0[i + 1]

            def f(p: float) -> float:
                y0_guess = np.interp(p, [p_lo, p_hi], [y0_lo, y0_hi])
                eq = find_equilibrium(p, params, x0=np.array([y0_guess]))
                mr = float(np.max(np.linalg.eigvals(jacobian(*eq, p, params)).real))
                return mr

            p_c = brentq(f, p_lo, p_hi, xtol=1e-8)
            y0_c_guess = np.interp(p_c, [p_lo, p_hi], [y0_lo, y0_hi])
            eq_c = find_equilibrium(p_c, params, x0=np.array([y0_c_guess]))
            eig_c = np.linalg.eigvals(jacobian(*eq_c, p_c, params))
            k = int(np.argmax(eig_c.real))
            omega = float(abs(eig_c.imag[k]))
            if omega > imag_threshold:
                hopf_points.append(HopfPoint(p_crit=float(p_c), omega_rad_s=omega, y0=float(eq_c[0])))

    hopf_points.sort(key=lambda h: h.p_crit)
    return hopf_points


# --------------------------------------------------------------------------
# RK4 time integration
# --------------------------------------------------------------------------
#
# The forward simulation is a long sequential recurrence (each timestep
# depends on the previous one), so it cannot be batched across time the way
# a typical PyTorch workload is. Per-step elementwise-op dispatch overhead
# (measured empirically: ~2-5 ms/step eager, ~0.9 ms/step under
# torch.jit.script) makes a multi-channel, multi-hour-equivalent simulation
# impractically slow in torch. The integrator below therefore runs on plain
# numpy (~0.13 ms/step, channel-vectorized), which is dispatch-cheap for
# small arrays; results are packaged into torch tensors at the boundary of
# generate_synthetic_bifurcation_dataset for direct use with torch-based
# downstream pipelines (see dataset.py).


def _derivative(state: np.ndarray, p: np.ndarray, params: JansenRitParams) -> np.ndarray:
    """Vector field dstate/dt for state of shape (6, n_channels)."""
    y0, y1, y2, y3, y4, y5 = state

    def sig(v: np.ndarray) -> np.ndarray:
        return 2.0 * params.e0 / (1.0 + np.exp(params.r * (params.v0 - v)))

    dy0 = y3
    dy1 = y4
    dy2 = y5
    dy3 = params.A * params.a * sig(y1 - y2) - 2.0 * params.a * y3 - params.a ** 2 * y0
    dy4 = (
        params.A * params.a * (p + params.C2 * sig(params.C1 * y0))
        - 2.0 * params.a * y4
        - params.a ** 2 * y1
    )
    dy5 = params.B * params.b * params.C4 * sig(params.C3 * y0) - 2.0 * params.b * y5 - params.b ** 2 * y2
    return np.stack((dy0, dy1, dy2, dy3, dy4, dy5), axis=0)


def _rk4_step(
    state: np.ndarray, p: np.ndarray, dt: float, params: JansenRitParams
) -> np.ndarray:
    k1 = _derivative(state, p, params)
    k2 = _derivative(state + 0.5 * dt * k1, p, params)
    k3 = _derivative(state + 0.5 * dt * k2, p, params)
    k4 = _derivative(state + dt * k3, p, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _smoothstep_ramp(
    t: np.ndarray, start: float, duration: float, lo: float, hi: float
) -> np.ndarray:
    """C1-continuous ramp from lo to hi over [start, start + duration]."""
    frac = np.clip((t - start) / duration, 0.0, 1.0)
    smooth = frac * frac * (3.0 - 2.0 * frac)
    return lo + (hi - lo) * smooth


# --------------------------------------------------------------------------
# Public dataset containers
# --------------------------------------------------------------------------


@dataclass
class BifurcationGroundTruth:
    """Exact, analytically-derived facts about the simulated regime change."""

    p_crit: float
    p_pre: float
    p_post: float
    transition_start_sec: float
    transition_duration_sec: float
    bifurcation_onset_sec: float
    bifurcation_onset_sample: int
    regime_established_sec: float
    regime_established_sample: int
    eigenvalues_pre: np.ndarray
    eigenvalues_post: np.ndarray


@dataclass
class SyntheticEEGDataset:
    """A simulated multi-channel EEG record with an embedded Hopf bifurcation."""

    eeg_uv: torch.Tensor           # (n_channels, n_samples), microvolts
    times_sec: torch.Tensor        # (n_samples,)
    sampling_rate: int
    channel_names: List[str]
    p_drive: torch.Tensor          # (n_samples,), noiseless mean input trajectory
    stability_index: torch.Tensor  # (n_samples,), frozen-time max real eigenvalue part
    ground_truth: BifurcationGroundTruth
    params: JansenRitParams = field(repr=False)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def generate_synthetic_bifurcation_dataset(
    duration_mins: float,
    sampling_rate: int = 256,
    transition_start_min: float = 45.0,
    transition_duration_min: float = 2.0,
    n_channels: int = 18,
    channel_names: Optional[Sequence[str]] = None,
    p_pre: Optional[float] = None,
    p_post: Optional[float] = None,
    bifurcation_margin: float = 25.0,
    input_noise_std: float = 5.0,
    channel_gain_jitter: float = 0.05,
    sensor_noise_uv: float = 2.0,
    sim_oversample: int = 4,
    params: Optional[JansenRitParams] = None,
    seed: int = 42,
) -> SyntheticEEGDataset:
    """Simulate multi-channel EEG that crosses a genuine Hopf bifurcation.

    The pyramidal population output y1 - y2 (David & Friston, 2003) is used
    as the observed local field potential of each simulated channel. All
    channels share the same underlying input drive p(t) (a shared, patient-
    level regime change) but receive independent process and sensor noise
    plus a small per-channel gain jitter, so the transition is coherent
    across channels without channels being trivially identical.

    Args:
        duration_mins: total recording duration in minutes.
        sampling_rate: output sampling rate in Hz (the model is integrated
            at ``sampling_rate * sim_oversample`` Hz and decimated down).
        transition_start_min: minute at which the input ramp begins.
        transition_duration_min: duration of the ramp, in minutes.
        n_channels: number of simulated channels.
        channel_names: optional channel labels (defaults to CH0..CHn-1).
        p_pre: constant input before the transition. If None, chosen
            automatically below the nearest computed Hopf point.
        p_post: constant input after the transition. If None, chosen
            automatically above the nearest computed Hopf point.
        bifurcation_margin: pulses/s offset from p_crit used to pick
            p_pre/p_post when they are not given explicitly.
        input_noise_std: std. dev. (pulses/s) of the physiological input
            noise driving each channel independently. Keep this well below
            ``bifurcation_margin`` (e.g. <= margin / 4): noise excursions
            that momentarily cross p_crit will contaminate the pre-
            transition segment with transient nonlinear/low-frequency
            activity instead of the intended noise-driven alpha rhythm.
        channel_gain_jitter: fractional per-channel amplitude jitter,
            emulating variable electrode/volume-conduction gain.
        sensor_noise_uv: additive white sensor noise, in microvolts.
        sim_oversample: integration rate multiplier over sampling_rate.
        params: Jansen-Rit parameters (defaults to Jansen & Rit, 1995).
        seed: random seed for noise reproducibility.

    Returns:
        A SyntheticEEGDataset with simulated traces and exact bifurcation
        ground truth.
    """
    if duration_mins <= 0:
        raise ValueError("duration_mins must be positive")
    if transition_start_min + transition_duration_min > duration_mins:
        raise ValueError("transition window must fit inside duration_mins")

    params = params or JansenRitParams()
    channel_names = list(channel_names) if channel_names is not None else [
        f"CH{i}" for i in range(n_channels)
    ]
    if len(channel_names) != n_channels:
        raise ValueError("channel_names must have length n_channels")

    # --- Locate a genuine Hopf bifurcation and pick pre/post drive levels ---
    hopf_points = find_hopf_points(params)
    if not hopf_points:
        raise RuntimeError(
            "No Hopf bifurcation found for the given JansenRitParams in the "
            "scanned input range; adjust params or the scan range."
        )
    hopf = hopf_points[0]
    p_crit = hopf.p_crit

    if p_pre is None:
        p_pre = max(0.0, p_crit - bifurcation_margin)
    if p_post is None:
        p_post = p_crit + bifurcation_margin

    # Seed the equilibrium solve at p_pre/p_post with the Hopf point's own
    # y0 so both endpoints resolve to the *same* branch the crossing was
    # found on, rather than an arbitrary coexisting root.
    branch_seed = np.array([hopf.y0])
    mr_pre, eig_pre, eq_pre = stability_at(p_pre, params, branch_seed)
    mr_post, eig_post, eq_post = stability_at(p_post, params, branch_seed)
    if mr_pre >= 0.0 or mr_post <= 0.0:
        raise RuntimeError(
            f"Chosen p_pre={p_pre:.3f} (max Re(eig)={mr_pre:.4f}) / "
            f"p_post={p_post:.3f} (max Re(eig)={mr_post:.4f}) do not bracket a "
            f"stable->unstable transition around p_crit={p_crit:.3f}; widen "
            "bifurcation_margin or pass p_pre/p_post explicitly."
        )

    # --- Time base ---------------------------------------------------
    fs_sim = int(sampling_rate * sim_oversample)
    dt = 1.0 / fs_sim
    n_steps = int(round(duration_mins * 60.0 * fs_sim))
    transition_start_sec = transition_start_min * 60.0
    transition_duration_sec = transition_duration_min * 60.0

    t_sim = np.arange(n_steps, dtype=np.float64) * dt
    p_base = _smoothstep_ramp(
        t_sim, transition_start_sec, transition_duration_sec, p_pre, p_post
    )

    # Exact bifurcation crossing time via root-finding on the (monotonic,
    # noise-free) drive trajectory restricted to the ramp interval.
    def p_base_fn(t: float) -> float:
        return float(
            _smoothstep_ramp(
                np.array(t), transition_start_sec, transition_duration_sec, p_pre, p_post
            )
            - p_crit
        )

    bifurcation_onset_sec = brentq(
        p_base_fn,
        transition_start_sec,
        transition_start_sec + transition_duration_sec,
        xtol=1e-9,
    )

    # --- Stochastic multi-channel drive and RK4 integration ----------
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_steps, n_channels))
    p_full = p_base[:, None] + input_noise_std * noise  # (n_steps, n_channels)

    state = np.zeros((6, n_channels), dtype=np.float64)
    state[:3, :] = np.asarray(eq_pre)[:, None]
    state = state + 1e-3 * rng.standard_normal((6, n_channels))

    trace = np.empty((n_steps, n_channels), dtype=np.float64)
    for step in range(n_steps):
        trace[step] = state[1] - state[2]  # y1 - y2, pyramidal LFP proxy (mV)
        state = _rk4_step(state, p_full[step], dt, params)

    # --- Frozen-time stability index (quasi-static linearization) ----
    # p ramps slowly relative to the synaptic time constants (a, b), so the
    # instantaneous local stability is well approximated by linearizing
    # about the equilibrium of the noise-free drive p_base(t) at each time.
    stability_sim = np.empty(n_steps)
    x0 = np.array(eq_pre)
    unique_p, inverse = np.unique(p_base, return_inverse=True)
    unique_stability = np.empty_like(unique_p)
    for i, p_val in enumerate(unique_p):
        mr, _eig, eq = stability_at(float(p_val), params, x0)
        unique_stability[i] = mr
        x0 = eq
    stability_sim[:] = unique_stability[inverse]

    # --- Decimate to output sampling rate -----------------------------
    trace_np = np.ascontiguousarray(trace.T)  # (n_channels, n_steps)
    if sim_oversample > 1:
        eeg_mv = np.ascontiguousarray(decimate(trace_np, sim_oversample, axis=1, zero_phase=True))
        p_drive_out = np.ascontiguousarray(
            decimate(p_base[None, :], sim_oversample, axis=1, zero_phase=True)[0]
        )
        stability_out = np.ascontiguousarray(
            decimate(stability_sim[None, :], sim_oversample, axis=1, zero_phase=True)[0]
        )
    else:
        eeg_mv = trace_np
        p_drive_out = p_base
        stability_out = stability_sim
    n_out = eeg_mv.shape[1]

    gain = 1.0 + channel_gain_jitter * rng.standard_normal(n_channels)
    eeg_uv = eeg_mv * gain[:, None] * 1000.0  # mV -> uV, plus per-channel gain
    eeg_uv = eeg_uv + sensor_noise_uv * rng.standard_normal(eeg_uv.shape)

    times_out = np.arange(n_out) / sampling_rate
    bifurcation_onset_sample = int(round(bifurcation_onset_sec * sampling_rate))
    regime_established_sec = transition_start_sec + transition_duration_sec
    regime_established_sample = int(round(regime_established_sec * sampling_rate))

    ground_truth = BifurcationGroundTruth(
        p_crit=p_crit,
        p_pre=p_pre,
        p_post=p_post,
        transition_start_sec=transition_start_sec,
        transition_duration_sec=transition_duration_sec,
        bifurcation_onset_sec=bifurcation_onset_sec,
        bifurcation_onset_sample=bifurcation_onset_sample,
        regime_established_sec=regime_established_sec,
        regime_established_sample=regime_established_sample,
        eigenvalues_pre=eig_pre,
        eigenvalues_post=eig_post,
    )

    return SyntheticEEGDataset(
        eeg_uv=torch.tensor(eeg_uv, dtype=torch.float32),
        times_sec=torch.tensor(times_out, dtype=torch.float64),
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        p_drive=torch.tensor(p_drive_out, dtype=torch.float64),
        stability_index=torch.tensor(stability_out, dtype=torch.float64),
        ground_truth=ground_truth,
        params=params,
    )


if __name__ == "__main__":
    # Smoke test: a short simulation with an early transition, printing the
    # located Hopf point and the resulting exact bifurcation timestamp.
    ds = generate_synthetic_bifurcation_dataset(
        duration_mins=2.0,
        transition_start_min=0.8,
        transition_duration_min=0.2,
        n_channels=4,
        sim_oversample=4,
    )
    gt = ds.ground_truth
    print(f"p_crit={gt.p_crit:.3f} p_pre={gt.p_pre:.3f} p_post={gt.p_post:.3f}")
    print(
        f"bifurcation onset: {gt.bifurcation_onset_sec:.3f}s "
        f"(sample {gt.bifurcation_onset_sample})"
    )
    print(f"eeg_uv shape: {tuple(ds.eeg_uv.shape)}")
    print(f"stability_index range: [{ds.stability_index.min():.4f}, {ds.stability_index.max():.4f}]")
