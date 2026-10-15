"""Grain experiments: settle a mode with grains on it, then sweep the tension.

Two stages, so one settling run can feed several sweeps:

    settled = settle(model, surface, mode, a0, drive="pll+agc", ...)
    result  = sweep(settled, a1, rate, t_hold)

The wave is driven by one of the drives from forcing/ (the phase-locked loop
by default, the table lookup as baseline) with the peak acceleration held
constant, so the grains feel the same shaking strength while the nodes move.
"""
import copy
import pathlib
import time

import numpy as np

from forcing import schedules
from forcing.compensation import response
from forcing.simulation import Simulation, make_drive
from .grains import Grains

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


class Settled:
    """State at the end of the settling stage (see ``settle``)."""


def _drive(kind, model, mode, sched, t_agc):
    kw = dict(quantity="acceleration")
    if kind == "pll+agc":
        kw["agc_start"] = t_agc
    return make_drive(kind, model, mode, sched, **kw)


def _advance(sim, surface, grains, t_end, record_dt, rec):
    every = max(1, int(round(record_dt / sim.dt)))
    nsteps = int(round((t_end - sim.t) / sim.dt))
    for i in range(nsteps):
        if i % every == 0:
            _record(sim, grains, rec)
        sim.step()
        surface.update(sim.u, sim.v)
        grains.step(sim.dt, surface)
    _record(sim, grains, rec)


def _record(sim, grains, rec):
    rec["t"].append(sim.t)
    rec["a"].append(sim.a)
    rec["X"].append(grains.X.astype(np.float32))
    rec["airborne"].append(1.0 - grains.on_surface.mean())
    rec["w"].append(sim.drive.w)
    rec["peak"].append(sim.history[-1]["peak"] if sim.history else np.nan)


def settle(model, surface, mode, a0, drive="pll+agc", amplitude=0.05, gamma=4.0, n_grains=400,
           t_spin=50.0, t_settle=1000.0, dt=0.02, record_dt=2.0, seed=0, grain_kwargs=None,
           verbose=True):
    """Start the wave in its steady state at slope a0, scatter grains evenly
    and let them find the nodes.

    amplitude : peak displacement of the standing wave
    gamma     : peak surface acceleration in units of g (grains lift off
                wherever the local value exceeds 1)
    """
    sched = schedules.constant(a0)
    sim = Simulation(model, _drive(drive, model, mode, sched, 0.5 * t_spin), sched, mode, dt=dt)
    sim.F0 = amplitude / response(model, a0, mode, sim.shape, "peak")
    sim.start_in_steady_state().run(t_spin)
    w = model.resonance(a0, mode)
    g = amplitude * w * w / gamma

    surface.update(sim.u, sim.v)
    grains = Grains(n_grains, surface, g, seed=seed, **(grain_kwargs or {}))
    rec = dict(t=[], a=[], X=[], airborne=[], w=[], peak=[])
    t0 = time.time()
    _advance(sim, surface, grains, t_spin + t_settle, record_dt, rec)
    if verbose:
        print(f"settled ({drive}): amplitude {amplitude}, frequency {w:.4f}, g = {g:.4g} "
              f"(peak acceleration {gamma} g), {time.time() - t0:.0f} s", flush=True)

    s = Settled()
    s.__dict__.update(model=model, surface=surface, mode=mode, a0=a0, drive=drive, sim=sim,
                      grains=grains, rec=rec, g=g, amplitude=amplitude, gamma=gamma,
                      record_dt=record_dt, t_spin=t_spin)
    return s


def sweep(settled, a1=None, rate=1e-4, t_hold=500.0, verbose=True):
    """Continue from a settled state: sweep the slope to a1 and hold.

    The settled state is not modified, so it can be reused for other rates.
    With a1 = None nothing is swept and the settling record is returned.
    """
    s = settled
    rec = {k: list(v) for k, v in s.rec.items()}
    sweeping = a1 is not None and a1 != s.a0
    sim = s.sim
    if sweeping:
        t_start = s.sim.t
        duration = abs(a1 - s.a0) / rate
        sched = schedules.linear_ramp(s.a0, a1, t_start, rate)
        sim = Simulation(s.model, _drive(s.drive, s.model, s.mode, sched, 0.0), sched, s.mode,
                         dt=s.sim.dt).continue_from(s.sim)
        grains = copy.deepcopy(s.grains)
        s.surface.update(sim.u, sim.v)
        t0 = time.time()
        _advance(sim, s.surface, grains, t_start + duration + t_hold, s.record_dt, rec)
        if verbose:
            print(f"sweep at {rate:g} ({s.drive}): {duration:.0f} + {t_hold:.0f} time units, "
                  f"{time.time() - t0:.0f} s", flush=True)

    a = np.array(rec["a"])
    nodes = np.array([s.model.mode_nodes(x, s.mode) for x in a])
    return dict(t=np.array(rec["t"]), a=a, X=np.array(rec["X"]), nodes=nodes,
                airborne=np.array(rec["airborne"]), w=np.array(rec["w"]), peak=np.array(rec["peak"]),
                sim=sim, t_sweep_start=s.sim.t, t_sweep_end=s.sim.t + (abs(a1 - s.a0) / rate if sweeping else 0.0),
                sweeping=sweeping, mode=s.mode, g=s.g, amplitude=s.amplitude, gamma=s.gamma,
                drive=s.drive, rate=rate if sweeping else 0.0, model=s.model)


# ------------------------------------------------------------------ statistics
def distance_to_still(X, still):
    """Distance from each grain to the nearest line that does not move.

    X     : positions, shape (n, ndim)
    still : tuple with one array of line positions per coordinate
    """
    d = [np.min(np.abs(X[:, k][:, None] - np.asarray(lines)[None, :]), axis=1)
         for k, lines in enumerate(still)]
    return np.min(d, axis=0)


def node_statistics(res, samples=20000, seed=1):
    """Mean distance from a grain to the nearest still line (nodes and
    clamped edges) over time, and the value evenly scattered grains would
    give for the same lines."""
    model = res["model"]
    rng = np.random.default_rng(seed)
    hi = np.array([model.L] if model.ndim == 1 else [model.Lx, model.Ly])
    U = hi * rng.random((samples, model.ndim))
    mean_d, uniform_d = [], []
    for X, a in zip(res["X"], res["a"]):
        still = model.still_points(a, res["mode"])
        mean_d.append(distance_to_still(X, still).mean())
        uniform_d.append(distance_to_still(U, still).mean())
    return np.array(mean_d), np.array(uniform_d)


def quiet_half_width(model, a, mode, gamma):
    """Half-width of the quiet zone around the first interior node: the
    stretch where the surface acceleration is below g, so grains cannot
    lift off.  Measured along x through the strongest row of the mode."""
    phi = model.mode_shape(a, mode)
    if model.ndim == 2:
        phi = phi[:, int(np.argmax(np.sum(phi * phi, axis=0)))]
    rel = np.abs(phi) / np.max(np.abs(phi))
    node = model.mode_nodes(a, mode)[0]
    quiet = rel < 1.0 / gamma
    i = int(np.argmin(np.abs(model.x - node)))
    lo = hi = i
    while lo > 0 and quiet[lo - 1]:
        lo -= 1
    while hi < len(quiet) - 1 and quiet[hi + 1]:
        hi += 1
    return 0.5 * (model.x[hi] - model.x[lo])


def node_following(res, window=0.3):
    """How the grain bands sit relative to the moving nodes.

    For each interior node and each recorded time, takes the grains within
    ``window`` * (node spacing) of the node and returns

        offset : mean grain position minus node position, shape (times, nodes)
        count  : number of grains in that window,          shape (times, nodes)

    On the membrane only the middle half in y is used, where the nodal lines
    are well defined.
    """
    model, nodes = res["model"], res["nodes"]
    half = window * float(np.min(np.diff(model.still_points(res["a"][0], res["mode"])[0])))
    offset = np.full(nodes.shape, np.nan)
    count = np.zeros(nodes.shape, dtype=int)
    for i, (X, nd) in enumerate(zip(res["X"], nodes)):
        x = X[:, 0]
        if model.ndim == 2:
            x = x[np.abs(X[:, 1] - 0.5 * model.Ly) < 0.25 * model.Ly]
        for k, xn in enumerate(nd):
            near = x[np.abs(x - xn) < half]
            count[i, k] = near.size
            if near.size:
                offset[i, k] = near.mean() - xn
    return offset, count, half


def following_summary(res, window=0.3, average_over=100.0):
    """Headline numbers for one sweep.

    Two views of the same question:

    * bands: the grains found near each node at a given time (see
      ``node_following``).  Says where the band sits, but not how many
      grains fell out of it.
    * members: the grains that were near a node when the sweep started,
      followed individually.  ``member_trailing`` is how far behind their
      node the typical (median) one is, with no cut-off; ``kept`` is the
      share still within the window of their own node, and
      ``at_any_node`` the share within the window of any interior node
      (a grain left behind can be picked up by the next node along).

    Trailing distances are positive when grains are behind the node (on the
    side it came from) and are averaged over the interior nodes.
    """
    model, t, nodes = res["model"], res["t"], res["nodes"]
    offset, count, half = node_following(res, window)
    travel = nodes[-1] - nodes[0]
    direction = -np.sign(travel)                            # positive = trailing
    behind = direction[None, :] * offset
    i0 = int(np.searchsorted(t, res["t_sweep_start"])) - 1
    i1 = int(np.searchsorted(t, res["t_sweep_end"]))
    i1 = min(i1, len(t) - 1)

    def band(ia, ib):
        return float(np.nanmean(behind[ia:ib + 1]))

    n_avg = max(1, int(average_over / max(t[1] - t[0], 1e-9)))
    # members: grains near node k at the start of the sweep
    X0 = res["X"][i0]
    keep = np.ones(len(X0), dtype=bool)
    if model.ndim == 2:
        keep = np.abs(X0[:, 1] - 0.5 * model.Ly) < 0.25 * model.Ly
    member_trail, kept, homed = {}, {}, {}
    for label, idx in (("end_of_sweep", i1), ("after_hold", len(t) - 1)):
        trail, inside, at_any, total = [], 0, 0, 0
        for k in range(nodes.shape[1]):
            mem = keep & (np.abs(X0[:, 0] - nodes[i0, k]) < half)
            if mem.any():
                x = res["X"][idx][mem, 0]
                d = direction[k] * (x - nodes[idx, k])
                trail.append(np.median(d))
                inside += int((np.abs(d) < half).sum())
                at_any += int((np.min(np.abs(x[:, None] - nodes[idx][None, :]), axis=1) < half).sum())
                total += int(mem.sum())
        member_trail[label] = float(np.mean(trail)) if trail else float("nan")
        kept[label] = inside / total if total else float("nan")
        homed[label] = at_any / total if total else float("nan")

    during = slice(i0, i1 + 1)
    acc = res["peak"] * res["w"] ** 2
    acc = acc[during][np.isfinite(acc[during])]
    return dict(node_travel=float(np.mean(np.abs(travel))), window=half,
                band_trailing_before=band(max(i0 - n_avg, 0), i0),
                band_trailing_end_of_sweep=band(max(i1 - max(1, min(n_avg, (i1 - i0) // 5)), i0), i1),
                band_trailing_after_hold=band(len(t) - 1 - n_avg, len(t) - 1),
                member_trailing_end_of_sweep=member_trail["end_of_sweep"],
                member_trailing_after_hold=member_trail["after_hold"],
                kept_end_of_sweep=kept["end_of_sweep"], kept_after_hold=kept["after_hold"],
                at_any_node_after_hold=homed["after_hold"],
                acceleration_min=float(acc.min() / acc[0]) if acc.size else float("nan"),
                acceleration_max=float(acc.max() / acc[0]) if acc.size else float("nan"))
