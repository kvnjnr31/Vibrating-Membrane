"""Quick regression checks.

MATLAB ports: reference numbers come from running the original .m files in
GNU Octave 8.4 (see matlab_ports/README.md).
Forcing and particles: checks against the eigenmode predictions.  Run from anywhere:

    python tests/run_checks.py
"""
import pathlib
import subprocess
import sys

CODE = pathlib.Path(__file__).resolve().parent.parent
PORTS = CODE / "matlab_ports"

CHECKS = {
    "membrane_1D (250 steps vs Octave)": (PORTS, """
import numpy as np, membrane_1D as m
log = {}
m.simulate(250, plot=False, record=lambda n, u, B, vyb, jj: log.update(B=B[1], v=vyb, nrm=np.linalg.norm(u[1]), k=jj.size))
ref = (-1.28830775, -1.49078766, 22.02530496, 43)
assert abs(log['B'] - ref[0]) < 1e-7 and abs(log['v'] - ref[1]) < 1e-7, log
assert abs(log['nrm'] - ref[2]) < 1e-7 and log['k'] == ref[3], log
"""),
    "reaction_fun (N = 5)": (PORTS, """
import reaction_fun as r
f = next(r.frames(1))
assert abs(f['Cl'][0] - 3.6339745962155616) < 1e-12 and abs(f['Cr'][0] - 5.366025403784438) < 1e-12
assert abs(f['F'][1] - 1.7320508075688772) < 1e-12
"""),
    "mass_spring_sim/reaction_fun (N = 50)": (PORTS / "mass_spring_sim", """
import reaction_fun as r
f = next(r.frames(1))
assert abs(f['Cl'][0] - 3.6339745962155616) < 1e-12 and abs(f['F'][0] - 0.006141848493044113) < 1e-12
"""),
    "old_functions/membrane_1D_old (impact at step 155, stops at 168)": (PORTS / "old_functions", """
import numpy as np, membrane_1D_old as m
log = []
try:
    m.simulate(210, plot=False, verbose=False, record=lambda n, u, B, vyb, s: log.append((n, s, B[1], np.linalg.norm(u[1]))))
    raise SystemExit('expected the run to stop at contact release')
except RuntimeError:
    pass
assert len(log) == 167 and [s for _, s, _, _ in log].index(1) == 154
assert abs(log[-1][2] - (-0.01761468)) < 1e-7 and abs(log[-1][3] - 0.07172319) < 1e-7
"""),
    "old_functions helpers": (PORTS / "old_functions", """
import numpy as np, tangent_fun, reaction_fun, bead_segment_force, Forces_old
xs, ys = tangent_fun.tangent_fun(10.0, 3.0, [1.0, 2.0], 2.0)
assert abs(xs - 2.9411764705882355) < 1e-12 and abs(ys + 0.764705882352942) < 1e-12
assert abs(reaction_fun.main()['dot'] - 49.96498824693282) < 1e-9
"""),
    "forcing: discrete eigenmodes satisfy the equations (string and membrane)": (CODE, """
import numpy as np
from forcing.string_model import TensionString
from forcing.membrane_model import TensionMembrane
s = TensionString(); phi = s.mode_shape(-0.1, 3); lam = s.eigenvalue(-0.1, 3)
assert np.abs(s.accel(phi, 0 * phi, -0.1, 0.0) + lam * phi).max() < 1e-9
m = TensionMembrane(Nx=41, Ny=29); phi = m.mode_shape(-0.1, (3, 1)); lam = m.eigenvalue(-0.1, (3, 1))
assert np.abs(m.accel(phi, 0 * phi, -0.1, 0.0) + lam * phi).max() < 1e-9
assert abs(m.resonance(0.0, (3, 1)) - np.sqrt((3 * np.pi / 10) ** 2 + (np.pi / 7) ** 2)) < 5e-3
"""),
    "forcing: Bessel solution matches the grid modes of the string": (CODE, """
import numpy as np
from forcing import bessel
from forcing.string_model import TensionString
s = TensionString(N=301, beta=0.015)
assert abs(bessel.eigenvalue(0.18, 3) - 0.63337) < 1e-5          # thesis value
assert abs(bessel.eigenvalue(0.0, 5) - (5 * np.pi / 10) ** 2) < 1e-12
for a in (-0.18, -0.07, 0.11):
    lb, ld = bessel.eigenvalue(a, 3), s.eigenvalue(a, 3)
    assert abs(np.sqrt(ld) - np.sqrt(lb)) < 0.005 * s.beta, (a, lb, ld)
    assert np.abs(bessel.nodes(a, 3) - s.mode_nodes(a, 3)).max() < 5e-4
    pb, pd = bessel.mode_shape(s.x, a, 3), s.mode_shape(a, 3)
    assert np.abs(pb - pd).max() < 1e-3 * np.abs(pb).max()
"""),
    "forcing: table drive keeps the string mode through a sweep, fixed drive loses it": (CODE, """
import numpy as np
from forcing.string_model import TensionString
from forcing.simulation import Simulation, make_drive, steady_state_energy
from forcing import schedules
s = TensionString(N=151, beta=0.05)
sch = schedules.linear_ramp(-0.15, -0.10, 300.0, 2e-4)
out = {}
for kind in ('table', 'fixed'):
    sim = Simulation(s, make_drive(kind, s, 3, sch), sch, 3, dt=0.04).run(650.0)
    out[kind] = (sim.summary(t_from=300.0), sim.history[-1]['energy'])
target = steady_state_energy(s, -0.10, s.resonance(-0.10, 3))
summ, energy = out['table']
assert abs(energy / target - 1) < 0.02, (energy, target)
assert summ['min_purity'] > 0.999 and summ['max_node_error'] < 0.03, summ
assert out['fixed'][1] < 0.5 * target, out['fixed']
"""),
    "forcing: amplitude gain and PLL drive through a sweep (no settling run)": (CODE, """
import numpy as np
from forcing.string_model import TensionString
from forcing.simulation import Simulation, make_drive, steady_state_energy
from forcing import schedules
s = TensionString(N=151, beta=0.05)
sch = schedules.linear_ramp(-0.15, -0.05, 50.0, 4e-4)
def run(kind, **kw):
    sim = Simulation(s, make_drive(kind, s, 3, sch, **kw), sch, 3, dt=0.04)
    sim.start_in_steady_state().run(450.0)
    return sim
target = steady_state_energy(s, -0.05, s.resonance(-0.05, 3))
plain = run('table')
assert abs(plain.history[0]['energy'] / steady_state_energy(s, -0.15, s.resonance(-0.15, 3)) - 1) < 0.01
drift = plain.series('peak'); drift = drift.min() / drift[0]
comp = run('table+gain').series('peak'); comp = (comp.min() / comp[0], comp.max() / comp[0])
assert drift < 0.9 and comp[0] > 0.98 and comp[1] < 1.02, (drift, comp)
pll = run('pll')
summ = pll.summary(t_from=50.0)
assert summ['min_purity'] > 0.995 and abs(pll.history[-1]['energy'] / target - 1) < 0.05, summ
hyb = run('pll+gain'); h = hyb.series('peak'); h = (h.min() / h[0], h.max() / h[0])
assert h[0] > 0.97 and h[1] < 1.03 and hyb.summary(t_from=50.0)['min_purity'] > 0.995, h
agc = run('pll+agc', agc_start=20.0).series('peak'); agc = (agc.min() / agc[0], agc.max() / agc[0])
assert agc[0] > drift + 0.02 and agc[1] < 1.03, (agc, drift)   # the slow loop only helps
"""),
    "particles: grains gather at the nodes and a sweep can reuse the settled state": (CODE, """
import numpy as np
from forcing.string_model import TensionString
from particles.surface import StringSurface
from particles.experiment import settle, sweep, node_statistics, following_summary
m = TensionString(N=151, beta=0.05)
st = settle(m, StringSurface(m), 3, -0.15, drive='pll+agc', n_grains=200, t_settle=500.0,
            dt=0.04, verbose=False, grain_kwargs=dict(roughness=0.3))
res = sweep(st, None)
X = res['X'][-1]
assert np.isfinite(X).all() and X.min() >= 0.0 and X.max() <= m.L
mean_d, uniform_d = node_statistics(res)
assert mean_d[-1] < 0.75 * uniform_d[-1], (mean_d[-1], uniform_d[-1])
t_before = st.sim.t
out = sweep(st, -0.10, 1e-3, t_hold=50.0, verbose=False)
assert st.sim.t == t_before and out['t'][-1] > t_before + 99.0     # settled state untouched
s = following_summary(out)
assert 0.9 < s['acceleration_min'] and s['acceleration_max'] < 1.1, s
"""),
}

failed = 0
for name, (cwd, src) in CHECKS.items():
    p = subprocess.run([sys.executable, "-c", src], cwd=cwd, capture_output=True, text=True)
    ok = p.returncode == 0
    failed += not ok
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        print(p.stderr.strip() or p.stdout.strip())
sys.exit(1 if failed else 0)
