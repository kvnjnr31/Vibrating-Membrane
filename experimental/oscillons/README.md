# Experimental: disk layer on a shaking floor

Separate from the string/membrane code. First step toward oscillons: a layer of inelastic disks (x-z plane) on a flat, vertically shaking floor.

- `layer.py` – soft-disk simulation (needs `numba`)
- `run_flat_floor.py` – runs one drive level and a control, looks for half-frequency waves

Status: parked. The first test was inconclusive (partial half-frequency response at 2 g, disordered at 3.5 g); parameters still need tuning.
