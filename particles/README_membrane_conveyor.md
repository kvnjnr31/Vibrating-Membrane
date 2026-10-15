# Membrane conveyor scripts

All scripts run from the `code` folder, for example `python particles/adjustable_conveyor.py run --d 8`. They need `numba` and, for movies, `ffmpeg`. The membrane look-up table must exist first: `python forcing/run_make_table.py --mode 5 1`.

| Script | What it does |
|---|---|
| `roll_grains.py` | Grain model on the membrane: rough grains that roll in contact (roughness 0.3, rolling resistance 0.05). Used by every script below. |
| `run_membrane_roll_sweep.py` | One tension sweep with these grains. Writes `fig_membrane_roll_sweep_mode51`. |
| `run_membrane_conveyor.py` | Ratchet conveyor with a feeder and an offload slot. Options: `--reverse`, `--control`, `--back`, `--feed-box`, `--offload`, `--slow-window`. |
| `run_membrane_movie.py` | Movie of a conveyor run (top view and side view). |
| `run_membrane_conveyor_figure.py` | Snapshots and fed/offloaded counts, ratchet against control. |
| `run_membrane_attractors.py` | Each nodal line as a path in (slope, antinode acceleration, grain count): movie and still figure. |
| `run_membrane_sweep_space.py` | Sweep portions of the three nodal-line counts plotted against each other. |
| `run_membrane_direction_figure.py` | Offload for slope rising, slope falling and the control. |
| `run_membrane_layout_figure.py` | Offload for three layouts of feeder, wall and slot. |
| `run_membrane_feedback.py`, `run_membrane_feedback_figure.py` | Feedback on feed rate and sweep rate to hold a set grain count; path in (input rate, sweep rate, output rate). |
| `adjustable_conveyor.py` | The adjustable conveyor: wall at the left edge, slot at the right edge, feeder at distance `d` from the slot, switched on or off by the grain count after each sweep. Commands: `run`, `figure`, `movie`, `embed`. |

Large frame files (`membrane_conveyor*.npz`, `adjustable_conveyor_d*.npz`) are not kept in this folder; rerun the scripts to rebuild them. The light files (`*_series.npz`, `*_stats.npz`, `.csv`) hold everything the figures need except the movies.
