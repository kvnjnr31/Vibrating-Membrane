"""Every step behind the figures, tables and movie of the paper, in one list.

Each step is (id, stage, command, outputs, note).  Stage "sim" runs a
simulation and writes data to results/; stage "fig" draws a figure from data
already in results/ (a few of these also run a short simulation themselves).
Commands are run from the code/ folder.

Used by run_simulations.py and make_figures.py.
"""

# 1050 / n, rounded: gather time of the matched-protocol mode scan (Supplementary Note 3, Table S3)
GATHER = {5: 210, 7: 150, 9: 117, 11: 95, 13: 81, 15: 70, 19: 55}

SIM = [
    # ---- drives on the string (Figs. 3-4, Tables 1-3) -------------------------------------------
    ("loop_string", "forcing/run_main_runs.py loop --mode 5",
     "main_loop_string_mode5.{npz,json}", "Fig. 3, Tables 1-2 (string)"),
    ("membrane_table", "forcing/run_make_table.py --mode 5 1",
     "lookup table for membrane mode (5,1)", "needed by the membrane runs"),
    ("loop_membrane", "forcing/run_main_runs.py loop --mode 5 1 --membrane",
     "main_loop_membrane_mode51.{npz,json}", "Table 2 (membrane column)"),
    ("checks", "forcing/run_main_runs.py checks --mode 5",
     "main_checks_string_mode5.json", "PLL bandwidth, calibration and pull-in numbers in the text"),
    ("rates", "forcing/run_main_runs.py rates --mode 5",
     "main_rates_string_mode5.csv", "Fig. 4, Table 3"),
    # ---- grains on a fixed standing wave (Fig. 5, Fig. S9) -----------------------------------------
    ("accel_sweep", "acceleration_sweep/run_acceleration_sweep.py",
     "accel_sweep.{npz,csv}", "Fig. 5, Fig. S9 (about 1 h on 2 cores)"),
    ("accel_extra", "acceleration_sweep/run_acceleration_extra.py",
     "accel_histograms_extra.npz", "Fig. 5, Fig. S9: rough grains rolling with mu_r = 0.05 and 0.002"),
    # ---- ratchet on the string, mode 5 (Fig. 6, Tables 4-5) ----------------------------------------
    ("grain_sweeps", "particles/run_grain_sweeps.py --mode 5 --a0 -0.13 --a1 0.13 --amplitude 0.029",
     "grain_sweeps_mode5.csv", "Table 4"),
    ("ratchet5_rising", "particles/run_ratchet.py --mode 5 --a0 -0.13 --a1 0.13 --rate 8e-5 --cycles 4 "
     "--control-cycles 2 --grains 3000 --amplitude 0.029 --seed 0 --tag _g3k_fwd_s0",
     "fig_ratchet_mode5_g3k_fwd_s0.npz", "Fig. 6A, C, D, Table 5 (ratchet with a rising, and the control)"),
    ("ratchet5_falling", "particles/run_ratchet.py --mode 5 --a0 0.13 --a1 -0.13 --rate 8e-5 --cycles 4 "
     "--control-cycles 0 --grains 3000 --amplitude 0.029 --seed 0 --tag _g3k_rev_s0",
     "fig_ratchet_mode5_g3k_rev_s0.npz", "Fig. 6B, D, Table 5 (ratchet with a falling)"),
] + [
    # ---- ratchet speed across modes (Fig. 7, Fig. S8, Table S3) ------------------------------------
    (f"modes_{n}", f"particles/run_conveyor_hops.py --mode {n} --quiet 60 --gather {g} --tag _v2",
     f"conveyor_hops_mode{n}_v2.{{csv,npz}}", "Fig. 7, Fig. S8, Table S3")
    for n, g in GATHER.items()
] + [
    # ---- self-regulating membrane conveyor (Figs. 8-10, Tables 6-7, movie) ---------------------------
] + [
    (f"conveyor_d{d}", f"particles/adjustable_conveyor.py run --d {d} --tag _ret",
     f"adjustable_conveyor_ret_d{d}.npz", "Figs. 8-10, Tables 6-7" + (", movie" if d == 6 else "") + " (about 1-2 h each)")
    for d in (2, 4, 6, 8)
] + [
    (f"conveyor_d{d}_seed1", f"particles/adjustable_conveyor.py run --d {d} --tag _ret_s1 --seed 1",
     f"adjustable_conveyor_ret_s1_d{d}.npz", "second random seed: noise check in the model section")
    for d in (6, 8)
] + [
    # ---- supplement: modes 3, 7 and 11 (Figs. S5-S7, Table S2) -------------------------------------
    ("grains_mode3", "particles/run_grains.py",
     "string_grains.npz", "Fig. S5 (mode 3, 400 grains, PLL with amplitude loop)"),
    ("grains_mode3_rates", "particles/run_grain_rate_scan.py",
     "string_grain_rate_scan.csv", "Table S2"),
] + [
    (f"ratchet11_{d}_s{s}",
     f"particles/run_ratchet.py --mode 11 --a0 {a0} --a1 {a1} --rate 8.5e-5 --cycles 5 --grains 500 "
     f"--amplitude 0.014 --seed {s} --tag _{d}" + (f"_s{s}" if s else ""),
     f"fig_ratchet_mode11_{d}" + (f"_s{s}" if s else "") + ".npz", "Figs. S6-S7 (seed 0); seeds 1-2 for the spread")
    for d, a0, a1 in (("forward", -0.064, 0.064), ("reverse", 0.064, -0.064)) for s in (0, 1, 2)
]

# figure id, command, file written to results/, name in paper/submission/Figures
FIG = [
    ("Fig1", "forcing/run_paper_figures.py --mode 5", "fig_drives_mode5", "Fig1"),
    ("Fig2", None, "fig_snapshots_mode5", "Fig2"),                       # written by the Fig1 command
    ("Fig3", "forcing/run_main_figures.py --mode 5 --only loop", "fig_loop_mode5", "Fig3"),
    ("Fig4", "forcing/run_main_figures.py --mode 5 --only rate", "fig_rate_mode5", "Fig4"),
    ("Fig5", "acceleration_sweep/run_acceleration_density.py", "Fig5", "Fig5"),
    ("Fig6", "particles/run_ratchet_figures.py --mode 5 --set _g3k", "fig_ratchet_conveyor_mode5_g3k", "Fig6"),
    ("Fig7", "particles/run_conveyor_band_figure.py", "Fig7", "Fig7"),
    ("Fig8", "particles/run_conveyor_snapshots.py --d 6 --cycle 30", "fig_conveyor_snapshots", "Fig8"),
    ("Fig9", "particles/run_reduced_model_figures.py", "Fig9", "Fig9"),
    ("Fig10", None, "Fig10", "Fig10"),                                   # written by the Fig9 command
    ("FigS1", "forcing/run_paper_figures.py --mode 3", "fig_drives_mode3", "FigS1"),
    ("FigS2", None, "fig_snapshots_mode3", "FigS2"),
    ("FigS3", "forcing/run_paper_figures.py --mode 7", "fig_drives_mode7", "FigS3"),
    ("FigS4", None, "fig_snapshots_mode7", "FigS4"),
    ("FigS5", "particles/run_grain_figures.py", "fig_string_grains", "FigS5"),
    ("FigS6", "particles/run_ratchet_figures.py --mode 11 --names forward reverse", "fig_ratchet_conveyor_mode11", "FigS6"),
    ("FigS7", None, "fig_ratchet_mechanism_mode11", "FigS7"),
    ("FigS8", "particles/run_conveyor_modes_figure.py", "fig_conveyor_modes", "FigS8"),
    ("FigS9", "acceleration_sweep/run_acceleration_density.py --raw", "fig_accel_raw", "FigS9"),
]

MOVIE = ("particles/adjustable_conveyor.py movie --d 6 --tag _ret --crf 32", "adjustable_conveyor_ret_d6.mp4", "Movie_S1.mp4")
