"""Step 1: do disks on a flat shaking floor form half-frequency waves?

    python run_flat_floor.py                 # gamma 3.5, with a gamma 2.0 control
    python run_flat_floor.py --gamma 2.6 --depth 10 --periods 150

Writes results/flat_floor_gamma<g>.{png,pdf} and prints, for each run, the
pattern amplitude, its correlation one and two drive periods apart (a
half-frequency wave gives about -1 and +1), the frequency of the strongest
spatial mode and its wavelength.
"""
import argparse
import pathlib

import numpy as np

from layer import Layer, surface_profile

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
FPP = 16                                      # frames per drive period


def analyse(layer, t, x, h, bins):
    prof = surface_profile(x, h, layer.m, layer.W, bins)
    pat = prof - prof.mean(axis=1, keepdims=True)
    periods = len(t) // FPP
    amp_phase = np.array([pat[p::FPP].std() for p in range(FPP)])
    p = int(np.argmax(amp_phase))                       # phase where the pattern is largest
    strobe = pat[p::FPP][:periods]
    norm = np.mean(strobe ** 2)
    c1 = np.mean(strobe[:-1] * strobe[1:]) / norm
    c2 = np.mean(strobe[:-2] * strobe[2:]) / norm
    spec = np.abs(np.fft.rfft(strobe, axis=1)) ** 2
    k = int(np.argmax(spec.mean(axis=0)[1:])) + 1
    mode_t = np.fft.rfft(pat, axis=1)[:, k]             # that spatial mode at every frame
    sig = (mode_t * np.conj(mode_t[0]) / abs(mode_t[0])).real
    ft = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freq = np.fft.rfftfreq(len(sig), 1.0 / FPP)         # in units of the drive frequency
    return dict(prof=prof, pat=pat, strobe=strobe, phase=p, amplitude=float(np.sqrt(norm)),
                c1=float(c1), c2=float(c2), k=k, wavelength=layer.W / k,
                f_peak=float(freq[np.argmax(ft[1:]) + 1]), freq=freq, ft=ft)


def run(gamma, args):
    layer = Layer(width=args.width, depth=args.depth, frequency=args.frequency, gamma=gamma,
                  restitution=args.restitution, friction=args.friction, seed=args.seed)
    layer.run(args.periods - args.keep)                 # discard the transient
    t, x, h = layer.run(args.keep, FPP)
    out = analyse(layer, t, x, h, args.bins)
    out.update(layer=layer, t=t, x=x, h=h, gamma=gamma)
    print(f"gamma {gamma:.2f}: {layer.n} disks, floor amplitude {layer.A:.2f} d, pattern amplitude "
          f"{out['amplitude']:.2f} d, correlation after 1 period {out['c1']:+.2f}, after 2 "
          f"{out['c2']:+.2f}, strongest mode at {out['f_peak']:.2f} of the drive frequency, "
          f"wavelength {out['wavelength']:.1f} d", flush=True)
    return out


def figure(main, control, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200,
                         "pdf.fonttype": 42})
    L = main["layer"]
    fig = plt.figure(figsize=(12.5, 10.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.8, 0.8, 1.5, 1.1], hspace=0.55, wspace=0.2,
                          left=0.07, right=0.98, top=0.96, bottom=0.06)
    p = main["phase"]
    for row, (shift, tag) in enumerate(((0, "A"), (FPP, "B"))):
        ax = fig.add_subplot(gs[row, :])
        k = len(main["t"]) - 2 * FPP + p + shift
        k = min(k, len(main["t"]) - 1)
        ax.scatter(main["x"][k], main["h"][k], s=9, color="0.4", linewidths=0, rasterized=True)
        ax.axhline(0, color="k", linewidth=1.5)
        ax.set_xlim(0, L.W)
        ax.set_ylim(-1, 3.2 * L.depth)
        ax.set_aspect("equal")
        ax.set_ylabel("height")
        ax.set_title(f"{tag}: Disks at the Same Drive Phase, "
                     f"{'One Period Later' if shift else 'Reference Period'}  "
                     f"(peak acceleration {main['gamma']:g} g)")
    ax.set_xlabel("x  (disk diameters)")
    for col, (res, tag) in enumerate(((main, "C"), (control, "D"))):
        ax = fig.add_subplot(gs[2, col])
        n = min(24, len(res["strobe"]))
        lim = max(np.abs(main["strobe"]).max(), 1e-9)
        ax.imshow(res["strobe"][-n:], aspect="auto", origin="lower", cmap="Greys",
                  extent=[0, L.W, 0, n], vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xlabel("x  (disk diameters)")
        ax.set_ylabel("drive period")
        ax.set_title(f"{tag}: Layer Height Once per Period, {res['gamma']:g} g")
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    ax = fig.add_subplot(gs[3, :])
    ax.plot(main["freq"], main["ft"] / main["ft"].max(), color="k", linewidth=1.8,
            label=f"{main['gamma']:g} g")
    ax.plot(control["freq"], control["ft"] / main["ft"].max(), color="0.55", linewidth=1.8,
            linestyle=(0, (4, 3)), label=f"{control['gamma']:g} g (control)")
    ax.axvline(0.5, color="0.8", linewidth=0.9, zorder=0)
    ax.set_xlim(0, 2.5)
    ax.set_xlabel("frequency / drive frequency")
    ax.set_ylabel("amplitude")
    ax.set_title("E: Temporal Spectrum of the Strongest Spatial Mode")
    ax.legend(loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gamma", type=float, default=3.5)
    p.add_argument("--control", type=float, default=2.0)
    p.add_argument("--width", type=float, default=120.0)
    p.add_argument("--depth", type=float, default=8.0)
    p.add_argument("--frequency", type=float, default=0.14, help="in sqrt(g/d)")
    p.add_argument("--restitution", type=float, default=0.6)
    p.add_argument("--friction", type=float, default=0.5)
    p.add_argument("--periods", type=int, default=120)
    p.add_argument("--keep", type=int, default=64, help="periods analysed at the end")
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    a = run(args.gamma, args)
    b = run(args.control, args)
    path = RESULTS / f"flat_floor_gamma{args.gamma:g}"
    figure(a, b, path)
    print(f"saved {path}.png")


if __name__ == "__main__":
    main()
