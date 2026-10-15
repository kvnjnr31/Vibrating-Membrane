"""Draw every figure of the paper and supplement from the data in results/,
and copy them, under their paper names, to ../paper/submission/Figures.

    python reproduce/make_figures.py                  # all figures (about 15 min)
    python reproduce/make_figures.py --only Fig6 FigS6
    python reproduce/make_figures.py --movie          # also render the conveyor movie (Movie S1)
    python reproduce/make_figures.py --no-copy        # leave the paper folder alone

Fig. 1-2 and S1-S4 run their own short simulations (2-3 min each); all other
figures read the data written by run_simulations.py.
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

CODE = pathlib.Path(__file__).resolve().parent.parent
RESULTS = CODE / "results"
sys.path.insert(0, str(CODE / "reproduce"))
from steps import FIG, MOVIE  # noqa: E402


def run(cmd):
    print(f"[run ] python {cmd}", flush=True)
    subprocess.run([sys.executable] + cmd.split(), cwd=CODE, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", default=None, help="figure ids, e.g. Fig6 FigS6")
    p.add_argument("--movie", action="store_true", help="render the conveyor movie as well")
    p.add_argument("--no-copy", action="store_true", help="do not copy into ../paper/submission/Figures")
    p.add_argument("--dest", default=str(CODE.parent / "paper" / "submission" / "Figures"))
    args = p.parse_args()
    want = set(args.only) if args.only else {f[0] for f in FIG}
    # a figure without its own command is written by the figure before it
    owner, done = {}, set()
    last = None
    for fid, cmd, _, _ in FIG:
        last = fid if cmd else last
        owner[fid] = last
    for fid, cmd, src, dst in FIG:
        if fid not in want:
            continue
        lead = owner[fid]
        lead_cmd = next(c for f, c, _, _ in FIG if f == lead)
        if lead not in done:
            run(lead_cmd)
            done.add(lead)
    dest = pathlib.Path(args.dest)
    if not args.no_copy:
        dest.mkdir(parents=True, exist_ok=True)
        for fid, _, src, dst in FIG:
            if fid not in want:
                continue
            for ext in ("png", "pdf"):
                f = RESULTS / f"{src}.{ext}"
                if f.exists():
                    shutil.copyfile(f, dest / f"{dst}.{ext}")
            print(f"[copy] results/{src}.png -> {dest.name}/{dst}.png")
    if args.movie:
        cmd, src, dst = MOVIE
        run(cmd)
        if not args.no_copy:
            shutil.copyfile(RESULTS / src, dest.parent / dst)
            print(f"[copy] results/{src} -> {dest.parent.name}/{dst}")


if __name__ == "__main__":
    main()
