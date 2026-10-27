"""Run the whole DCP:  python run_all.py                (real data, needs internet)
                       python run_all.py --binary       (drop neutral class in the split)
                       python run_all.py --demo         (synthetic data, offline smoke test)"""
import subprocess
import sys

demo = "--demo" in sys.argv
binary = "--binary" in sys.argv
py = sys.executable
steps = [[py, "stage1_collect.py"] + (["--demo"] if demo else []),
         [py, "stage2_clean.py"],
         [py, "stage3_morph.py"] + (["--heuristic"] if demo else []),
         [py, "stage5_split.py"] + (["--binary"] if binary else []),
         [py, "stage4_annotate.py", "make", "--n", "60" if demo else "600"]]
for s in steps:
    print("\n=== ", " ".join(s[1:]), " ===")
    if subprocess.call(s) != 0:
        sys.exit(f"step failed: {s}")
