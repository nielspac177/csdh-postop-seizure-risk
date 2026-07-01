"""run_all.py — DAG-ordered, reproducible execution of the full analysis pipeline.

This is the single entry point that reproduces every result in the manuscript from
the raw data files (`csdh_clean.csv`, `eicu_csdh_cohort_final.csv`) on a fresh clone.
It supersedes the partial ordering in the Makefile by encoding the real dependency
DAG, running each stage with the interpreter that launched this script, checking that
each script's declared outputs exist and are non-empty, and writing a timestamped log.

Design notes
------------
* Reproducibility: every numerical script already sets SEED=42 and N_JOBS=1 (Apple
  Silicon thread-safety). This runner adds nothing stochastic of its own.
* Cache-awareness: stages that only consume cached OOF predictions (calibration,
  conformal, DCA) come after the modeling stages that produce those caches.
* Graceful degradation: stages whose source data are not redistributable (NIS, the
  radiology NLP corpus) are marked `optional=True` — a missing-data failure is logged
  and skipped rather than aborting the whole pipeline.

Usage
-----
    python scripts/run_all.py                 # run the whole pipeline
    python scripts/run_all.py --list          # print the DAG and exit
    python scripts/run_all.py --dry-run       # show what would run, run nothing
    python scripts/run_all.py --only models   # run a single stage
    python scripts/run_all.py --from 25_conformal_prediction   # resume from a script
    python scripts/run_all.py --skip-docs     # skip document/figure builders
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPTS_DIR.parent / "results"
LOG_DIR = SCRIPTS_DIR.parent / "logs"

# ──────────────────────────────────────────────────────────────────────────
# The pipeline DAG.  Each stage is a list of (script, [expected_output_globs]).
# Stages run in declared order; scripts within a stage run in declared order.
# `optional` stages are skipped (not failed) when their script exits non-zero
# because the underlying data are not redistributable.
# ──────────────────────────────────────────────────────────────────────────
Stage = dict  # {name, scripts: [(file, [globs])], optional: bool, is_doc: bool}


def _stage(name, scripts, optional=False, is_doc=False) -> Stage:
    return {"name": name, "scripts": scripts, "optional": optional, "is_doc": is_doc}


PIPELINE: list[Stage] = [
    _stage("modeling", [
        ("05_temporal_leakage.py",  ["05_leakage_audit.csv"]),
        ("18_bidmc_optimize.py",    ["18_bidmc_optimized.csv"]),
        ("21_imbalance_sweep.py",   ["21_imbalance_sweep.csv"]),
        ("22_diverse_stacking.py",  ["22_diverse_stacking.csv"]),
        ("23_tabpfn_eval.py",       ["23_tabpfn.csv"]),
        ("24_firth_bayes_lr.py",    ["24_firth_bayes_lr.csv"]),
        ("19_transfer_learning.py", ["19_transfer_learning.csv"]),
        ("38b_firth_oof_refit.py",  []),  # writes OOF caches consumed downstream
    ]),
    _stage("validation", [
        ("02_calibration.py",   ["02_calibration_metrics.csv"]),
        ("03_dca.py",           ["03_dca_summary_at_thresholds.csv"]),
        ("04_loho.py",          ["04_loho_summary.csv"]),
        ("06_overfitting.py",   ["06_overfitting_metrics.csv"]),
        ("07_missing_data.py",  ["07_imputation_comparison.csv"]),
        ("08_eicu_cohort.py",   ["08_cohort_comparison.csv"]),
        ("09_competing_risks.py", ["09_competing_risks.csv"]),
    ]),
    _stage("conformal", [
        ("25_conformal_prediction.py", ["25_conformal.csv"]),
    ]),
    _stage("nis", [
        ("12_nis_seizure_reclassify.py", ["12_nis_seizure_codes.csv"]),
        ("13_nis_grouped_lasso.py",      ["13_nis_grouped_lasso.csv"]),
    ], optional=True),
    _stage("radiology", [
        ("15_radiology_nlp.py", []),
    ], optional=True),
    _stage("cea", [
        ("10_11_cea_pairwise.py", ["10_pairwise_summary.csv"]),
        ("14_decision_tree.py",   ["14_decision_tree_rollback.csv"]),
    ]),
    _stage("voi", [
        ("16_voi_evpi.py", ["16_voi_evpi.csv"]),
    ]),
    _stage("deployable", [
        ("38_deployable_model_reanalysis.py", ["38_postopB_operating_points.csv",
                                               "38_message_stability_summary.csv"]),
        ("39_aed_harm_threshold.py",          ["39_aed_harm_threshold.csv"]),
    ]),
    _stage("figures", [
        ("26_main_figures.py",      []),
        ("29_main_figures_jnnp.py", []),
        ("34_graphical_abstract.py", []),
    ], is_doc=True),
    _stage("exports", [
        ("28_make_callgraph.py",            []),
        ("30_export_calculator_assets.py",  []),
        ("31_export_callgraph_json.py",     []),
    ], is_doc=True),
    _stage("documents", [
        ("27_build_jnnp_manuscript.py",   []),
        ("35_build_submission_package.py", []),
        ("32_build_code_companion.py",    []),
        ("33_build_code_companion_pdf.py", []),
        ("36_build_talk_slides.py",       []),
    ], is_doc=True),
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _check_outputs(globs: list[str]) -> list[str]:
    """Return the list of expected outputs that are missing or empty."""
    missing = []
    for g in globs:
        hits = list(RESULTS_DIR.glob(g))
        if not hits or all(h.stat().st_size == 0 for h in hits):
            missing.append(g)
    return missing


def run_script(script: str, log) -> tuple[bool, float]:
    """Run one script with the current interpreter; return (ok, seconds)."""
    path = SCRIPTS_DIR / script
    if not path.exists():
        log(f"    ✗ MISSING SCRIPT: {script}")
        return False, 0.0
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(path)],
                          cwd=SCRIPTS_DIR.parent,
                          capture_output=True, text=True)
    dt = time.time() - t0
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    if proc.returncode != 0:
        log(f"    ✗ {script} exited {proc.returncode} ({dt:.1f}s)")
        log("    stderr tail:\n" + "\n".join(proc.stderr.strip().splitlines()[-8:]))
        return False, dt
    log(f"    ✓ {script} ({dt:.1f}s)  {tail.splitlines()[-1] if tail else ''}")
    return True, dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the DAG and exit")
    ap.add_argument("--dry-run", action="store_true", help="show what would run")
    ap.add_argument("--only", metavar="STAGE", help="run only this stage")
    ap.add_argument("--from", dest="from_script", metavar="SCRIPT",
                    help="resume from this script (basename, with or without .py)")
    ap.add_argument("--skip-docs", action="store_true",
                    help="skip figure/export/document builder stages")
    args = ap.parse_args()

    if args.list:
        for st in PIPELINE:
            flags = " [optional]" if st["optional"] else ""
            flags += " [docs]" if st["is_doc"] else ""
            print(f"{st['name']}{flags}")
            for scr, outs in st["scripts"]:
                print(f"    {scr}" + (f"  → {', '.join(outs)}" if outs else ""))
        return 0

    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_all_{stamp}.log"
    log_fh = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"run_all.py — pipeline start {_now()}")
    log(f"interpreter: {sys.executable}")
    log(f"log file:    {log_path}\n")

    started = bool(args.from_script is None)
    from_norm = (args.from_script or "").removesuffix(".py")
    total_t0 = time.time()
    failures, skipped = [], []

    for st in PIPELINE:
        if args.only and st["name"] != args.only:
            continue
        if args.skip_docs and st["is_doc"]:
            log(f"[stage {st['name']}] skipped (--skip-docs)")
            continue
        log(f"[stage {st['name']}]" + (" (optional)" if st["optional"] else ""))
        for scr, outs in st["scripts"]:
            if not started:
                if scr.removesuffix(".py") == from_norm:
                    started = True
                else:
                    log(f"    · skip {scr} (before --from)")
                    continue
            if args.dry_run:
                log(f"    · would run {scr}")
                continue
            ok, _ = run_script(scr, log)
            if ok and outs:
                missing = _check_outputs(outs)
                if missing:
                    log(f"    ! {scr} ran but outputs missing/empty: {missing}")
                    ok = False
            if not ok:
                if st["optional"]:
                    log(f"    → skipping optional stage failure ({scr})")
                    skipped.append(scr)
                else:
                    failures.append(scr)

    dt = time.time() - total_t0
    log(f"\npipeline finished in {dt/60:.1f} min — "
        f"{len(failures)} failure(s), {len(skipped)} optional skip(s)")
    if failures:
        log("FAILURES: " + ", ".join(failures))
    log_fh.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
