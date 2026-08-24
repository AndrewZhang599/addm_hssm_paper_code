"""Pooled multi-session loader for the hierarchical aDDM fits.

`prepare_monkey_data` is lifted verbatim from cell 3 of `monkey_data_recovery.ipynb`
so the single-session and hierarchical fits see identical trial frames.  The only
addition here is `load_monkey`, which applies it *per session* (it re-references
fixation onsets to that session's own `firstFixLat`, so it must never see two
sessions at once) and carries the `session` label through as a grouping factor.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

MOTOR_DELAY = 0.100  # s -- the decision terminates this long before liftRT

DEFAULT_ROOT = (
    "/users/azhan378/data/azhang/addm_hssm_paper_code/monkey_data/two_monkey_42_session"
)


def prepare_monkey_data(behav, motor_delay=MOTOR_DELAY):
    """Monkey behaviour table -> the aDDM frame.

    Verbatim from `monkey_data_recovery.ipynb` cell 3 -- do not drift from it.
    """
    b = behav.sort_values("trialNum").reset_index(drop=True)

    t0 = b["firstFixLat"].to_numpy(float)                    # clock origin
    rt = (b["liftRT"].to_numpy(float) - motor_delay) - t0    # decision time

    # fixation onsets, re-referenced to the first fixation
    onsets = np.column_stack(
        [t0, b["secondFixLat"].to_numpy(float), b["thirdFixLat"].to_numpy(float)]
    ) - t0[:, None]
    onsets[:, 0] = 0.0

    # *** the truncation ***  only fixations that begin before the (already
    # truncated) decision time are part of the accumulation; the rest are gone,
    # and d shrinks with them.
    keep = np.isfinite(onsets) & (onsets < rt[:, None])
    keep[:, 0] = True
    sacc = np.where(keep, onsets, 0.0)                       # zero-pad past d

    out = pd.DataFrame(
        {
            "rt": rt,
            "response": np.where(b["leftChosen"].to_numpy(int) == 1, 1.0, -1.0),
            "r1": b["lVal"].to_numpy(float),                 # left  = item 1
            "r2": b["rVal"].to_numpy(float),                 # right = item 2
            "flag": np.where(b["firstIsLeft"].to_numpy(int) == 1, 0, 1).astype(int),
            "d": keep.sum(1).astype(int),
            "sigma": 1.0,
        }
    )
    out["sacc_array"] = pd.Series([tuple(row) for row in sacc], index=out.index)
    return out


def session_files(monkey_dir, n_sessions=None, sessions=None):
    """The CSVs to pool, in a stable (filename-sorted) order.

    `sessions` (explicit names, without .csv) wins over `n_sessions` (first N).
    Subsetting is always by whole session -- a partial session would hand the
    random effect a group with an arbitrary trial count.
    """
    monkey_dir = str(monkey_dir).rstrip("/")
    files = sorted(glob.glob(os.path.join(monkey_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {monkey_dir!r}")

    if sessions:
        wanted = set(sessions)
        files = [f for f in files if Path(f).stem in wanted]
        missing = wanted - {Path(f).stem for f in files}
        if missing:
            raise ValueError(f"Sessions not found in {monkey_dir!r}: {sorted(missing)}")
    elif n_sessions is not None:
        if n_sessions < 1:
            raise ValueError(f"--n-sessions must be >= 1, got {n_sessions}")
        files = files[:n_sessions]
    return files


def load_monkey(monkey_dir, n_sessions=None, sessions=None):
    """Pool one monkey's session CSVs into a single aDDM frame.

    One monkey per call, by design -- the hierarchical model treats sessions as
    the only grouping level, so mixing two monkeys here would silently pool them.

    Parameters
    ----------
    monkey_dir
        Path to a single monkey folder, e.g. ``.../two_monkey_42_session/monkey_c``.
    n_sessions
        Keep only the first N sessions (filename order).  ``None`` keeps all.
    sessions
        Explicit list of session names (CSV stems); overrides ``n_sessions``.

    Returns
    -------
    DataFrame with the aDDM columns plus a categorical ``session`` column.
    """
    frames = []
    for f in session_files(monkey_dir, n_sessions=n_sessions, sessions=sessions):
        b = pd.read_csv(f).sort_values("trialNum").reset_index(drop=True)
        out = prepare_monkey_data(b)
        # prepare_monkey_data sorts by trialNum and resets the index the same way,
        # so this row order matches.
        out["session"] = b["session"].to_numpy()
        frames.append(out)

    df = pd.concat(frames, ignore_index=True)
    df["session"] = df["session"].astype("category")  # bambi grouping factor
    return df


def monkey_label(monkey_dir):
    """'monkey_c' -> 'c'; anything else -> the directory basename."""
    base = Path(str(monkey_dir).rstrip("/")).name
    return base[len("monkey_"):] if base.startswith("monkey_") else base


def describe(df):
    """One-line-per-field summary used by the runner's log and by the notebook."""
    g = df.groupby("session", observed=True)
    return {
        "n_trials": int(len(df)),
        "n_sessions": int(df["session"].nunique()),
        "sessions": list(map(str, df["session"].cat.categories)),
        "trials_per_session": {str(k): int(v) for k, v in g.size().items()},
        "rt_min": float(df["rt"].min()),
        "rt_median": float(df["rt"].median()),
        "rt_max": float(df["rt"].max()),
        "n_rt_nonpositive": int((df["rt"] <= 0).sum()),
        "n_rt_nan": int(df["rt"].isna().sum()),
        "d_counts": {int(k): int(v) for k, v in df["d"].value_counts().sort_index().items()},
        "p_left": float((df["response"] == 1).mean()),
    }


if __name__ == "__main__":
    import json
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    for monkey in ("monkey_c", "monkey_k"):
        d = load_monkey(os.path.join(root, monkey))
        print(f"=== {monkey} ===")
        print(json.dumps(describe(d), indent=2)[:900])
