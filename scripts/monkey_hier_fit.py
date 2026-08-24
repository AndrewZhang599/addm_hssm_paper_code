#!/usr/bin/env python
"""Fit the hierarchical aDDM for ONE monkey and save the idata to scratch.

Sessions enter as random intercepts on eta, kappa, a, b and x0; t stays fixed at
0 because `rt` is already non-decision-time corrected (see monkey_hier_data.py).

One invocation fits one monkey.  There is deliberately no flag that takes two
monkey directories -- the two monkeys are separate models and separate jobs.

    python monkey_hier_fit.py --monkey-dir .../monkey_c --n-sessions 3 \
        --draws 50 --tune 50 --chains 1

    python monkey_hier_fit.py --monkey-dir .../monkey_k --dry-run
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make the sibling loader importable regardless of cwd (sbatch runs from anywhere).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# The model, in one place.  Bounds and prior centres are documented in the plan;
# INTERCEPT_MU_LINK values are on the LINK (linear-predictor) scale, not the
# parameter scale: param = lo + (hi - lo) * sigmoid(eta_link), so a negative
# entry just means "below the midpoint of the bounds", never a negative param.
# ---------------------------------------------------------------------------
PARAMS = ("eta", "kappa", "a", "b", "x0")

BOUNDS = {
    "eta": (0.0, 1.0),
    "kappa": (0.0, 5.0),
    "a": (0.1, 6.0),
    # Symmetric boundary-collapse slope; boundaries are +/-(a - b*tau).
    # Widened from the aDDMConfig stock (0, 3): in job 5173665 the posterior piled
    # up against that ceiling (P(b > 2.8) = 0.94), inflating `a` to compensate and
    # squashing b's session spread as the sigmoid saturated near the bound.
    "b": (0.0, 6.0),
    "x0": (-2.0, 2.0),
}
# b's centre of 1.0 on (0, 6) is 1/6 of the range -- exactly where 0.5 sat on the
# old (0, 3), so the prior's shape relative to the bounds is unchanged and only the
# ceiling moves. Deliberately NOT recentred onto the ~2.9 that the unconverged run
# suggested. link = log((1-0)/(6-1)) = -1.609.
INTERCEPT_MU_LINK = {"eta": -0.64, "kappa": -1.31, "a": -1.37, "b": -1.61, "x0": 0.03}
INTERCEPT_SD = 1.5
GRP_SD = {"eta": 0.5, "kappa": 0.5, "a": 0.5, "b": 0.5, "x0": 0.3}

DEFAULT_OUT = os.environ.get(
    "SCRATCH_IDATA", f"/oscar/scratch/{os.environ.get('USER', 'unknown')}/addm_hier"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Hierarchical aDDM (sessions as random intercepts) for one monkey.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--monkey-dir",
        required=True,
        help="Path to ONE monkey folder, e.g. .../two_monkey_42_session/monkey_c",
    )
    p.add_argument(
        "--n-sessions", type=int, default=None,
        help="Fit only the first N sessions (filename order). Default: all.",
    )
    p.add_argument(
        "--sessions", nargs="+", default=None,
        help="Explicit session names (CSV stems). Overrides --n-sessions.",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT, help="Where the .nc is written.")
    p.add_argument("--tag", default=None, help="Extra string in the output filename.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing .nc.")

    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--target-accept", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=2024)

    p.add_argument(
        "--params", nargs="+", default=list(PARAMS), choices=list(PARAMS),
        help="Which parameters get a (1|session) random intercept. The rest are "
             "sampled as single pooled values with the default uniform-on-bounds prior.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build the model, run the checks, time one logp + one gradient, then stop.",
    )
    p.add_argument(
        "--gpu", default=None,
        help="Value for CUDA_VISIBLE_DEVICES. Default: leave as inherited (Slurm sets "
             "it under --gres=gpu:N); falls back to '0' when unset.",
    )
    p.add_argument(
        "--chain-method", default="auto",
        choices=["auto", "parallel", "vectorized", "sequential"],
        help="How numpyro runs the chains. 'parallel' pmaps one chain per XLA device "
             "(needs >= --chains GPUs, uses NCCL). 'vectorized' vmaps all chains onto a "
             "SINGLE device. 'sequential' runs them one after another. 'auto' picks "
             "parallel when there are enough devices, else vectorized.",
    )
    p.add_argument(
        "--nccl-p2p-disable", action="store_true",
        help="Set NCCL_P2P_DISABLE=1. Needed on multi-GPU Oscar allocations, where "
             "HSSM+numpyro otherwise stalls at 100%% during sampling.",
    )
    return p.parse_args(argv)


def setup_env(args):
    """Must run before jax/hssm are imported."""
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # XLA otherwise pre-allocates 75% of the card up front.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    n_visible = len([v for v in visible.split(",") if v.strip()])

    # Multi-GPU sampling goes through NCCL, which is where HSSM+numpyro has been
    # observed to stall at 100% on Oscar. Disabling peer-to-peer transport is the
    # known mitigation; do it automatically rather than leaving it to be forgotten.
    if n_visible > 1 or args.nccl_p2p_disable:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        print(f"[env] NCCL_P2P_DISABLE={os.environ['NCCL_P2P_DISABLE']} "
              f"({n_visible} GPUs visible)")

    print(f"[env] CUDA_VISIBLE_DEVICES={visible}  ({n_visible} device(s))")
    return n_visible


def install_shims():
    """Compat shims for this checkout (verbatim from monkey_data_recovery.ipynb cell 0)."""
    import numpyro.infer as _npi
    import xarray as xr

    # (1) HSSM injects nuts_sampler_kwargs={"jitter": False} for the numpyro backend,
    #     but pymc 6.2 forwards that dict verbatim to numpyro's NUTS, which has no
    #     `jitter` argument.  Swallow it.
    if not getattr(_npi.NUTS.__init__, "_shim", False):
        _orig_nuts_init = _npi.NUTS.__init__

        def _nuts_init(self, *a, jitter=None, **kw):
            _orig_nuts_init(self, *a, **kw)

        _nuts_init._shim = True
        _npi.NUTS.__init__ = _nuts_init

    # (2) HSSM._clean_posterior_group does `idata["posterior"][list_of_names]`, which
    #     xarray 2026.7's DataTree no longer allows.  Route it through `.dataset`.
    from hssm.base import HSSMBase

    # NOTE: HSSM calls this as `self._clean_posterior_group(dt=...)` (base.py:739,
    # 830, 932). The shim in monkey_data_recovery.ipynb cell 0 names the parameter
    # `idata`, which is stale against this checkout and raises
    # "unexpected keyword argument 'dt'" *after* sampling finishes -- i.e. it throws
    # away a completed fit. Accept both spellings.
    def _clean_posterior_group(self, dt=None, idata=None):
        dt = dt if dt is not None else idata
        if dt is None or "posterior" not in dt:
            return
        post = dt["posterior"].dataset
        deterministics = {v.name for v in self.pymc_model.deterministics}
        drop = {k for k in self.model.distributional_components if k in deterministics}
        keep = [v for v in post.data_vars if v not in drop and "_mean" not in str(v)]
        dt["posterior"] = xr.DataTree(post[keep])

    HSSMBase._clean_posterior_group = _clean_posterior_group


def force_chain_method(method):
    """Make numpyro's `chain_method` reachable from HSSM.

    pymc 6.2's `_sample_external_nuts` (pymc/sampling/mcmc.py:505-530) calls
    `sample_jax_nuts` with a fixed argument list that does not include
    `chain_method` and does not forward **kwargs, so the parameter is otherwise
    unreachable through HSSM -> bambi -> pm.sample and silently stays at its
    default of "parallel".  `nuts_kwargs` is not a way in either: it is forwarded
    to the NumPyro *kernel*, not to sample_jax_nuts.

    So wrap the module attribute, exactly as HSSM's own
    `_force_jax_nuts_no_jitter` does for `jitter` (hssm/utils.py:610-658).
    Returns a callable that restores the original.
    """
    import functools

    import pymc.sampling.jax as pymc_jax

    original = getattr(pymc_jax, "sample_jax_nuts", None)
    if original is None:
        print("[warn] pymc.sampling.jax.sample_jax_nuts missing; "
              "cannot set chain_method", file=sys.stderr)
        return lambda: None

    @functools.wraps(original)
    def _with_chain_method(*a, **kw):
        kw["chain_method"] = method
        return original(*a, **kw)

    pymc_jax.sample_jax_nuts = _with_chain_method

    def restore():
        pymc_jax.sample_jax_nuts = original

    return restore


def resolve_chain_method(requested, chains, n_devices):
    """Pick the chain method and say why. Returns (method, note)."""
    if requested == "auto":
        if n_devices >= chains > 1:
            return "parallel", f"{n_devices} devices >= {chains} chains"
        if chains > 1:
            return "vectorized", (
                f"only {n_devices} device(s) for {chains} chains -- vmapping them onto "
                "one GPU instead of letting numpyro fall back to sequential"
            )
        return "parallel", "single chain"

    if requested == "parallel" and chains > n_devices:
        print(
            f"[warn] chain_method='parallel' with {chains} chains but only "
            f"{n_devices} visible device(s). numpyro falls back to SEQUENTIAL -- "
            "request more GPUs (sbatch --gres=gpu:N) or use --chain-method vectorized.",
            file=sys.stderr,
        )
    return requested, "user-specified"


def build_model(df, hier_params):
    """The hierarchical aDDM.  See section 1 of the plan for the formulas."""
    from dataclasses import replace

    from hssm.addm import aDDM
    from hssm.addm.config import aDDMConfig

    cfg = replace(aDDMConfig(), bounds={**aDDMConfig().bounds, **BOUNDS})

    include = [
        {
            "name": p,
            "formula": f"{p} ~ 1 + (1|session)",
            "prior": {
                "Intercept": {
                    "name": "Normal",
                    "mu": INTERCEPT_MU_LINK[p],
                    "sigma": INTERCEPT_SD,
                },
                "1|session": {
                    "name": "Normal",
                    "mu": 0.0,
                    "sigma": {"name": "HalfNormal", "sigma": GRP_SD[p]},
                },
            },
        }
        for p in hier_params
    ]

    return aDDM(
        data=df,
        model_config=cfg,
        include=include,
        t=0.0,                      # only t is fixed; b is estimated
        link_settings="log_logit",  # -> gen_logit on every regressed param
        prior_settings="safe",
    )


def preflight(model, hier_params):
    """The two cheap checks. Returns (ok, report) -- abort the run if not ok."""
    import numpy as np

    report = {}
    ok = True

    trialwise = {p: bool(model.params[p].is_trialwise) for p in hier_params}
    report["is_trialwise"] = trialwise
    if not all(trialwise.values()):
        print(f"[FAIL] not all regressed params are trial-wise: {trialwise}", file=sys.stderr)
        ok = False
    else:
        print(f"[ok] trial-wise on all {len(hier_params)} regressed params")

    t0 = time.perf_counter()
    logp_fn = model.pymc_model.compile_logp()
    ip = model.pymc_model.initial_point()
    lp = float(logp_fn(ip))
    report["logp_at_initial_point"] = lp
    report["logp_compile_and_eval_s"] = round(time.perf_counter() - t0, 3)
    if not np.isfinite(lp):
        print(f"[FAIL] logp at the initial point is {lp}", file=sys.stderr)
        ok = False
    else:
        print(f"[ok] logp at initial point = {lp:.2f}")

    return ok, report


def time_gradient(model, report):
    """Cost of one gradient -- the number that sets the wall-clock estimate."""
    t0 = time.perf_counter()
    dlogp_fn = model.pymc_model.compile_dlogp()
    ip = model.pymc_model.initial_point()
    dlogp_fn(ip)
    report["dlogp_compile_and_first_eval_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        dlogp_fn(ip)
    per = (time.perf_counter() - t0) / reps
    report["dlogp_per_eval_s"] = round(per, 4)
    print(f"[timing] one gradient = {per:.4f} s")
    return per


def out_paths(args, n_sessions):
    stem = (
        f"addm_hier_{args.monkey}_{n_sessions}sess"
        f"_d{args.draws}_t{args.tune}_c{args.chains}_seed{args.seed}"
    )
    if sorted(args.params) != sorted(PARAMS):
        stem += "_re-" + "".join(p[0] for p in args.params)
    if args.tag:
        stem += f"_{args.tag}"
    out_dir = Path(args.out_dir)
    return out_dir / f"{stem}.nc", out_dir / f"{stem}.json"


def main(argv=None):
    args = parse_args(argv)
    n_visible = setup_env(args)

    import monkey_hier_data as mhd

    args.monkey = mhd.monkey_label(args.monkey_dir)
    if not Path(args.monkey_dir).is_dir():
        sys.exit(f"[FAIL] --monkey-dir is not a directory: {args.monkey_dir}")

    df = mhd.load_monkey(
        args.monkey_dir, n_sessions=args.n_sessions, sessions=args.sessions
    )
    desc = mhd.describe(df)
    print(
        f"[data] monkey={args.monkey}  sessions={desc['n_sessions']}  "
        f"trials={desc['n_trials']}  rt {desc['rt_min']:.3f}/{desc['rt_median']:.3f}/"
        f"{desc['rt_max']:.3f}  P(left)={desc['p_left']:.3f}"
    )
    if desc["n_rt_nonpositive"] or desc["n_rt_nan"]:
        sys.exit(
            f"[FAIL] {desc['n_rt_nonpositive']} non-positive and {desc['n_rt_nan']} NaN "
            "RTs in the pooled frame; fix the loader before fitting."
        )

    nc_path, json_path = out_paths(args, desc["n_sessions"])
    if nc_path.exists() and not args.overwrite and not args.dry_run:
        sys.exit(f"[FAIL] {nc_path} exists. Pass --overwrite to replace it.")
    nc_path.parent.mkdir(parents=True, exist_ok=True)

    install_shims()
    import arviz as az
    import hssm

    hssm.set_floatX("float64")

    import jax

    n_devices = jax.device_count()
    print(f"[env] hssm {hssm.__version__} | arviz {az.__version__} | "
          f"jax {jax.__version__} | devices: {jax.devices()}")
    if n_devices != n_visible:
        print(f"[warn] CUDA_VISIBLE_DEVICES lists {n_visible} but jax sees "
              f"{n_devices}; using {n_devices}.", file=sys.stderr)

    pooled = [p for p in PARAMS if p not in args.params]
    print(
        f"[model] (1|session) on {args.params}"
        + (f"; pooled (single value): {pooled}" if pooled else "")
    )

    t_build = time.perf_counter()
    model = build_model(df, args.params)
    build_s = round(time.perf_counter() - t_build, 2)
    print(model)
    print(f"[model] built in {build_s}s")

    ok, report = preflight(model, args.params)
    if not ok:
        sys.exit("[FAIL] preflight checks failed; not sampling.")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "args": {k: v for k, v in vars(args).items()},
        "hssm_version": hssm.__version__,
        "arviz_version": az.__version__,
        "data": desc,
        "bounds": BOUNDS,
        "intercept_mu_link": INTERCEPT_MU_LINK,
        "intercept_sd": INTERCEPT_SD,
        "group_sd": GRP_SD,
        "hierarchical_params": list(args.params),
        "pooled_params": pooled,
        "build_seconds": build_s,
        "preflight": report,
    }

    if args.dry_run:
        per = time_gradient(model, report)
        est = per * args.chains * (args.draws + args.tune) * 2 ** 6
        print(
            f"[estimate] {args.chains} chains x {args.draws + args.tune} iters "
            f"x ~2^6 leapfrogs ~= {est / 3600:.1f} h (very rough)"
        )
        meta["preflight"] = report
        meta["dry_run"] = True
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(meta, indent=2, default=str))
        print(f"[done] dry run; report -> {json_path}")
        return 0

    # Only intercepts that actually exist as model variables.
    named = set(model.pymc_model.named_vars)
    initvals = {
        f"{p}_Intercept": INTERCEPT_MU_LINK[p]
        for p in args.params
        if f"{p}_Intercept" in named
    }
    print(f"[sample] initvals={initvals}")

    chain_method, why = resolve_chain_method(args.chain_method, args.chains, n_devices)
    print(f"[sample] chain_method={chain_method} ({why})")
    meta_chain = {"chain_method": chain_method, "reason": why, "n_devices": n_devices}
    restore = force_chain_method(chain_method)

    t0 = time.perf_counter()
    try:
        idata = model.sample(
            sampler="numpyro",
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            random_seed=args.seed,
            target_accept=args.target_accept,
            initvals=initvals,
            idata_kwargs={"log_likelihood": False},
        )
    finally:
        restore()
    sample_s = round(time.perf_counter() - t0, 1)
    print(f"[sample] {sample_s}s ({sample_s / 3600:.2f} h)")

    meta["sample_seconds"] = sample_s
    meta["sampling"] = meta_chain
    try:
        stats = idata["sample_stats"]
        stats = stats.dataset if hasattr(stats, "dataset") else stats
        meta["divergences"] = int(stats["diverging"].sum())
        print(f"[diagnostics] divergences = {meta['divergences']}")
    except Exception as e:  # diagnostics must never sink a finished fit
        meta["divergences"] = f"unavailable: {e}"

    idata.to_netcdf(str(nc_path))
    json_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"[done] idata -> {nc_path}")
    print(f"[done] meta  -> {json_path}")

    try:
        pop = [f"{p}_Intercept" for p in args.params]
        print(az.summary(idata, var_names=pop, filter_vars="like").to_string())
    except Exception as e:
        print(f"[warn] az.summary failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
