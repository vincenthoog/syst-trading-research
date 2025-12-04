"""
analyse_FF_returns.py

Cross-sectional trading example using the explicit score-driven filter
to model **returns directly** (Option 1):

    Y_t = monthly excess returns of Fama-French 49 industry portfolios

We:
  - fit the explicit score-driven model on Y_in (returns),
  - get in-/out-of-sample mean forecasts mu_hat
  - build cross-sectional long/short weights from mu_hat_out,
  - risk-scale via an EWMA covariance model,
  - evaluate the resulting PnL.

This is meant as an illustrative research script, not production code.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================

FILTER_METHOD     = "explicit"     # currently this script only uses explicit_analyse_oos
SCALING           = "invFisher"    # 'gasp' | 'invFisher' | 'sqrtInvFisher' | 'identity' | 'invHessian'
START_MODE        = "standard"     # 'standard' | 'local_hess' | 'ewma' | 'inv_hessian'

# Trading & risk
TARGET_VOL_ANN    = 0.10           # annualized target portfolio volatility (e.g. 0.10–0.15)
WEIGHT_CAP        = 0.05           # absolute weight cap per industry
CS_STD_MIN        = 1e-6           # avoid divide-by-zero in cross-sectional std

# --- NEW: stop-loss config ---
APPLY_STOP_LOSS   = False   # set False if you want the “raw” backtest
MAX_DRAWDOWN      = 0.20    # 0.20 = stop if you ever lose 20% from peak equity

# Covariance model (monthly)
ALPHA_COV         = 0.05           # EWMA update for Sig
LAMBDA_STRUCT     = 0.25           # shrink Sig towards structured Sig*
LAMBDA_DIAG       = 0.10           # additional diagonal shrink towards diag(Sig)

# Walk-forward windows (in MONTHS, since we trade monthly returns)
WINDOW_IN         = 240            # in-sample length (months)
WINDOW_OUT        = 60             # out-of-sample length per step (months)
STEP              = 60             # re-train every STEP months

# Exposure / neutrality
ENFORCE_SUM_TO_ZERO = True         # enforce Sum w = 0 each month
EXPO_MODE           = "market"     # 'market' (single factor) or 'pca' (first K eigenvectors)
K_FACTORS           = 1            # if EXPO_MODE='pca', how many PCs to use


# ============================================================
# Imports relative to project layout / score-driven-filters package
# ============================================================

here = Path(__file__).resolve().parent
project_root = here.parent           # adjust if your structure differs
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

try:
    from score_driven_filters.get_data_and_W import get_data, build_W_from_corr
    from score_driven_filters.explicit_filter import explicit_analyse_oos
except ImportError:
    # fallback if running in the same directory as the source files
    from get_data_and_W import get_data, build_W_from_corr
    from explicit_filter import explicit_analyse_oos


# ============================================================
# Utilities
# ============================================================

def ann_stats(pnl: pd.Series, ann: int = 12) -> dict:
    """
    Annualized stats assuming 'ann' periods per year.
    Here pnl is monthly, so ann=12.
    """
    r = pnl.dropna()
    if r.empty:
        return {
            "N": 0,
            "ann_ret": np.nan,
            "ann_vol": np.nan,
            "IR": np.nan,
            "maxDD": np.nan,
        }

    mu  = r.mean() * ann
    vol = r.std(ddof=1) * np.sqrt(ann)
    ir  = mu / vol if vol > 0 else np.nan
    eq  = (1.0 + r).cumprod()
    mdd = (eq / eq.cummax() - 1.0).min()

    return {
        "N": int(r.shape[0]),
        "ann_ret": float(mu),
        "ann_vol": float(vol),
        "IR": float(ir),
        "maxDD": float(mdd),
    }


def build_exposure_matrix(train_df: pd.DataFrame, mode: str = "market", k: int = 1) -> np.ndarray:
    """
    Returns B (R x K) exposure matrix used for neutrality constraints.
    - 'market': single equal-weight 'market' exposure (K=1)
    - 'pca'   : first k principal components (from asset covariance)
    """
    R = train_df.shape[1]
    if mode == "market":
        return np.ones((R, 1)) / np.sqrt(R)   # normalized constant-loading market

    # PCA on cross-sectional covariance of returns
    C = np.cov(train_df.values, rowvar=False)  # R x R
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:k]]             # R x k
    return eigvecs


def project_to_neutrality(raw_w: np.ndarray, B: np.ndarray, enforce_sum_to_zero: bool = True) -> np.ndarray:
    """
    Projects raw_w (R,) into the subspace with B' w = 0 (and optionally sum(w)=0).
    """
    w = raw_w.copy()
    R = w.shape[0]
    if enforce_sum_to_zero:
        B_full = np.column_stack([np.ones(R), B])  # (R x (1+K))
    else:
        B_full = B

    BtB = B_full.T @ B_full
    try:
        inv_BtB = np.linalg.inv(BtB)
    except np.linalg.LinAlgError:
        inv_BtB = np.linalg.pinv(BtB)

    alpha = inv_BtB @ (B_full.T @ w)
    w_star = w - B_full @ alpha
    return w_star


def ewma_cov_update(Sigma_prev: np.ndarray, eps_t: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    EWMA covariance update:
        Sig_{t+1} = (1-α)*Sig_t + α*(eps_t eps_t')
    """
    outer = np.outer(eps_t, eps_t)
    return (1.0 - alpha) * Sigma_prev + alpha * outer


def struct_target_cov(Sigma: np.ndarray, mode: str = "market", l_diag: float = 0.10) -> np.ndarray:
    """
    Construct a structured covariance Sig* from Sig:
      - market factor + diagonal residual, then optionally shrink diag.

    This is a simple shrinkage target for illustration.
    """
    R = Sigma.shape[0]
    m = np.ones(R) / np.sqrt(R)
    beta_m = Sigma @ m
    var_m  = float(m.T @ beta_m)
    if var_m <= 0:
        var_m = 1e-6

    Sigma_m = np.outer(beta_m, beta_m) / var_m
    diag_res = np.diag(np.diag(Sigma - Sigma_m))
    Sigma_star = Sigma_m + (1.0 - l_diag) * diag_res
    return Sigma_star


def shrink_cov(Sigma: np.ndarray, Sigma_star: np.ndarray,
               l_struct: float = 0.25, l_diag: float = 0.10) -> np.ndarray:
    """
    Sig_shrunk = (1-lam_struct)*Sig + lam_struct*Sig*
    then an extra diag shrink towards diag(Σ_shrunk).
    """
    Sig = (1.0 - l_struct) * Sigma + l_struct * Sigma_star
    diag = np.diag(np.diag(Sig))
    Sig2 = (1.0 - l_diag) * Sig + l_diag * diag
    return Sig2


def clip_weights(w: np.ndarray, cap: float = 0.05) -> np.ndarray:
    """
    Enforce |w_i| <= cap by scaling if needed.
    """
    mx = np.max(np.abs(w))
    if mx <= cap or cap <= 0:
        return w
    return w * (cap / mx)


def run_backtest(
    target_vol_ann: float = TARGET_VOL_ANN,
    weight_cap: float = WEIGHT_CAP,
    apply_stop_loss: bool = APPLY_STOP_LOSS,
    max_drawdown: float = MAX_DRAWDOWN,
    window_in: int = WINDOW_IN,
    window_out: int = WINDOW_OUT,
    step: int = STEP,
    expo_mode: str = EXPO_MODE,
    k_factors: int = K_FACTORS,
    scaling: str = SCALING,
    start_mode: str = START_MODE,
    alpha_cov: float = ALPHA_COV,
    lambda_struct: float = LAMBDA_STRUCT,
    lambda_diag: float = LAMBDA_DIAG,
    verbose: bool = True,
) -> dict:
    """
    Run the score-driven cross-sectional backtest on Fama-French 49 industries.

    Returns
    -------
    dict with keys:
        'pnl'      : pd.Series of (possibly stop-loss-adjusted) monthly PnL
        'pnl_raw'  : pd.Series of raw monthly PnL (no stop-loss)
        'weights'  : pd.DataFrame of portfolio weights (monthly)
        'mu_hat'   : pd.DataFrame of out-of-sample mean forecasts
        'stats'    : dict of annualized performance statistics (on 'pnl')
        'config'   : dict of config parameters actually used
    """
    # --------------------------------------------------------
    # 1. Load daily Fama-French 49 industry excess returns
    # --------------------------------------------------------
    if verbose:
        print("Loading Fama-French daily value-weighted excess returns (49 industries)...")

    vw_xs_daily, _ = get_data(check=False, freq="daily")  # decimals, daily xs returns
    assert isinstance(vw_xs_daily.index, pd.DatetimeIndex)

    # Monthly excess returns: compound daily excess returns within each month
    rets_m = vw_xs_daily.resample("ME").apply(lambda df: (1.0 + df).prod() - 1.0)
    rets_m = rets_m.dropna()
    if verbose:
        print("Monthly sample:", rets_m.index.min(), "->", rets_m.index.max(), rets_m.shape)

    Y_all = rets_m.copy()   # this is what we model with the explicit filter
    dates = Y_all.index
    cols = list(Y_all.columns)
    R = Y_all.shape[1]

    # --------------------------------------------------------
    # 2. Walk-forward setup
    # --------------------------------------------------------
    all_pnl = []
    all_w   = []
    all_mu  = []

    t0 = 0
    step_id = 0

    while True:
        t_in_start = t0
        t_in_end   = t0 + window_in
        t_out_end  = t_in_end + window_out

        if t_out_end > Y_all.shape[0]:
            break

        if verbose:
            print(f"\n=== Walk {step_id}: in-sample [{t_in_start}:{t_in_end}), "
                  f"out-of-sample [{t_in_end}:{t_out_end}) ===")
        step_id += 1

        Y_in  = Y_all.iloc[t_in_start:t_in_end]
        Y_out = Y_all.iloc[t_in_end:t_out_end]

        idx_in  = Y_in.index
        idx_out = Y_out.index

        cT_in, cR = Y_in.shape
        cT_out, _ = Y_out.shape

        # Design matrices: intercept-only
        X_in  = np.ones((cT_in,  cR, 1), dtype=np.float64)
        X_out = np.ones((cT_out, cR, 1), dtype=np.float64)

        # ----------------------------------------------------
        # 3. Build W from in-sample correlations of returns
        # ----------------------------------------------------
        W_df = build_W_from_corr(
            Y_in,
            k=16,
            min_corr=0.01,
            standardize=True,
            corr_method="pearson",
        )
        W = W_df.to_numpy()

        # ----------------------------------------------------
        # 4. Fit explicit score-driven model (returns as Y)
        # ----------------------------------------------------
        res = explicit_analyse_oos(
            Y_in.to_numpy(),
            X_in,
            mu_in=None,
            Y_out=Y_out.to_numpy(),
            X_out=X_out,
            mu_out=None,
            W=W,
            scaling=scaling,
            start_mode=start_mode,
            cb_start=False,
            cb_general=False,
            save_str=f"start_params_{scaling}_{start_mode}.pkl",
            maxiter=1500,
        )

        # mu_out_hat and y_hat_out are in the same units as Y (monthly decimal returns)
        mu_out_hat = pd.DataFrame(res["mu_out_hat"], index=idx_out, columns=cols)
        y_hat_out  = pd.DataFrame(res["y_hat_out"],   index=idx_out, columns=cols)

        all_mu.append(mu_out_hat)

        # ----------------------------------------------------
        # 5. Build cross-sectional weights from mu_out_hat
        # ----------------------------------------------------
        B = build_exposure_matrix(Y_in, mode=expo_mode, k=k_factors)

        w_raw_list = []
        for t in idx_out:
            mu_t = mu_out_hat.loc[t].values
            mu_mean = mu_t.mean()
            mu_std  = mu_t.std(ddof=1)
            if mu_std < CS_STD_MIN:
                z_t = np.zeros_like(mu_t)
            else:
                z_t = (mu_t - mu_mean) / mu_std

            # Market / factor neutrality
            w_t = project_to_neutrality(z_t, B, enforce_sum_to_zero=ENFORCE_SUM_TO_ZERO)
            # Cap max abs weight
            w_t = clip_weights(w_t, cap=weight_cap)
            w_raw_list.append(w_t)

        w_raw = pd.DataFrame(w_raw_list, index=idx_out, columns=cols)

        # ----------------------------------------------------
        # 6. EWMA covariance + vol targeting
        # ----------------------------------------------------
        # Initialize Sig from in-sample returns
        Sigma = np.cov(Y_in.values, rowvar=False)
        if not np.all(np.isfinite(Sigma)):
            Sigma = np.nan_to_num(Sigma)
        # add small ridge if needed
        Sigma = Sigma + 1e-8 * np.eye(R)

        target_monthly_vol = target_vol_ann / np.sqrt(12.0)

        pnl_step = []
        w_step   = []

        for t in idx_out:
            r_t = Y_out.loc[t].values   # monthly excess returns
            w_t = w_raw.loc[t].values   # unscaled weights

            # ex-ante vol estimate using current Sigma
            vol_t = np.sqrt(max(w_t @ Sigma @ w_t, 1e-12))
            scale = target_monthly_vol / vol_t
            w_scaled = w_t * scale

            # realized PnL for this month
            pnl_t = float(w_scaled @ r_t)
            pnl_step.append(pnl_t)
            w_step.append(w_scaled)

            # update Sigma using centered returns
            eps_t = r_t - r_t.mean()
            Sigma = ewma_cov_update(Sigma, eps_t, alpha=alpha_cov)
            Sigma_star = struct_target_cov(Sigma, mode="market", l_diag=lambda_diag)
            Sigma = shrink_cov(Sigma, Sigma_star, l_struct=lambda_struct, l_diag=lambda_diag)

        pnl_step = pd.Series(pnl_step, index=idx_out)
        w_step   = pd.DataFrame(w_step, index=idx_out, columns=cols)

        all_pnl.append(pnl_step)
        all_w.append(w_step)

        t0 += step

    # --------------------------------------------------------
    # 7. Concatenate and evaluate
    # --------------------------------------------------------
    pnl_raw = pd.concat(all_pnl).sort_index()
    w_all   = pd.concat(all_w).sort_index()
    mu_all  = pd.concat(all_mu).sort_index()

    pnl = pnl_raw.copy()

    # --------------------------------------------------------
    # Optional: apply portfolio max-drawdown stop-loss
    # --------------------------------------------------------
    if apply_stop_loss:
        equity = 1.0
        peak   = 1.0
        trading = True
        pnl_sl = []

        for r in pnl_raw:
            if trading:
                equity *= (1.0 + r)
                peak = max(peak, equity)
                dd = equity / peak - 1.0  # <= 0; e.g. -0.25 = -25% drawdown

                pnl_sl.append(r)

                # if drawdown exceeds threshold, stop trading from NEXT month
                if dd <= -max_drawdown:
                    trading = False
            else:
                # flat (cash) after stop-loss is hit
                pnl_sl.append(0.0)

        pnl = pd.Series(pnl_sl, index=pnl_raw.index, name="pnl_stop")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    stats = ann_stats(pnl, ann=12)
    r = pnl.dropna()
    N = r.shape[0]
    if N > 1:
        m = r.mean()
        s = r.std(ddof=1)
        t_mean = m / (s / np.sqrt(N))
        stats["tstat_mean_monthly"] = float(t_mean)
    else:
        stats["tstat_mean_monthly"] = np.nan

    # Pack everything into a result dict
    result = {
        "pnl": pnl,
        "pnl_raw": pnl_raw,
        "weights": w_all,
        "mu_hat": mu_all,
        "stats": stats,
        "config": {
            "target_vol_ann": target_vol_ann,
            "weight_cap": weight_cap,
            "apply_stop_loss": apply_stop_loss,
            "max_drawdown": max_drawdown,
            "window_in": window_in,
            "window_out": window_out,
            "step": step,
            "expo_mode": expo_mode,
            "k_factors": k_factors,
            "scaling": scaling,
            "start_mode": start_mode,
            "alpha_cov": alpha_cov,
            "lambda_struct": lambda_struct,
            "lambda_diag": lambda_diag,
        },
    }
    return result



# ============================================================
# Main: load data, run model, trade
# ============================================================
def main():
    results = run_backtest()

    pnl   = results["pnl"]
    stats = results["stats"]

    print("\n=== Final PnL stats (score-driven returns strategy, monthly) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    plt.figure()
    plt.plot((1.0 + pnl).cumprod())
    plt.title("Score-driven returns strategy: cumulative PnL (monthly)")
    plt.xlabel("time")
    plt.ylabel("equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
