"""
ProphetWrapper — nn.Module adapter around Facebook Prophet for federated learning.

Wraps a Prophet GAM so that its seasonality Fourier coefficients can be
exchanged via FedAvg-compatible state_dict() / load_state_dict() calls,
enabling FedGAM aggregation over Iroh/QUIC without changing the FL transport
layer.

What gets federated (shared across clients via FedGAM aggregation):
  • Fourier coefficients of annual seasonality   (sin_year_k, cos_year_k  k=1..N)
  • Fourier coefficients of weekly seasonality   (sin_week_k, cos_week_k  k=1..3)
  • External regressor coefficient for velmedia  (beta_velmedia)
  
What stays local (NOT aggregated):
  • Trend parameters (k, m, delta[]) — reflect local province industry/geography
  • Changepoint positions             — province-specific
  
Rationale: annual and weekly seasonality patterns for NO₂/O₃ are structurally
shared across Castilla y León provinces (same climate zone, same national calendar),
whereas the trend encodes local emissions (Ponferrada industrial vs. Soria rural).
Averaging trends across incompatible provinces would produce a meaningless composite.

Compatibility with FLClient/FLServer:
  state_dict()        → OrderedDict of torch.Tensor (federatable params only)
  load_state_dict()   → injects tensors as Prophet warm-start initial values
  train_prophet()     → fits Prophet on local data, using state_dict as init
  predict_ica()       → returns ICA class (0/1/2) from forecasted NO₂ value

Prophet must be installed: pip install prophet
If Prophet is unavailable the class falls back to a trivial majority-class predictor
so that test infrastructure can run without the full Stan toolchain.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Number of Fourier terms per seasonality period (matches IJIMAI paper config)
N_FOURIER_ANNUAL = 10   # Prophet default for yearly seasonality
N_FOURIER_WEEKLY = 3    # Prophet default for weekly seasonality

# ICA thresholds for NO2 (μg/m³).
# Default values are overridden at runtime by load_ica_thresholds() which reads
# the data/air-quailty/datasets/air_quality_ica_thresholds.json file produced
# by preprocess_e7.py (training-set tercile boundaries).
ICA_THRESHOLDS = [8.0, 15.0]   # placeholder; updated by load_ica_thresholds()


def load_ica_thresholds(data_dir: str = "./data") -> None:
    """Load data-driven ICA thresholds from the preprocessing output JSON."""
    import json
    from pathlib import Path
    path = Path(data_dir) / "air-quailty" / "datasets" / "air_quality_ica_thresholds.json"
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        ICA_THRESHOLDS[0] = float(d["q1"])
        ICA_THRESHOLDS[1] = float(d["q2"])
        log.info("ICA thresholds loaded: Q1=%.2f Q2=%.2f μg/m³", *ICA_THRESHOLDS)
    else:
        log.warning("ICA thresholds file not found at %s — using defaults", path)


def _no2_to_ica(no2: float) -> int:
    if no2 < ICA_THRESHOLDS[0]:
        return 0
    if no2 < ICA_THRESHOLDS[1]:
        return 1
    return 2


# state_dict keys that encode the SHARED periodic structure (federated by FedGAM).
# Everything else in the state_dict (k_trend, m_trend, delta_trend) is LOCAL trend.
SEASONALITY_KEYS = (
    "sin_year",
    "cos_year",
    "sin_week",
    "cos_week",
    "beta_velmedia",
)


class ProphetWrapper(nn.Module):
    """
    Prophet GAM wrapped as an nn.Module for FL compatibility.

    The model maintains Fourier seasonality tensors as nn.Parameters so that
    torch.save / load_state_dict work seamlessly with the existing FL transport.

    Args:
        n_annual: Number of Fourier terms for annual seasonality.
        n_weekly: Number of Fourier terms for weekly seasonality.
    """

    def __init__(
        self,
        n_annual: int = N_FOURIER_ANNUAL,
        n_weekly: int = N_FOURIER_WEEKLY,
        n_changepoints: int = 15,
    ) -> None:
        super().__init__()
        self.n_annual = n_annual
        self.n_weekly = n_weekly
        self.n_changepoints = n_changepoints

        # Federatable parameters — initialised to zero (Prophet reinitialises
        # them during fit; they are overwritten by load_state_dict before each
        # round's local fit to provide the global warm-start)
        self.sin_year = nn.Parameter(torch.zeros(n_annual), requires_grad=False)
        self.cos_year = nn.Parameter(torch.zeros(n_annual), requires_grad=False)
        self.sin_week = nn.Parameter(torch.zeros(n_weekly), requires_grad=False)
        self.cos_week = nn.Parameter(torch.zeros(n_weekly), requires_grad=False)
        self.beta_velmedia = nn.Parameter(torch.zeros(1), requires_grad=False)

        # --- Trend parameters (Prophet linear-growth model, scaled space) ---
        # Registered as BUFFERS so they appear in state_dict() (and are therefore
        # transported and aggregatable) WITHOUT inflating param_count(), which
        # reports the federatable seasonality parameter budget only.
        # FedGAM keeps these LOCAL per client; FedAvg averages them globally.
        # This is the variable that isolates "aggregation algorithm" from "model".
        self.register_buffer("k_trend", torch.zeros(1))           # base growth rate
        self.register_buffer("m_trend", torch.zeros(1))           # trend offset
        self.register_buffer("delta_trend", torch.zeros(n_changepoints))  # rate deltas
        # Whether trend warm-start values are populated (skip injection until first fit)
        self.register_buffer("_has_trend", torch.zeros(1))


        # Internal Prophet model — recreated each round after warm-start injection
        self._prophet: Optional[object] = None
        self._fitted: bool = False

        # Fallback majority class when Prophet is unavailable
        self._majority_class: int = 0

    # ------------------------------------------------------------------
    # nn.Module interface (required by FLClient / FLServer)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Not used for Prophet inference; kept for nn.Module compatibility.
        Returns dummy logits of shape (N, 3) where the predicted class has
        the highest logit based on the current majority class.
        """
        n = x.shape[0]
        logits = torch.zeros(n, 3)
        logits[:, self._majority_class] = 1.0
        return logits

    # ------------------------------------------------------------------
    # Prophet training
    # ------------------------------------------------------------------

    def train_prophet(
        self,
        dates: list[str],
        no2_values: list[float],
        velmedia: list[float],
        changepoint_prior_scale: float = 0.05,
        n_changepoints: int = 15,
        use_warmstart: bool = True,
    ) -> None:
        """
        Fit Prophet on local province data, using current state_dict parameters
        as warm-start initial values for both seasonality AND trend.

        Args:
            dates:       List of date strings "YYYY-MM-DD".
            no2_values:  Corresponding NO2 measurements (μg/m³, NOT normalised).
            velmedia:    Corresponding wind speed values.
            use_warmstart: If True, inject current seasonality + trend tensors as
                           Stan ``init`` so the global aggregate actually influences
                           the local fit. If False, Prophet fits from its own
                           default init (original behaviour, federation has no
                           effect — kept only for ablation).
        """
        # Check cmdstan is available before attempting Prophet fit
        # Uses file system check to avoid hanging on cmdstanpy.cmdstan_path()
        # when cmdstan binary is not installed.
        _cmdstan_home = Path.home() / ".cmdstan"
        _cmdstan_ok = _cmdstan_home.exists() and any(_cmdstan_home.iterdir())
        if not _cmdstan_ok:
            log.warning(
                "cmdstan not found — Prophet requires cmdstan for Stan MCMC/MAP. "
                "Install with: python3 -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'. "
                "Falling back to majority-class predictor."
            )
            if no2_values:
                from collections import Counter
                self._majority_class = Counter(_no2_to_ica(v) for v in no2_values).most_common(1)[0][0]
            self._fitted = False
            return

        try:
            import pandas as pd
            from prophet import Prophet
        except ImportError:
            log.warning(
                "Prophet not installed — falling back to majority-class predictor. "
                "Install with: pip install prophet"
            )
            if no2_values:
                classes = [_no2_to_ica(v) for v in no2_values]
                from collections import Counter
                self._majority_class = Counter(classes).most_common(1)[0][0]
            self._fitted = False
            return

        import pandas as pd

        df = pd.DataFrame({
            "ds"      : pd.to_datetime(dates),
            "y"       : no2_values,
            "velmedia": velmedia,
        })
        df = df.dropna(subset=["y"])

        if len(df) < 30:
            log.warning("ProphetWrapper: only %d rows — skipping fit", len(df))
            self._fitted = False
            return

        # Build warm-start init dict from current federated parameters
        init: dict = {}
        sin_y = self.sin_year.detach().cpu().numpy()
        cos_y = self.cos_year.detach().cpu().numpy()
        sin_w = self.sin_week.detach().cpu().numpy()
        cos_w = self.cos_week.detach().cpu().numpy()
        beta_v = float(self.beta_velmedia.detach().cpu().item())

        # Prophet expects beta as a flat array [annual_sin, annual_cos, weekly_sin,
        # weekly_cos, regressor_coefs] interleaved in Fourier order. The regressor
        # coefficient (velmedia) is appended after the seasonality terms.
        n_season = 2 * self.n_annual + 2 * self.n_weekly
        beta_season = np.zeros(n_season + 1)  # +1 for velmedia regressor
        for i in range(self.n_annual):
            beta_season[2 * i]     = sin_y[i]
            beta_season[2 * i + 1] = cos_y[i]
        off = 2 * self.n_annual
        for i in range(self.n_weekly):
            beta_season[off + 2 * i]     = sin_w[i]
            beta_season[off + 2 * i + 1] = cos_w[i]
        beta_season[n_season] = beta_v

        # Only warm-start when requested AND we actually have non-zero global
        # parameters to inject (skips the very first round before any aggregate).
        if use_warmstart and not np.all(beta_season == 0):
            init["beta"] = beta_season
            # Inject trend warm-start if populated from a previous fit/aggregate.
            if float(self._has_trend.item()) > 0.0:
                k_val     = float(self.k_trend.item())
                m_val     = float(self.m_trend.item())
                delta_arr = self.delta_trend.detach().cpu().numpy().astype(float)
                if delta_arr.shape[0] != n_changepoints:
                    # Resize defensively if n_changepoints differs from buffer size
                    resized = np.zeros(n_changepoints)
                    take = min(n_changepoints, delta_arr.shape[0])
                    resized[:take] = delta_arr[:take]
                    delta_arr = resized
                init["k"]     = k_val
                init["m"]     = m_val
                init["delta"] = delta_arr

        m = Prophet(
            yearly_seasonality     = self.n_annual,
            weekly_seasonality     = self.n_weekly,
            daily_seasonality      = False,
            changepoint_prior_scale= changepoint_prior_scale,
            n_changepoints         = n_changepoints,
        )
        m.add_regressor("velmedia")

        try:
            # Pass the warm-start init dict so the global aggregate genuinely
            # influences the local MAP fit. With abundant local data the local
            # likelihood dominates and the init has limited effect; with scarce
            # local data (limited-history scenario) the shared init matters more.
            if init:
                m.fit(df, init=init)
            else:
                m.fit(df)
        except Exception as exc:
            log.warning("Prophet fit failed: %s — retrying without warm-start", exc)
            try:
                m.fit(df)
            except Exception as exc2:
                log.warning("Prophet fit failed again: %s — using majority class", exc2)
                from collections import Counter
                self._majority_class = Counter(_no2_to_ica(v) for v in no2_values).most_common(1)[0][0]
                self._fitted = False
                return

        self._prophet = m
        self._fitted  = True

        # Extract fitted seasonality coefficients back into parameters
        try:
            fitted_params = m.params  # dict of Stan param arrays, shape (n_chains, ...)
            beta_fit = np.median(fitted_params["beta"], axis=0)  # (2*n_annual + 2*n_weekly,)

            if len(beta_fit) >= 2 * self.n_annual:
                syn = np.array([beta_fit[2 * i]     for i in range(self.n_annual)])
                cyn = np.array([beta_fit[2 * i + 1] for i in range(self.n_annual)])
                self.sin_year.data.copy_(torch.tensor(syn, dtype=torch.float32))
                self.cos_year.data.copy_(torch.tensor(cyn, dtype=torch.float32))

            off = 2 * self.n_annual
            if len(beta_fit) >= off + 2 * self.n_weekly:
                swn = np.array([beta_fit[off + 2 * i]     for i in range(self.n_weekly)])
                cwn = np.array([beta_fit[off + 2 * i + 1] for i in range(self.n_weekly)])
                self.sin_week.data.copy_(torch.tensor(swn, dtype=torch.float32))
                self.cos_week.data.copy_(torch.tensor(cwn, dtype=torch.float32))

            # Regressor coefficient — last element if present
            if "extra_regressors_multiplicative" in fitted_params:
                bv = float(np.median(fitted_params["extra_regressors_multiplicative"]))
                self.beta_velmedia.data.fill_(bv)
            elif len(beta_fit) > 2 * self.n_annual + 2 * self.n_weekly:
                bv = float(beta_fit[2 * self.n_annual + 2 * self.n_weekly])
                self.beta_velmedia.data.fill_(bv)

            # --- Extract trend parameters (k, m, delta) into buffers ---
            # These are in Prophet's internal scaled space. FedGAM keeps them
            # local (never re-injected from the global aggregate); FedAvg averages
            # them across provinces, which mixes incompatible local trends.
            if "k" in fitted_params:
                self.k_trend.data.fill_(float(np.median(fitted_params["k"])))
            if "m" in fitted_params:
                self.m_trend.data.fill_(float(np.median(fitted_params["m"])))
            if "delta" in fitted_params:
                delta_fit = np.median(np.asarray(fitted_params["delta"]), axis=0).ravel()
                buf = self.delta_trend.detach().cpu().numpy().copy()
                take = min(buf.shape[0], delta_fit.shape[0])
                buf[:] = 0.0
                buf[:take] = delta_fit[:take]
                self.delta_trend.data.copy_(torch.tensor(buf, dtype=torch.float32))
            self._has_trend.data.fill_(1.0)

        except Exception as exc:
            log.debug("Could not extract Prophet params: %s", exc)

        log.info(
            "ProphetWrapper fitted on %d rows — sin_year_1=%.3f cos_year_1=%.3f",
            len(df), float(self.sin_year[0]), float(self.cos_year[0]),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_ica(
        self,
        dates: list[str],
        velmedia: list[float],
    ) -> list[int]:
        """
        Predict ICA class for each date. Returns list of 0/1/2 integers.
        Falls back to majority class if Prophet was not fitted.
        """
        if not self._fitted or self._prophet is None:
            return [self._majority_class] * len(dates)

        try:
            import pandas as pd
            future = pd.DataFrame({
                "ds"      : pd.to_datetime(dates),
                "velmedia": velmedia,
            })
            forecast = self._prophet.predict(future)
            return [_no2_to_ica(float(v)) for v in forecast["yhat"].values]
        except Exception as exc:
            log.warning("Prophet predict failed: %s — using majority class", exc)
            return [self._majority_class] * len(dates)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def inject_seasonality(self, sd: dict) -> None:
        """
        Load ONLY the shared Fourier seasonality coefficients (and the velmedia
        regressor) from a global aggregate, leaving the local trend buffers
        (k_trend, m_trend, delta_trend) untouched.

        This implements the FedGAM semantics: the seasonality is federated while
        each province keeps its own locally-fitted trend. Contrast with
        ``load_state_dict`` (full FedAvg semantics), which overwrites trend too.
        """
        with torch.no_grad():
            for key in SEASONALITY_KEYS:
                if key in sd:
                    getattr(self, key).data.copy_(sd[key].to(torch.float32))

    def param_count(self) -> int:
        """Number of federatable parameters."""
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        return self.param_count() * 4
