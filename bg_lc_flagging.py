import logging
import numpy as np
import matplotlib.pyplot as plt


class lc_flagging:
    def __init__(self, logger: logging.Logger, time: list[float], exp_time: float,
                 flux: list[float], flux_err: list[float], filename: str) -> None:
        self.logger = logger
        self.time = np.asarray(time, dtype=float)
        self.exp_time = float(exp_time)
        self.flux = np.asarray(flux, dtype=float)
        self.flux_err = np.asarray(flux_err, dtype=float)
        self.filename = filename

        self.n_sigma = 3.0
        self.gap_tol = 1.5
        self.spacing_tol = 0.05
        self.runlen_bump = 0.10
        self.spacing_bump = 1.5

    def need_len(self, run) -> int:
        n = self.flux.size
        i0, i1 = run[0], run[-1]
        touches_start = i0 == 0
        touches_end = i1 == n - 1
        gap_left = (not touches_start) and \
            (self.time[i0] - self.time[i0 - 1]) > self.gap_tol * self.exp_time
        gap_right = (not touches_end) and \
            (self.time[i1 + 1] - self.time[i1]) > self.gap_tol * self.exp_time
        one_sided = touches_start or touches_end or gap_left or gap_right
        return 3 if one_sided else 2

    def plot(self, result: dict = None, save_path: str = None, show: bool = False):
        if result is None:
            result = self.iterative_masking()

        base = result["baseline"]
        sigma = result["sigma"]
        threshold = base - self.n_sigma * sigma

        is_flag = np.zeros(self.flux.size, bool)
        for d in result["dips"]:
            is_flag[np.asarray(d["indices"])] = True

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.errorbar(self.time[~is_flag], self.flux[~is_flag],
                    yerr=self.flux_err[~is_flag] if self.flux_err.size else None,
                    fmt="o", ms=4, color="0.45", ecolor="0.8",
                    elinewidth=0.8, capsize=0, zorder=2, label="data")

        if is_flag.any():
            ax.errorbar(self.time[is_flag], self.flux[is_flag],
                        yerr=self.flux_err[is_flag] if self.flux_err.size else None,
                        fmt="o", ms=6, color="red", ecolor="red",
                        elinewidth=0.8, capsize=0, zorder=4, label="flagged")

        ax.axhline(base, color="black", lw=1.3, zorder=3,
                   label=f"mean = {base:.4g}")
        ax.axhline(threshold, color="crimson", lw=1.2, ls=":", zorder=3,
                   label=f"-{self.n_sigma:g}sigma threshold")

        ax.set_xlabel("time")
        ax.set_ylabel("flux")
        ax.set_title(f"{self.filename}  -  {result['n_dips']} dip(s), score={result['score']:.2f}")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        if save_path is None:
            save_path = f"{self.filename}_flagged.png"
        fig.savefig(save_path, dpi=130)
        self.logger.info("%s: plot saved to %s", self.filename, save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)
        return save_path

    def iterative_masking(self) -> dict:
        mask = np.ones(self.flux.size, bool)
        m = s = np.nan
        for _ in range(20):
            m, s = self.flux[mask].mean(), self.flux[mask].std()
            new = np.abs(self.flux - m) < self.n_sigma * s
            if np.array_equal(new, mask):
                break
            mask = new

        baseline = self.flux[mask]
        base_mean = baseline.mean()
        resid = self.flux - base_mean

        sigma = 1.4826 * np.median(np.abs(baseline - np.median(baseline)))
        if sigma == 0:
            self.logger.warning("%s: MAD scale is zero; skipping", self.filename)
            return self._empty_result(mask, m, s)

        p2p = np.median(np.abs(np.diff(baseline))) / np.sqrt(2)
        ratio = p2p / sigma
        survival_fraction = mask.sum() / self.flux.size
        n_high_cut = int(np.sum(self.flux > m + self.n_sigma * s))

        below = resid < -self.n_sigma * sigma
        runs, cur = [], []
        for i in np.where(below)[0]:
            if cur and (self.time[i] - self.time[cur[-1]]) > self.gap_tol * self.exp_time:
                runs.append(cur)
                cur = []
            cur.append(i)
        if cur:
            runs.append(cur)

        dips = [r for r in runs if len(r) >= self.need_len(r)]

        dip_stats = []
        for r in dips:
            r = np.asarray(r)
            f = self.flux[r]
            depth_flux = base_mean - f.min()
            mf_snr = (base_mean - f).sum() / (sigma * np.sqrt(r.size))
            dip_stats.append({
                "indices": r,
                "t_start": float(self.time[r[0]]),
                "t_end": float(self.time[r[-1]]),
                "t_centre": float(self.time[r].mean()),
                "n_points": int(r.size),
                "duration": float(self.time[r[-1]] - self.time[r[0]]),
                "depth_flux": float(depth_flux),
                "depth_frac": float(depth_flux / base_mean) if base_mean else np.nan,
                "depth_sigma": float(depth_flux / sigma),
                "mf_snr": float(mf_snr),
            })


        consistent_spacings = False
        spacing_frac_scatter = np.nan
        if len(dip_stats) >= 2:
            centres = np.sort(np.array([d["t_centre"] for d in dip_stats]))
            dt = np.diff(centres)
            dt = dt[dt > self.exp_time]
            if dt.size >= 2:
                spacing_frac_scatter = float(dt.std() / dt.mean())
                consistent_spacings = spacing_frac_scatter < self.spacing_tol

        span = self.time.max() - self.time.min()
        in_dip_time = sum(d["duration"] for d in dip_stats)
        duty_cycle = float(in_dip_time / span) if span > 0 else np.nan

        if dip_stats:
            best = max(dip_stats, key=lambda d: d["mf_snr"])
            max_run = max(d["n_points"] for d in dip_stats)
            score = best["mf_snr"] * (1.0 + self.runlen_bump * (max_run - 2))
            if consistent_spacings:
                score *= self.spacing_bump
        else:
            score = 0.0

        result = {
            "filename": self.filename,
            "baseline": float(base_mean),
            "sigma": float(sigma),
            "n_dips": len(dip_stats),
            "dips": dip_stats,
            "score": float(score),
            "quality": {
                "survival_fraction": float(survival_fraction),
                "n_high_cut": n_high_cut,
                "p2p_over_mad": float(ratio),
                "duty_cycle": duty_cycle,
                "spacing_frac_scatter": spacing_frac_scatter,
                "consistent_spacings": consistent_spacings,
            },
        }
        self.logger.info(
            "%s: %d dip(s), score=%.2f, survival=%.2f, p2p/MAD=%.2f, consistent_spacings=%s",
            self.filename, len(dip_stats), score, survival_fraction,
            ratio, consistent_spacings,
        )
        self.plot(result)
        return result

    def _empty_result(self, mask, m, s) -> dict:
        return {
            "filename": self.filename,
            "baseline": float(self.flux[mask].mean()),
            "sigma": 0.0,
            "n_dips": 0,
            "dips": [],
            "score": 0.0,
            "quality": {
                "survival_fraction": float(mask.sum() / self.flux.size),
                "n_high_cut": int(np.sum(self.flux > m + self.n_sigma * s)),
                "p2p_over_mad": np.nan,
                "duty_cycle": np.nan,
                "spacing_frac_scatter": np.nan,
                "consistent_spacings": False,
            },
        }