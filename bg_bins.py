import numpy as np
import logging

class binning:
    def __init__(self, logger: logging.Logger, x: np.ndarray, y: np.ndarray, bin_fact: int):
        self.logger = logger
        self.x = x
        self.y = y
        self.bin_fact = bin_fact
    @staticmethod
    def bin_time_flux_error(x, y, binfact) -> tuple[np.ndarray, np.ndarray]:
        n_bins = len(x) // binfact
        trim = n_bins * binfact
        x_b = np.mean(x[:trim].reshape(n_bins, binfact), axis=1)
        y_b = np.mean(y[:trim].reshape(n_bins, binfact), axis=1)
        return x_b, y_b

    def nightly_bin_lc(self) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info("Starting nightly binning of light curve")
        diff = np.diff(self.x)
        night_loc = np.where(diff > 0.5)[0]
        night_loc = np.insert(night_loc, 0, 0)
        night_loc = np.append(night_loc, len(self.x))
        bjdn_b       = np.array([])
        ratio_norm_b = np.array([])
        for i in range(1, len(night_loc)):
            sl = slice(night_loc[i-1], night_loc[i])
            bjdn_sl = self.x[sl]
            ratio_norm_sl = self.y[sl]
            if len(bjdn_sl) == 0:
                continue
            if len(bjdn_sl) < self.bin_fact:
                bjdn_b       = np.append(bjdn_b, bjdn_sl)
                ratio_norm_b = np.append(ratio_norm_b, ratio_norm_sl)
                continue
            try:
                bjdn_sl_b, ratio_norm_sl_b = self.bin_time_flux_error(bjdn_sl, ratio_norm_sl, self.bin_fact)
            except Exception as e:
                self.logger.info(f"Skipping night {i} due to error: {e}")
                continue
            bjdn_b       = np.append(bjdn_b, bjdn_sl_b)
            ratio_norm_b = np.append(ratio_norm_b, ratio_norm_sl_b)
        return bjdn_b, ratio_norm_b

    def bin_data_on_phase(self) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info("Starting phase binning of light curve")
        if len(self.x) <= (len(self.bin_fact) - 1):
            self.logger.info("Not enough data points to bin with the specified bin factor.")
            return self.x, self.y
        n_binned = int(len(self.x) / self.bin_fact)
        binned_len = int(n_binned * self.bin_fact)
        temp = sorted(zip(self.x[:binned_len], self.y[:binned_len]))
        phase_s, flux_s = map(np.array, zip(*temp))
        phase_bin = np.average(phase_s.reshape(n_binned, self.bin_fact), axis=1)
        flux_bin = np.average(flux_s.reshape(n_binned, self.bin_fact), axis=1)
        self.logger.info(f"Completed phase binning into {n_binned} bins with bin factor {self.bin_fact}")
        return phase_bin, flux_bin