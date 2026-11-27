import argparse
import csv
import math


class RobustDistanceFilter:
    def __init__(self, alpha=0.1, jump_threshold=0.12,
                 confirm_samples=3, confirm_tolerance=0.08, max_gap_s=0.5):
        self.alpha = alpha
        self.jump_threshold = jump_threshold
        self.confirm_samples = confirm_samples
        self.confirm_tolerance = confirm_tolerance
        self.max_gap_s = max_gap_s
        self.reset()

    def reset(self):
        self.value = None
        self.last_time = None
        self.pending_value = None
        self.pending_count = 0

    def _clear_pending(self):
        self.pending_value = None
        self.pending_count = 0

    def update(self, time_s, observed_m):
        if observed_m is None:
            return None

        z = float(observed_m)
        if not math.isfinite(z) or z <= 0:
            return None

        if (self.value is None or time_s - self.last_time > self.max_gap_s):
            self.value = z
            self._clear_pending()
        elif abs(z - self.value) <= self.jump_threshold:
            self.value = (1 - self.alpha) * self.value + self.alpha * z
            self._clear_pending()
        else:
            if (self.pending_value is None or abs(z - self.pending_value) > self.confirm_tolerance):
                self.pending_value = z
                self.pending_count = 1
            else:
                self.pending_count += 1
                self.pending_value += (z - self.pending_value) / self.pending_count

            if self.pending_count >= self.confirm_samples:
                self.value = z
                self._clear_pending()

        self.last_time = time_s
        return self.value


def run_filter_test(output_csv="filter_test.csv", alpha=0.10):
    distance_filter = RobustDistanceFilter(alpha=alpha)
    rows = []

    for n in range(601):
        t = n / 100
        reference = 2.50 if n < 200 else 3.00
        observed = (
            reference
            + 0.02 * math.sin(6 * math.pi * t)
            + 0.01 * math.sin(34 * math.pi * t)
        )
        if n == 150:
            observed += 0.40

        filtered = distance_filter.update(t, observed)
        rows.append((n, t, reference, observed, filtered))

    with open(f"./output/{output_csv}", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["n", "time_s", "reference_m", "observed_m", "filtered_m"])
        writer.writerows(rows)

    def mae(lo, hi, column):
        return sum(abs(rows[n][column] - rows[n][2]) for n in range(lo, hi + 1)) / (hi - lo + 1)

    delay = None
    for n in range(200, len(rows) - 2):
        if all(abs(rows[j][4] - 3.00) <= 0.05 for j in (n, n + 1, n + 2)):
            delay = (n - 200) / 100
            break

    metrics = {
        "stable_0_99_mae_raw": mae(0, 99, 3),
        "stable_0_99_mae_filtered": mae(0, 99, 4),
        "stable_300_399_mae_raw": mae(300, 399, 3),
        "stable_300_399_mae_filtered": mae(300, 399, 4),
        "outlier_error_raw": abs(rows[150][3] - rows[150][2]),
        "outlier_error_filtered": abs(rows[150][4] - rows[150][2]),
        "step_delay_s": delay,
    }
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved {len(rows)} rows to {output_csv}")
    return metrics


if __name__ == "__main__":
    run_filter_test("filter_test.csv", 0.1)