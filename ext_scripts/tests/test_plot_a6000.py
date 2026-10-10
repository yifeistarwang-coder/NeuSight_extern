import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

from a6000_common import workloads
from plot_a6000 import (
    CATEGORIES, common_statistics, operator_statistics, plot_common, plot_e2e,
    plot_operators, validate_e2e,
)


def make_e2e():
    rows = []
    training_count = 0
    for workload in workloads():
        complete = workload["mode"] == "inf" or training_count < 4
        if workload["mode"] == "train":
            training_count += 1
        row = {**workload, "prediction_status": "success", "predicted_e2e_ms": 12.,
               "status": "success" if complete else "oom", "valid_repeats": 3 if complete else 0,
               "median_e2e_ms": 11. if complete else np.nan,
               "min_e2e_ms": 10. if complete else np.nan, "max_e2e_ms": 12. if complete else np.nan,
               "median_e2e_ape_pct": 100. / 11. if complete else np.nan}
        for run, measured in enumerate((10., 11., 12.), 1):
            row[f"run_{run:02d}_status"] = row["status"]
            row[f"run_{run:02d}_e2e_ms"] = measured if complete else np.nan
            row[f"run_{run:02d}_e2e_ape_pct"] = abs(12. - measured) / measured * 100 if complete else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_operators(e2e):
    rows = []
    for position, workload in enumerate(e2e[e2e["status"] == "success"].to_dict("records")):
        for category in CATEGORIES:
            zero = workload["mode"] == "inf" and position < 2 and category == "VECln"
            rows.append({"name": workload["name"], "mode": workload["mode"], "category": category,
                         "phase": "e2e", "valid_repeats": 3, "predicted_ms": 2.,
                         "median_measured_ms": 0. if zero else 1., "ape_pct": np.nan if zero else 100.})
    return pd.DataFrame(rows)


def make_common(e2e):
    rows = []
    for mode in ("inf", "train"):
        selected = e2e[e2e["mode"].eq(mode) & e2e["status"].eq("success")].head(3)
        for device in ("NVIDIA_H100_80GB_HBM3", "NVIDIA_L4"):
            for row in selected.to_dict("records"):
                rows.append({"mode": mode, "reference_device": device, "name": row["name"],
                             "scope": "all_gpu_common", "a6000_ape_pct": row["median_e2e_ape_pct"],
                             "reference_ape_pct": 5.})
    return pd.DataFrame(rows)


class A6000PlotTests(unittest.TestCase):
    def setUp(self):
        self.e2e = make_e2e()

    def test_oom_excluded_from_primary_mape(self):
        metrics = validate_e2e(self.e2e).set_index("mode")
        self.assertEqual(metrics.loc["inf", "valid_count"], 12)
        self.assertEqual(metrics.loc["train", "valid_count"], 4)
        self.assertAlmostEqual(metrics.loc["train", "mape_pct"], 100. / 11.)

    def test_failed_configuration_must_not_have_zero_ape(self):
        self.e2e.loc[self.e2e["status"] == "oom", "median_e2e_ape_pct"] = 0.
        with self.assertRaisesRegex(ValueError, "must not have a primary APE"):
            validate_e2e(self.e2e)

    def test_incomplete_success_rejected(self):
        self.e2e.loc[0, "valid_repeats"] = 2
        self.e2e.loc[0, "run_03_status"] = "oom"
        with self.assertRaisesRegex(ValueError, "Incomplete successful"):
            validate_e2e(self.e2e)

    def test_non_positive_measurement_rejected(self):
        self.e2e.loc[0, "run_01_e2e_ms"] = 0.
        with self.assertRaisesRegex(ValueError, "Non-positive"):
            validate_e2e(self.e2e)

    def test_incorrect_ape_rejected(self):
        self.e2e.loc[0, "median_e2e_ape_pct"] = 0.
        with self.assertRaisesRegex(ValueError, "primary APE"):
            validate_e2e(self.e2e)

    def test_zero_operator_labels_excluded(self):
        metrics = operator_statistics(make_operators(self.e2e), self.e2e).set_index(["mode", "category"])
        self.assertEqual(metrics.loc[("inf", "VECln"), "valid_count"], 10)
        self.assertEqual(metrics.loc[("train", "VECln"), "valid_count"], 4)
        self.assertEqual(metrics.loc[("inf", "VECln"), "mape_pct"], 100.)
        self.assertEqual(metrics.loc[("inf", "VECln"), "mean_absolute_error_ms"], 1.)

    def test_zero_filled_operator_ape_rejected(self):
        medians = make_operators(self.e2e)
        medians.loc[medians["median_measured_ms"] == 0, "ape_pct"] = 0.
        with self.assertRaisesRegex(ValueError, "Zero operator labels"):
            operator_statistics(medians, self.e2e)

    def test_common_comparison_uses_equal_workload_sets(self):
        details = make_common(self.e2e)
        result = common_statistics(details, self.e2e)
        self.assertEqual(len(result), 6)
        self.assertTrue(result["valid_count"].eq(3).all())
        with self.assertRaisesRegex(ValueError, "Unequal workload sets"):
            common_statistics(details.drop(index=0), self.e2e)

    def test_comparison_cannot_include_oom(self):
        details = make_common(self.e2e)
        details.loc[0, "name"] = self.e2e[self.e2e["status"] == "oom"].iloc[0]["name"]
        with self.assertRaisesRegex(ValueError, "unsuccessful"):
            common_statistics(details, self.e2e)

    def test_figures_render_and_text_fits(self):
        metrics = validate_e2e(self.e2e)
        operators = operator_statistics(make_operators(self.e2e), self.e2e)
        common = common_statistics(make_common(self.e2e), self.e2e)
        figures = (plot_e2e(self.e2e, metrics), plot_operators(operators), plot_common(common))
        try:
            for figure in figures:
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                bounds = figure.bbox
                for text in figure.findobj(Text):
                    if text.get_visible() and text.get_text():
                        box = text.get_window_extent(renderer)
                        self.assertGreaterEqual(box.x0, bounds.x0 - 1, text.get_text())
                        self.assertGreaterEqual(box.y0, bounds.y0 - 1, text.get_text())
                        self.assertLessEqual(box.x1, bounds.x1 + 1, text.get_text())
                        self.assertLessEqual(box.y1, bounds.y1 + 1, text.get_text())
                image = io.BytesIO()
                figure.savefig(image, format="png", dpi=100)
                pixels = np.array(Image.open(image).convert("RGB"))
                self.assertGreater((pixels < 240).any(axis=2).mean(), .03)
        finally:
            for figure in figures:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
