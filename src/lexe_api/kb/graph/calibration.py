"""
Isotonic Calibration for Category Graph v2.5

Calibra confidence scores usando isotonic regression per ridurre ECE.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Calibra confidence scores usando isotonic regression."""

    def __init__(self):
        self.model: Optional[IsotonicRegression] = None
        self.version: str = ""
        self.trained_on: str = ""

    def fit(self, scores_raw: np.ndarray, labels_correct: np.ndarray) -> None:
        """
        Train su golden set (solo train split!).

        Args:
            scores_raw: Array di confidence scores grezzi (0-1)
            labels_correct: Array binario (1 = predizione corretta, 0 = errata)
        """
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(scores_raw, labels_correct)
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trained_on = datetime.now().isoformat()

    def calibrate(self, score_raw: float) -> float:
        """
        Trasforma score grezzo in confidence calibrata.

        Args:
            score_raw: Confidence score grezzo (0-1)

        Returns:
            Confidence calibrata (0-1)
        """
        if self.model is None:
            return score_raw  # Fallback se non trainato
        return float(self.model.predict([score_raw])[0])

    def calibrate_batch(self, scores_raw: np.ndarray) -> np.ndarray:
        """
        Calibra un batch di scores.

        Args:
            scores_raw: Array di scores grezzi

        Returns:
            Array di scores calibrati
        """
        if self.model is None:
            return scores_raw
        return self.model.predict(scores_raw)

    def save(self, path: Path) -> None:
        """
        Salva calibratore per produzione.

        Args:
            path: Path del file JSON di output
        """
        if self.model is None:
            raise ValueError("Cannot save untrained calibrator")

        data = {
            "version": self.version,
            "trained_on": self.trained_on,
            "x_thresholds": self.model.X_thresholds_.tolist(),
            "y_thresholds": self.model.y_thresholds_.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        """
        Carica calibratore salvato.

        Args:
            path: Path del file JSON da caricare
        """
        data = json.loads(path.read_text())
        self.version = data["version"]
        self.trained_on = data["trained_on"]
        self.model = IsotonicRegression(out_of_bounds="clip")
        # Reconstruct the fitted model
        self.model.X_thresholds_ = np.array(data["x_thresholds"])
        self.model.y_thresholds_ = np.array(data["y_thresholds"])
        # Set the necessary attributes for prediction
        self.model.X_min_ = self.model.X_thresholds_[0]
        self.model.X_max_ = self.model.X_thresholds_[-1]
        self.model.f_ = lambda x: np.interp(x, self.model.X_thresholds_, self.model.y_thresholds_)


# Singleton globale
_calibrator: Optional[IsotonicCalibrator] = None


def get_calibrator(cal_path: Optional[Path] = None) -> IsotonicCalibrator:
    """
    Get or create singleton calibrator.

    Args:
        cal_path: Optional path to calibrator file. Defaults to data/calibrator_v1.json

    Returns:
        IsotonicCalibrator instance
    """
    global _calibrator
    if _calibrator is None:
        _calibrator = IsotonicCalibrator()
        # Prova a caricare da file se esiste
        if cal_path is None:
            cal_path = Path("data/calibrator_v1.json")
        if cal_path.exists():
            try:
                _calibrator.load(cal_path)
            except Exception as e:
                print(f"Warning: Failed to load calibrator from {cal_path}: {e}")
    return _calibrator


def reset_calibrator() -> None:
    """Reset the singleton calibrator (useful for testing)."""
    global _calibrator
    _calibrator = None


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        confidences: Array di confidence scores (0-1)
        accuracies: Array binario (1 = corretto, 0 = errato)
        n_bins: Numero di bin per il calcolo

    Returns:
        ECE score (lower is better, 0 = perfectly calibrated)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += np.abs(avg_conf - avg_acc) * prop_in_bin
    return ece
