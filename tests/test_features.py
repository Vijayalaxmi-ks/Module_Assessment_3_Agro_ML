import numpy as np
import pytest

def calculate_vpd(temp_c: float, humidity_pct: float) -> float:
    """Calculates Vapor Pressure Deficit (VPD) in kPa."""
    svp = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    avp = svp * (humidity_pct / 100.0)
    return float(np.round(svp - avp, 3))

def test_vpd_calculation_known_values():
    vpd = calculate_vpd(25.0, 50.0)
    assert pytest.approx(vpd, abs=0.05) == 1.584

def test_vpd_zero_at_100_percent_humidity():
    vpd = calculate_vpd(30.0, 100.0)
    assert vpd == 0.0

def test_total_npk():
    n, p, k = 120.0, 45.0, 180.0
    total_npk = n + p + k
    assert total_npk == 345.0

def test_ph_deviation():
    neutral_ph = 7.0
    alkaline_ph = 8.5
    acidic_ph = 5.5

    assert abs(neutral_ph - 7.0) == 0.0
    assert abs(alkaline_ph - 7.0) == 1.5
    assert abs(acidic_ph - 7.0) == 1.5