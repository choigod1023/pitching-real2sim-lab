"""Three-state evidence gates and conditional engineering noise calculations."""
import numpy as np

BODY_REQUIRED = ('raw_access', 'research_use_allowed', 'repeated_pitcher', 'trial_link',
                 'body_motion', 'independent_speed_link_units')
RELEASE_REQUIRED = BODY_REQUIRED + ('hand_observation', 'ball_center_observation',
                                    'physical_release_evidence', 'physical_sync_checked',
                                    'heldout_geometry_checked')


def readiness(evidence):
    if any(value not in ('TRUE', 'FALSE', 'UNKNOWN') for value in evidence.values()):
        raise ValueError('three_state_evidence_required')
    state = {key: evidence.get(key, 'UNKNOWN') for key in RELEASE_REQUIRED}
    return dict(body_speed_observational_ready=all(state[k] == 'TRUE' for k in BODY_REQUIRED),
                hand_ball_validation_ready=all(state[k] == 'TRUE' for k in RELEASE_REQUIRED),
                blocking_items={k: v for k, v in state.items() if v != 'TRUE'},
                scientific_gate_passed=False, athlete_effect_validated=False,
                coaching_recommendation_ready=False)


def slope_noise_sd_kmh(position_sd_mm, fps, samples):
    if not np.isfinite([position_sd_mm, fps, samples]).all() or position_sd_mm < 0 or fps <= 0 or int(samples) != samples or samples < 3:
        raise ValueError('invalid_slope_scenario')
    time = np.arange(int(samples))/fps
    return float(position_sd_mm/1000/np.sqrt(np.sum((time-time.mean())**2))*3.6)


def scale_bias_kmh(speed_kmh, relative_scale_error):
    if not np.isfinite([speed_kmh, relative_scale_error]).all() or speed_kmh < 0:
        raise ValueError('invalid_scale_scenario')
    return float(abs(speed_kmh*relative_scale_error))


def timing_budget_ms(speed_error_budget_kmh, acceleration_mps2):
    if not np.isfinite([speed_error_budget_kmh, acceleration_mps2]).all() or speed_error_budget_kmh < 0 or acceleration_mps2 <= 0:
        raise ValueError('invalid_timing_scenario')
    return float(speed_error_budget_kmh/3.6/acceleration_mps2*1000)
