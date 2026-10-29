"""Block retired prediction routes; permit explicit historical diagnostics only."""
import copy
from functools import wraps

MODEL_ID = 'legacy_fixed_grip_15cm'
RETIREMENT_MESSAGE = (
    '고정 그립 모델은 구속 예측과 자세 추천 경로에서 제외했습니다. '
    '과거 진단을 재현하려면 --legacy-diagnostic를 명시하세요. '
    '이 옵션은 예측 또는 추천 모델을 다시 활성화하지 않습니다.'
)


class RetiredPredictionModelError(RuntimeError):
    """An unsupported prediction route was requested without diagnostic opt-in."""


def model_usage_metadata():
    return {'model_id': MODEL_ID, 'model_status': 'retired', 'diagnostic_only': True,
            'prediction_available': False, 'coaching_recommendation_ready': False,
            'predicted_ball_speed_kmh': None, 'predicted_speed_gain_kmh': None}


def require_legacy_diagnostic(enabled=False):
    if enabled is not True:
        raise RetiredPredictionModelError(RETIREMENT_MESSAGE)
    return model_usage_metadata()


def legacy_diagnostic_only(function):
    @wraps(function)
    def wrapped(*args, legacy_diagnostic=False, **kwargs):
        usage = require_legacy_diagnostic(legacy_diagnostic)
        result = function(*args, **kwargs)
        if isinstance(result, dict):
            result.update(usage)
        return result
    return wrapped


def diagnostic_result(result):
    """Do not export assumed-point calculations under prediction-like field names."""
    annotated = copy.deepcopy(result)
    annotated.update(model_usage_metadata())
    annotated['human_trial_ready'] = False
    renames = {'release_speed_kmh': 'diagnostic_assumed_point_speed_kmh',
               'plate_speed_kmh': 'diagnostic_plate_speed_kmh',
               'speed_change_kmh': 'diagnostic_speed_difference_kmh',
               'target_error_cm': 'diagnostic_target_error_cm',
               'target_error_reduction_cm': 'diagnostic_target_difference_cm'}
    for row in annotated.get('records', []):
        for original, renamed in renames.items():
            if original in row:
                row[renamed] = row.pop(original)
        row.update(model_usage_metadata())
        row['human_trial_ready'] = False
    return annotated
