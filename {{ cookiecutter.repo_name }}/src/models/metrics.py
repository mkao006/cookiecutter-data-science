from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    max_error
)


def compute_metrics_collection(actual, prediction):
    return {
        'primary_metric': {
            'explained_variance_score': explained_variance_score(actual, prediction)
        },
        'secondary_metric': {
            'mean_squared_error': mean_squared_error(actual, prediction),
            'max_error': max_error(actual, prediction)
        }
    }
