import logging
from src.utils.store import PipelineStore
from src.utils.config import load_config
from src.models.metrics import compute_metrics_collection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    logger = logging.getLogger(__name__)
    logger.info('building features from processed data')

    store = PipelineStore()
    config = load_config()

    # get the data
    transformed_data = store.get_processed('car_price_transformed.csv')

    # train/test split the data
    if len(config['features']) == 0:
        X = transformed_data.drop(config['target'], axis=1)
    else:
        X = transformed_data[config['features']]
    y = transformed_data[config['target']]

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, **config['train_test_split'])

    # train the model
    lm = LinearRegression(**config['linear_regression'])
    lm.fit(train_x, train_y)

    # assess performance
    train_metrics = compute_metrics_collection(train_y, lm.predict(train_x))
    test_metrics = compute_metrics_collection(test_y, lm.predict(test_x))

    # save the model and the metrics
    store.put_model('saved_model.pkl', lm)
    store.put_metrics('train_metrics.json', train_metrics)
    store.put_metrics('test_metrics.json', test_metrics)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
