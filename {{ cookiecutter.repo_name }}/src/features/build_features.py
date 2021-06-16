import logging
from src.utils.store import PipelineStore
from src.features.pipeline import feature_pipeline


def main():
    logger = logging.getLogger(__name__)
    logger.info('building features from processed data')

    # HINT: You can use the dataframe_transformer to create dataframe
    #       compatible SKLearn pipelines.

    # get the data
    store = PipelineStore()
    processed_data = store.get_processed('car_price_processed.csv')

    # build new feature
    transformed_data = feature_pipeline.fit_transform(processed_data)

    # save the data back
    store.put_processed('car_price_transformed.csv', transformed_data)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
