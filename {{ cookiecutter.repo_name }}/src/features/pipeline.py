import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.features.dataframe_transformer import DataFrameFunctionTransformer
from src.features.dataframe_transformer import DataFrameSKLearnTransformer


@DataFrameFunctionTransformer
def log_car_price(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df_copy = input_df.copy()
    input_df_copy['log_price'] = np.log(input_df_copy['price'])
    return input_df_copy


@DataFrameFunctionTransformer
def translate_car_numbers(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df_copy = input_df.copy()
    cardoor_mapping = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8
        }

    input_df_copy['doornumber'] = input_df['doornumber'].map(cardoor_mapping)
    return input_df_copy


@DataFrameFunctionTransformer
def scale_numeric_input(input_df: pd.DataFrame) -> pd.DataFrame:
    numeric_input = input_df.select_dtypes(include='number')

    # DANGER: This is actually wrong, since I think it would refit
    #         every single time. We should only have transform after
    #         we have fitted.
    #
    #         If we are able to save this feature_pipeline and apply
    #         it on the new data, then there is no problem.
    from sklearn.preprocessing import StandardScaler
    scaler = DataFrameSKLearnTransformer(StandardScaler())
    return scaler.fit_transform(numeric_input)


# How can I keep the transformation pipeline consistent over different dataset?

feature_pipeline = Pipeline(
    steps=[
        ('log_price', log_car_price),
        ('translate_cardoor_number', translate_car_numbers),
        ('scale_numeric', scale_numeric_input)
    ]
)
