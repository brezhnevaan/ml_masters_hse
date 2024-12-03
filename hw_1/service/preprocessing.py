import numpy as np
import pandas as pd
import joblib
import re

scaler = joblib.load('scaler.pkl')
encoder = joblib.load('onehot_encoder.pkl')
loaded_medians = joblib.load('dict_medians.pkl')
owner_rank_mapping = joblib.load('owner_rank_mapping.pkl')

def input_to_df(input_data) -> np.array:
    if isinstance(input_data, dict):
        data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        data = input_data.copy()
    else:
        raise ValueError('input_data должен быть dict или csv')

    return data

def mileage_converter(row: pd.Series) -> float:
    fuel_densities = {
        'diesel': 0.850,
        'petrol': 0.745,
        'lpg': 0.540,
        'cng': 0.720
    }

    fuel = row['fuel'].lower()
    density = fuel_densities.get(fuel, None)
    mileage = row['mileage']

    if isinstance(mileage, str) and ' ' in mileage:
        value, mesure = mileage.split()
        value = float(value)

        if mesure == 'kmpl':
            return value
        elif mesure == 'km/kg':
            if density is not None:
                return value * density
            else:
                return None
        else:
            return None

    return None

def torque_processing(torque: str) -> list:
    values = []
    torque = str(torque).lower()

    first_value = re.match(r'(\d+\.?\d*)', torque)
    if first_value:
        if 'nm' in torque:
            values.append(float(first_value.group(1)))
        elif 'kg' in torque:
            values.append(round(float(first_value.group(1)) * 9.80665, 2))
        else:
            values.append(None)
    else:
        values.append(None)

    second_value = re.search(r'(?:\s|/)(\d[\d,-]*)', torque)
    if second_value:
        second_value_updated = second_value.group(1).replace(',', '').split('-')
        values.append(max([float(val) for val in second_value_updated]))
    else:
        values.append(None)

    return values

def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df['power_per_liter'] = df.max_power / df.engine

    return df

def feature_log(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    name = f'log_{feature}'
    df[name] = np.log(df[feature])
    df = df.drop(columns=[feature])

    return df

def feature_hyper(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    name = f'hp_{feature}'
    df[name] = df[feature] ** 3
    df = df.drop(columns=[feature])

    return df

def preprocess_numerical_features(data: pd.DataFrame, scaler) -> pd.DataFrame:
    numerical_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'owner']

    data['owner'] = data['owner'].map(owner_rank_mapping)
    data['mileage'] = data.apply(mileage_converter, axis=1)
    data_numerical = data[numerical_columns]

    data_numerical['engine'] = data_numerical['engine'].str.extract(r'(\d+)').astype(float)
    data_numerical['max_power'] = data_numerical['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
    data_numerical[['torque', 'max_torque_rpm']] = data_numerical.torque.apply(lambda x: torque_processing(x)).apply(pd.Series)

    data_numerical = data_numerical.fillna(loaded_medians)
    data_numerical['owner'] = data_numerical['owner'].astype(int)
    data_numerical['engine'] = data_numerical['engine'].astype(int)

    data_numerical = add_extra_features(data_numerical)
    data_numerical = feature_hyper(data_numerical, 'year')
    data_numerical = feature_log(data_numerical, 'engine')

    numerical_columns = ['km_driven', 'owner', 'mileage', 'max_power', 'torque', 'max_torque_rpm', 'power_per_liter', 'hp_year', 'log_engine']

    data_scaled = scaler.transform(data_numerical[numerical_columns])
    scaled_df = pd.DataFrame(data_scaled, columns=numerical_columns)

    return scaled_df

def preprocess_categorical_features(data: pd.DataFrame, encoder) -> pd.DataFrame:
    categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'seats']
    data_categorical = data[categorical_columns]

    for col in categorical_columns:
        if col == 'seats':data_categorical[col] = data_categorical[col].fillna(loaded_medians[col]).astype(int)
        else:
            data_categorical[col] = data_categorical[col].fillna('unknown')

    data_categorical['name'] = data_categorical.name.str.split().apply(lambda x: ' '.join(x[:2]))

    data_encoded = encoder.transform(data_categorical).toarray()
    encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out())

    return encoded_df

def preprocessing(input_data) -> pd.DataFrame:
    df = input_to_df(input_data)
    df_numerical = preprocess_numerical_features(df, scaler)
    df_categorical = preprocess_categorical_features(df, encoder)
    df_processed = pd.concat([df_categorical, df_numerical], axis=1)

    return df_processed