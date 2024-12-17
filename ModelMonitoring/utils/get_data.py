import pandas as pd
import pyodbc
import json

from . import MODEL_EVIDENTLY_ODBC

def __ref_query(table_name: str, conn) -> pd.DataFrame:

    # left join because target and predictions are in a different table
    query = f"""
    SELECT ref.{table_name}.*, ref.predictions_target_{table_name}.target, ref.predictions_target_{table_name}.prediction
    FROM ref.{table_name}
    LEFT JOIN ref.predictions_target_{table_name} ON ref.{table_name}.policy_person_id = ref.predictions_target_{table_name}.policy_person_id"""
    ref_df = pd.read_sql(query, conn)
    ref_df['prediction'] = ref_df['prediction'].replace({'STANDARD': 0, 'NONSTANDARD': 1})

    return ref_df

def __cur_query(table_name: str, conn, year_month: str=None) -> pd.DataFrame:
    # there are some policies that the model does not predict on
    query = f"""
    SELECT 
        a.policy_person_id, 
        a.encoding, 
        a.prediction, 
        b.decision
    FROM 
        prd.{table_name} a
    LEFT JOIN 
        prd.target_{table_name} b
    ON 
        a.policy_person_id = b.policy_person_id 
    WHERE 
        a.prediction IS NOT NULL"""
    if year_month is not None:
        query = query + f" AND a.est_request_date LIKE '{year_month}%'"
    cur_df = pd.read_sql(query, conn)

    # encoding column is a JSON vector containing all features
    cur_df['encoding'] = cur_df['encoding'].apply(json.loads)
    encoding_df = pd.json_normalize(cur_df['encoding'])
    result_cur_df = pd.concat([cur_df[['policy_person_id', 'prediction', 'decision']], encoding_df], axis=1)

    result_cur_df['decision'] = result_cur_df['decision'].replace({'STANDARD': 0, 'NONSTANDARD': 1})
    result_cur_df['prediction'] = result_cur_df['prediction'].replace({'STANDARD': 0, 'NONSTANDARD': 1})
    result_cur_df.rename(columns={'decision': 'target'}, inplace=True)

    return result_cur_df

def run_query(table_name: str, year_month: str=None) -> (pd.DataFrame, pd.DataFrame):
    """
    get data from SQL server or wherever data is stored
    data should contain training data and current data with model predictions

    :param table_name: name of table (excluding "ref." or "prd.")
    :param year_month: "YYYY-MM" for specific month's of data (ie '2024-10')
    :return reference data (including model predictions), current data
    """

    conn = pyodbc.connect(MODEL_EVIDENTLY_ODBC)

    ref_df = __ref_query(table_name, conn)
    cur_df = __cur_query(table_name, conn, year_month)

    conn.close()

    return ref_df, cur_df

def process_data(ref: pd.DataFrame, cur: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Processes incoming data:
        -   reorders cur df to match ref
        -   sets all datatypes to match, removes 'object' types

    :param ref: ref data
    :param cur: cur data
    :return: ref, cur
    """

    # reorder columns to match
    cur = cur[ref.columns]

    # cur data comes from JSON vector, match all datatypes to ref (remove object types as well)
    # FUTURE: update to handle broader cases not have each edge case as an if statement
    for col in ref.columns:
        ref_type = ref[col].dtype
        cur_type = cur[col].dtype

        if (ref_type == 'int64' or ref_type == 'Int64') and cur_type == 'float64':
            ref[col] = ref[col].astype('Int64')
            cur[col] = cur[col].astype('Int64')
        elif ref_type == 'float64' and (cur_type == 'int64' or cur_type == 'Int64'):
            cur[col] = cur[col].astype('float64')
        elif ref_type == 'object' and cur_type == 'float64':
            ref[col] = ref[col].astype('float64')
        elif ref_type == 'float64' and cur_type == 'object':
            cur[col] = cur[col].astype('float64')
        elif ref_type == 'bool' and cur_type == 'float64':
            cur[col] = cur[col].astype('bool')

    return ref, cur

def split_data(ref: pd.DataFrame, cur: pd.DataFrame, ref_ratio: float) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the data. Can handle cases where ref>>cur and vise versa

    :param ref: ref data
    :param cur: cur data
    :param ref_ratio: percentage of data that is ref (ie 0.7)
    :return: ref, cur
    """
    # reduce ref and cur to desired ratio (default ref 70% cur 30%)
    total_samples = len(ref) + len(cur)
    ref_samples, cur_samples = int(total_samples * ref_ratio), int(total_samples * (1 - ref_ratio))
    ref_samples, cur_samples = min(ref_samples, len(ref)), min(cur_samples, len(cur))
    if ref_samples + cur_samples < total_samples:
        if len(ref) > len(cur):
            ref_samples = int((cur_samples / (1 - ref_ratio)) * ref_ratio)
        else:
            cur_samples = int((ref_samples / ref_ratio) * (1 - ref_ratio))

    ref = ref.sample(n=ref_samples)
    cur = cur.sample(n=cur_samples)

    return ref, cur
