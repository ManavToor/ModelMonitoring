from ModelMonitoring.ModelMonitoring.utils import get_data
from ModelMonitoring.ModelMonitoring.utils.run_report import Model
from ModelMonitoring.ModelMonitoring.utils.export import Export

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset

def juvenile_standard_life(year_month: str = None, ref_ratio: float = 0.7) -> None:
    """
    run report on juvenile standard life model
    :param year_month: if report for a specific month is required, provide as "YYYY-MM" or else data will be all time
    :param ref_ratio: default ref 70% (cur 30%), change if different ratio required, pass None to not split
    """
    model_name = 'juvenile_standard_life'

    ref, cur = get_data.run_query(model_name, year_month)

    target = 'target'
    target_names = ['0', '1']
    prediction = 'prediction'
    numerical = ['numerical', 'features']
    categorical = ['categorical', 'features']
    id_col = 'policy_person_id'

    # drop all values with missing target
    cur = cur.dropna(subset=[target])

    ref, cur = get_data.process_data(ref, cur)

    # set target and prediction to strings
    ref[target] = ref[target].astype('object')
    ref[prediction] = ref[prediction].astype('object')
    cur[target] = cur[target].astype('object')
    cur[prediction] = cur[prediction].astype('object')

    # find empty columns
    empty_ref = ref.columns[ref.isnull().all()].tolist()
    empty_cur = cur.columns[cur.isnull().all()].tolist()
    empty = empty_cur + empty_ref

    # remove empty columns in column map
    for i in empty:
        if i in numerical:
            numerical.remove(i)
        elif i in categorical:
            categorical.remove(i)

    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.target_names = target_names
    column_mapping.prediction = prediction
    column_mapping.id = id_col
    column_mapping.numerical_features = numerical
    column_mapping.categorical_features = categorical

    model = Model(column_mapping)
    export = Export(model_name, monthly=(False if year_month is None else True))
    export.connect()
    #export.setup_database()

    if ref_ratio is not None:
        ref, cur = get_data.split_data(ref, cur, ref_ratio)

    quality = model.generate_report([DataQualityPreset()], ref, cur)
    export.quality_report(quality)

    classification = model.generate_report([ClassificationPreset()], ref, cur)
    export.classification_model_report(classification)

    # drift requires non-empty columns
    ref = ref.drop(columns=empty)
    cur = cur.drop(columns=empty)
    drift = model.generate_report([DataDriftPreset()], ref, cur)
    export.data_drift_report(drift)

    # target by feature
    target_by_feature = {}
    list_of_features = numerical + categorical
    for feature in list_of_features:
        target_by_feature[feature] = {}
        target_by_feature[feature]['ref'] = Model.target_by_feature(ref, feature, int, target, str)
        target_by_feature[feature]['cur'] = Model.target_by_feature(cur, feature, int, target, str)
    export.target_by_feature_report(target_by_feature)

    export.disconnect()
