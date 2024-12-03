from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset, RegressionPreset, ClassificationPreset
from evidently import ColumnMapping

from ModelMonitoring.ModelMonitoring.utils.get_data import run_query
from ModelMonitoring.ModelMonitoring.utils.run_report import Model
from ModelMonitoring.ModelMonitoring.utils.export import Export

if __name__ == '__main__':
    # Pseudocode:

    WORKSPACE = '' # path to folder where all analytics are stored
    DATABASE = '' # path to SQL database

    target = ''
    prediction = ''
    numerical_features = ['']
    categorical_features = ['']

    reference_data, current_data = run_query()

    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    model = Model(WORKSPACE, column_mapping)

    model.generate_report('drift', [DataDriftPreset()], reference_data, current_data)
    model.generate_report('quality', [DataQualityPreset()], reference_data, current_data)
    model.generate_report('target', [TargetDriftPreset()], reference_data, current_data)

    # chose one or the other
    model.generate_report('regression', [RegressionPreset()], reference_data, current_data)
    model.generate_report('classification', [ClassificationPreset()], reference_data, current_data)

    export = Export(WORKSPACE)

    export.connect(DATABASE)
    export.setup_database()

    export.data_drift_report('drift.json')
    export.quality_report('quality.json')
    export.target_drift_report('target.json', target, prediction)

    # chose on or the other
    export.regression_model_report('regression.json')
    export.classification_model_report('classification.json')

    export.disconnect()