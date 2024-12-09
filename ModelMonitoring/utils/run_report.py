import pandas as pd

from evidently.report import Report
from evidently import ColumnMapping

class Model:
    def __init__(self, workspace: str, column_mapping: ColumnMapping):
        """
        methods to handle generation of evidently reports

        example usage:
            raw_data = pd.read_csv('raw_data.csv')

            target = 'actual_value'
            prediction = 'model_prediction'
            numerical_features = ['temperature', 'time', 'wind_speed']
            categorical_features = ['season', 'is_holiday']

            reference = raw_data.loc['training_data']
            current = raw_data.loc['live_data']

            model = Model('workspace/model1')
            model.map_columns(target, prediction, numerical_features, categorical_features)

            model.generate_report('drift', [DataDriftPreset()], reference, current)
            model.generate_report('regression', [RegressionPreset()], reference, current)

        :param workspace: file path to model folder
        :param column_mapping: column_mapping object in evidently
        """
        self.SAVE_PATH = workspace
        self.column_mapping = column_mapping

    @staticmethod
    def __create_report_object(metrics: list):
        """
        There are custom options for evidently reports. Currently, only the colour schema is changed, updated to
        Equitable teal.

        :param metrics: same as when using Report(metrics)
        :return: evidently Report object
        """
        from evidently.options import ColorOptions

        color_scheme = ColorOptions(primary_color='#00a4ac')

        report = Report(metrics=metrics, options=[color_scheme])

        return report

    def generate_report(self, name: str, metrics: list, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
        """
        Generates desired report for model, saves it as JSON file

        :param name: filename to save as
        :param metrics: list of metrics to pass into evidently Report object
        :param reference_data: training data
        :param current_data: current data
        """

        report = self.__create_report_object(metrics=metrics)
        report.run(current_data=current_data, reference_data=reference_data, column_mapping=self.column_mapping)

        report.save_json(self.SAVE_PATH + '/' + name + '.json')
        report.save_html(self.SAVE_PATH + '/' + name + '.html') # remove for prod
