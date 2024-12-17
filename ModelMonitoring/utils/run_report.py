import pandas as pd

from evidently.report import Report
from evidently import ColumnMapping

from datetime import datetime


class Model:
    def __init__(self,column_mapping: ColumnMapping):
        """
        methods to handle generation of evidently reports,
        target by feature does not save in evidently reports, so a custom function was built

        example usage:
            raw_data = pd.read_csv('raw_data.csv')

            column_mapping = ColumnMapping()
            column_mapping.target = 'actual_value'
            column_mapping.prediction = 'model_prediction'

            reference = raw_data.loc['training_data']
            current = raw_data.loc['live_data']

            model = Model(column_mapping)
            model.map_columns(target, prediction, numerical_features, categorical_features)

            drift_data = model.generate_report([DataDriftPreset()], reference, current)
            regression_data = model.generate_report([RegressionPreset()], reference, current)

        :param column_mapping: column_mapping object in evidently
        """

        self.column_mapping = column_mapping

    @staticmethod
    def __create_report_object(metrics: list):
        """
        There are custom options for evidently html visual reports.
        -   Set colour scheme to equitable teal.
        -   Turn off aggregated data (when this is toggled on, evidently removes metrics in favour of time)

        :param metrics: same as when using Report(metrics)
        :return: evidently Report object
        """
        from evidently.options import ColorOptions
        from evidently.options.agg_data import RenderOptions

        color_scheme = ColorOptions(primary_color='#00a4ac')
        # setting this to true will mean better accuracy, but report will take forever to run, default is False
        render = RenderOptions(raw_data=False)

        report = Report(metrics=metrics, options=[color_scheme, render])

        return report

    def generate_report(self, metrics: list, reference_data: pd.DataFrame, current_data: pd.DataFrame, savepath: str = None) -> str:
        """
        Generates desired report for model, returns a JSON string object

        :param metrics: list of metrics to pass into evidently Report object
        :param reference_data: training data
        :param current_data: current data
        :param savepath: incase a html visual is required, savepath for visual C:/Path/Filename
        """

        report = self.__create_report_object(metrics=metrics)
        report.run(current_data=current_data, reference_data=reference_data, column_mapping=self.column_mapping)

        # for debugging, or incase someone wants a quick visual
        if savepath is not None:
            report.save_html(f"{savepath}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.html")

        return report.json()

    @staticmethod
    def target_by_feature(data: pd.DataFrame, feature: str, feature_type, target: str, target_type) -> dict:
        """
        Target by feature metric does not save to JSON (likely a bug in evidently), this function performs the metric

        :param data: dataset, either reference or current
        :param feature: name of feature column (this function only does one feature at a time)
        :param feature_type: data type of feature
        :param target: name of target column
        :param target_type: data type of column
        :return: for each target value, every feature value and it's occurrence
        """

        data = data[[feature, target]].dropna(subset=[feature])
        
        # Convert columns to the appropriate types
        data[feature] = data[feature].apply(feature_type)
        data[target] = data[target].apply(target_type)

        # Group by target and feature, then count occurrences
        grouped = data.groupby([target, feature]).size().unstack(fill_value=0)

        # Convert the DataFrame to a nested dictionary
        out = grouped.to_dict(orient='index')

        return out

