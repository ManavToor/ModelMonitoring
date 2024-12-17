import json
import pyodbc
import math

from . import MODEL_EVIDENTLY_ODBC


class Export:
    def __init__(self, model_name: str, monthly: bool = False):
        """
        methods for taking saved JSON files from evidently and uploading them to SQL databases for Power BI

        example usage:
            export = Export(model_name='my_model')
            export.connect()
            export.setup_database()
            export.drift_report(drift_data)
            export.target_drift_report(target_data, 'target', 'prediction')
            export.disconnect()

        :param monthly: Is this monthly data (T) or all time data (F)
        """

        self.model_name = model_name
        self.schema = 'evi_metrics' # schema for SQL tables
        # all tables are prefixed, "m_" for monthly data, "t_" for all-time data
        self.prefix = 'm_' if monthly else 't_'

        self.database = {
            'data_drift_report': {                          # general information regarding data drift
                'drift_share': 'NUMERIC(19,4)',             # dataset drift detection threshold (percent)
                'number_of_columns': 'INT',
                'number_of_drifted_columns': 'INT',
                'share_of_drifted_columns': 'FLOAT',        # percentage of columns that are drifted
                'dataset_drift': 'NUMERIC(19,4)'            # is drift detected (0=No, -1=Yes)
            },
            'data_drift_columns_report': {                  # dataset drift information regarding each individual column
                'column_name': 'TEXT',
                'stattest_name': 'VARCHAR(255)',            # type of test used to detect drift (ie K-S p_value)
                'drift_score': 'NUMERIC(19,4)',
                'drift_detected': 'VARCHAR(255)'            # is drift detected (0=No, -1=Yes)
            },
            'dataset_summary': {                            # general information about dataset
                'ref_cur': 'CHAR(3)',                       # reference data (ref) or current data (cur)
                'target': 'VARCHAR(255)',                   # name of target column
                'prediction': 'VARCHAR(255)',               # name of prediction column
                'date_column': 'VARCHAR(255)',              # name of date column
                'id_column' : 'VARCHAR(255)',               # name of id column
                'number_of_columns': 'INT',
                'number_of_rows': 'INT',
                'number_of_missing_values': 'INT',
                'number_of_categorical_columns': 'INT',
                'number_of_numeric_columns': 'INT',
                'number_of_text_columns': 'INT',
                'number_of_datetime_columns': 'INT',
                'number_of_constant_columns': 'INT',
                'number_of_almost_constant_columns': 'INT',
                'number_of_duplicated_columns': 'INT',
                'number_of_almost_duplicated_columns': 'INT',
                'number_of_empty_rows': 'INT',
                'number_of_empty_columns': 'INT',
                'number_of_duplicated_rows': 'INT'
            },
            'column_summary': {                             #  general information about each column
                'column_name': 'VARCHAR(255)',
                'cur_ref': 'CHAR(3)',                       # reference data (ref) or current data (cur)
                'column_type': 'VARCHAR(255)',              # data type of column (ie numeric, categorical, datetime)
                'count': 'INT',                             # number of datapoints
                'missing': 'INT',                           # number of missing datapoints
                'missing_percentage': 'NUMERIC(19,4)',
                'mean': 'NUMERIC(19,4)',                    # mean value
                'std': 'NUMERIC(19,4)',                     # standard deviation
                'min': 'NUMERIC(19,4)',                     # smallest value
                'p25': 'NUMERIC(19,4)',                     # 25% smallest value
                'p50': 'NUMERIC(19,4)',                     # median value
                'p75': 'NUMERIC(19,4)',                     # 75% smallest value
                'max': 'NUMERIC(19,4)',                     # largest value
                'unique_values': 'INT',
                'unique_percentage': 'NUMERIC(19,4)',
                'most_common': 'NUMERIC(19,4)',
                'most_common_percentage': 'NUMERIC(19,4)',
                'new_in_current_values_count': 'INT',       # categorical only, number of new values in current data
                'unused_in_current_values_count': 'INT'     # categorical only, number of unused values in current data
            },
            #'target_drift_correlation': {                   # information regarding target drift correlation
            #    'cur_ref': 'CHAR(3)',                       # reference data (ref) or current data (cur)
            #    'correlation_type': 'VARCHAR(255)',         # correlation type (pearson, spearman, kendall)
            #    'tgt_prd': 'CHAR(3)',                       # target (tgt) or prediction (prd)
            #    'column_name': 'VARCHAR(255)',
            #    'col_value': 'NUMERIC(19,4)'                # correlation value
            #},
            #'regression_metrics': {                         # general information regarding regression model
            #    'cur_ref': 'CHAR(3)',                       # reference data (ref) or current data (cur)
            #    'mean_error': 'NUMERIC(19,4)',
            #    'error_std': 'NUMERIC(19,4)',
            #    'mean_abs_error': 'NUMERIC(19,4)',
            #    'abs_error_std': 'NUMERIC(19,4)',
            #    'mean_abs_perc_error': 'NUMERIC(19,4)',     # mean absolute percentage error
            #    'abs_perc_error_std': 'NUMERIC(19,4)',      # 1 standard deviation of absolute percentage error
            #    'majority_mean_error': 'NUMERIC(19,4)',
            #    'majority_std_error': 'NUMERIC(19,4)',
            #    'underestimation_mean_error': 'NUMERIC(19,4)',
            #    'underestimation_std_error': 'NUMERIC(19,4)',
            #    'overestimation_mean_error': 'NUMERIC(19,4)',
            #    'overestimation_std_error': 'NUMERIC(19,4)'
            #},
            #'error_normality_line': {                       # identity line for Q-Q plot
            #    'slope': 'NUMERIC(19,4)',
            #    'intercept': 'NUMERIC(19,4)',
            #    'r': 'NUMERIC(19,4)'
            #},
            #'error_normality_data': {                       # values for Q-Q plot
            #    'x': 'NUMERIC(19,4)',                       # theoretical quantities
            #    'y': 'NUMERIC(19,4)'                        # dataset quantities
            #},
            #'error_bias': {                                 # general error bias data for regression model
            #    'column_name': 'VARCHAR(255)',
            #    'feature_type': 'VARCHAR(255)',             # data type of column (ie numeric, categorical, datetime)
            #    'current_majority': 'NUMERIC(19,4)',        # 90% of predictions in current data
            #    'current_under': 'NUMERIC(19,4)',           # top-5% of the predictions with underestimation in current data
            #    'current_over': 'NUMERIC(19,4)',            # top-5% of predictions with overestimation in current data
            #    'current_range': 'NUMERIC(19,4)',           # current range percentage
            #    'ref_majority': 'NUMERIC(19,4)',            # 90% of predictions in reference data
            #    'ref_under': 'NUMERIC(19,4)',               # top-5% of the predictions with underestimation in reference data
            #    'ref_over': 'NUMERIC(19,4)',                # top-5% of predictions with overestimation in reference data
            #    'ref_range': 'NUMERIC(19,4)'                # reference range percentage
            #},
            'classification_metrics': {                     # general information regarding classification model
                'cur_ref': 'CHAR(3)',                       # reference data (ref) or current data (cur)
                'accuracy': 'NUMERIC(19,4)',                # correct classification / total classification
                'precision': 'NUMERIC(19,4)',               # true positives / (true positives + false positives)
                'recall': 'NUMERIC(19,4)',                  # true positives / (true positives + false negatives)
                'f1': 'NUMERIC(19,4)'                       # f1 score
            },
            'confusion_matrix': {                       # confusion matrix for classification model (both models are binary)
                'cur_ref': 'CHAR(3)',                   # reference data (ref) or current data (cur)
                'label0': 'VARCHAR(255)',                   # negative label
                'label1': 'VARCHAR(255)',                   # positive label
                '_00': 'INT',                               # true negatives
                '_01': 'INT',                               # false positive
                '_10': 'INT',                               # false negative
                '_11': 'INT'                                # true positive
            },
            'column_dist': {                                # data distribution of each column
                'column_name': 'VARCHAR(255)',
                'ref_x': 'TEXT',                            # reference data x values
                'ref_y': 'TEXT',                            # reference data y values
                'cur_x': 'TEXT',                            # current data x values
                'cur_y': 'TEXT'                             # current data y values
            },
            'target_by_feature': {                          # distribution of target by feature (both models are binary)
                'feature': 'VARCHAR(255)',                  # column/feature name
                'cur_ref': 'CHAR(3)',
                'x_0': 'TEXT',                              # list of feature values when model outputs 0
                'y_0': 'TEXT',                              # occurrence of each feature value when model outputs 0
                'x_1': 'TEXT',                              # list of feature values when model outputs 1
                'y_1': 'TEXT'                               # occurrence of each feature value when model outputs 1
            }
        }

        self.path_to_db = None
        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """
        establish connection to SQL database
        """

        self.conn = pyodbc.connect(MODEL_EVIDENTLY_ODBC)
        self.cursor = self.conn.cursor()

    def disconnect(self) -> None:
        """
        disconnect from SQL database
        """

        self.conn.close()

    def data_drift_report(self, data: str) -> None:
        """
        Data drift report structure (ignoring some keys):

        metrics
            |- data_drift_metrics **
            |- data_drift_metrics_&_results **
                |- data_drift_metrics *
                |- drift_by_columns
                    |- columns *
                        |- column_metrics *
                        |- current
                        |   |- small_distribution
                        |       |- x
                        |       |- y
                        |- reference
                            |- small_distribution
                                |- x
                                |- y

        * These are not the actual key names, and may categorize multiple keys
        ** These are list items

        :param data: JSON string object containing data
        """

        data = json.loads(data)

        # data_drift_report
        report_general = data['metrics'][0]['result']
        self.clear_table('data_drift_report')
        self.write_to_table('data_drift_report', report_general['drift_share'], report_general['number_of_columns'], report_general['number_of_drifted_columns'], report_general['share_of_drifted_columns'], report_general['dataset_drift'])

        data = data['metrics'][1]['result']['drift_by_columns']

        # data_drift_columns_report, tables with names of columns
        self.clear_table('column_dist')
        self.clear_table('data_drift_columns_report')
        for column in data:
            col = data[column]

            x_cur = col['current']['small_distribution']['x']
            y_cur = col['current']['small_distribution']['y']

            x_ref = col['reference']['small_distribution']['x']
            y_ref = col['reference']['small_distribution']['y']

            # sometimes x has an extra item at the end
            if len(x_cur) != len(y_cur):
                x_cur = x_cur[:-1]
            if len(x_ref) != len(y_ref):
                x_ref = x_ref[:-1]
            
            self.write_to_table('column_dist', column, json.dumps(x_ref), json.dumps(y_ref), json.dumps(x_cur), json.dumps(y_cur))

            self.write_to_table('data_drift_columns_report', column, col['stattest_name'], col['drift_score'], col['drift_detected'])

    def quality_report(self, data: str) -> None:
        """
        Quality report structure (ignoring some keys):

        metrics
            |- DatasetSummaryMetric **
            |   |- result
            |       |- current
            |       |   |- data_set_summary *
            |       |   |- nans_by_column <- missing values
            |       |- reference
            |           |- data_set_summary *
            |           |- nans_by_column <- missing values
            |- ColumnSummaryMetric * **
                |- result
                    |- reference_characteristics
                    |- current_characteristics

        * These are not the actual key names, and may categorize multiple keys
        ** These are list items

        :param data: JSON string object containing data
        """

        data = json.loads(data)

        data = data['metrics']

        # dataset_summary
        current = data[0]['result']['current']
        reference = data[0]['result']['reference']

        self.clear_table('dataset_summary')
        self.write_to_table('dataset_summary', 'cur', current['target'], current['prediction'], current['date_column'], current['id_column'], current['number_of_columns'], current['number_of_rows'], current['number_of_missing_values'], current['number_of_categorical_columns'], current['number_of_numeric_columns'], current['number_of_text_columns'], current['number_of_datetime_columns'], current['number_of_constant_columns'], current['number_of_almost_constant_columns'], current['number_of_duplicated_columns'], current['number_of_almost_duplicated_columns'], current['number_of_empty_rows'], current['number_of_empty_columns'], current['number_of_duplicated_rows'])
        self.write_to_table('dataset_summary', 'ref', reference['target'], reference['prediction'], reference['date_column'], reference['id_column'], reference['number_of_columns'], reference['number_of_rows'], reference['number_of_missing_values'], reference['number_of_categorical_columns'], reference['number_of_numeric_columns'], reference['number_of_text_columns'], reference['number_of_datetime_columns'], reference['number_of_constant_columns'], reference['number_of_almost_constant_columns'], reference['number_of_duplicated_columns'], reference['number_of_almost_duplicated_columns'],reference['number_of_empty_rows'], reference['number_of_empty_columns'], reference['number_of_duplicated_rows'])

        # column_summary
        self.clear_table('column_summary')
        for column in data:
            if column['metric'] != 'ColumnSummaryMetric':
                continue

            reference = column['result']['reference_characteristics']
            current = column['result']['current_characteristics']
            if column['result']['column_type'] == 'num':
                self.write_to_table('column_summary', column['result']['column_name'], 'ref', column['result']['column_type'], reference['count'], reference['missing'], reference['missing_percentage'], reference['mean'], reference['std'], reference['min'], reference['p25'], reference['p50'], reference['p75'], reference['max'], reference['unique'], reference['unique_percentage'], reference['most_common'], reference['most_common_percentage'], None, None)
                self.write_to_table('column_summary', column['result']['column_name'], 'cur', column['result']['column_type'], current['count'], current['missing'], current['missing_percentage'], current['mean'], current['std'], current['min'], current['p25'], current['p50'], current['p75'], current['max'], current['unique'], current['unique_percentage'], current['most_common'], current['most_common_percentage'], None, None)
            elif column['result']['column_type'] == 'cat':
                self.write_to_table('column_summary', column['result']['column_name'], 'ref', column['result']['column_type'], reference['count'], reference['missing'], reference['missing_percentage'], None, None, None, None, None, None, None, reference['unique'], reference['unique_percentage'], reference['most_common'], reference['most_common_percentage'], reference['new_in_current_values_count'], reference['unused_in_current_values_count'])
                self.write_to_table('column_summary', column['result']['column_name'], 'cur', column['result']['column_type'], current['count'], current['missing'], current['missing_percentage'], None, None, None, None, None, None, None, current['unique'], current['unique_percentage'], current['most_common'], current['most_common_percentage'], current['new_in_current_values_count'], current['unused_in_current_values_count'])

    def target_drift_report(self, data: str, target: str, prediction: str) -> None:
        """
        Target drift report structure (ignoring some keys):

        metrics
            |- ColumnDriftMetric ** <- (target) ignore already in drift report
            |- ColumnValuePlot ** <- empty
            |- ColumnCorrelationsMetric ** (target)
            |   |- result
            |       |- current
            |       |   |- pearson/spearman/kendall *
            |       |       |- x
            |       |       |- y
            |       |- reference
            |           |- pearson/spearman/kendall *
            |               |- x
            |               |- y
            |- ColumnDriftMetric ** <- (prediction) ignore already in drift report
            |- ColumnCorrelationsMetric ** (prediction)
            |   |- result
            |       |- current
            |       |   |- pearson/spearman/kendall *
            |       |       |- x
            |       |       |- y
            |       |- reference
            |           |- pearson/spearman/kendall *
            |               |- x
            |               |- y
            |- TargetByFeaturesTable ** <- empty

        * These are not the actual key names, and may categorize multiple keys
        ** These are list items

        :param data: JSON string object containing data
        :param target: name of target column
        :param prediction: name of prediction column
        """

        data = json.loads(data)

        # list to hold names of all columns
        columns = []

        # target_drift_correlation
        self.clear_table('target_drift_correlation')
        for metric in data['metrics']:
            if metric['metric'] != 'ColumnCorrelationsMetric':
                continue

            metric = metric['result']

            # get all column names if list is empty
            if len(columns) == 0:
                for col in metric['current']['pearson']['values']['x']:
                    columns.append(col)

            tgt_prd = ''
            if metric['column_name'] == target:
                tgt_prd = 'tgt'
            elif metric['column_name'] == prediction:
                tgt_prd = 'prd'

            for col in range(len(columns)):
                self.write_to_table('target_drift_correlation', 'current', 'pearson', tgt_prd, columns[col], metric['current']['pearson']['values']['y'][col])
                self.write_to_table('target_drift_correlation', 'current', 'spearman', tgt_prd, columns[col], metric['current']['spearman']['values']['y'][col])
                self.write_to_table('target_drift_correlation', 'current', 'kendall', tgt_prd, columns[col], metric['current']['kendall']['values']['y'][col])
                self.write_to_table('target_drift_correlation', 'reference', 'pearson', tgt_prd, columns[col], metric['reference']['pearson']['values']['y'][col])
                self.write_to_table('target_drift_correlation', 'reference', 'spearman', tgt_prd, columns[col], metric['reference']['spearman']['values']['y'][col])
                self.write_to_table('target_drift_correlation', 'reference', 'kendall', tgt_prd, columns[col], metric['reference']['kendall']['values']['y'][col])

    def regression_model_report(self, data: str) -> None:
        """
        Drift report structure (ignoring some keys):

        metrics
            |- RegressionQualityMetric **
            |   |- result
            |       |- current
            |       |- reference
            |       |- error_normality
            |       |   |- order_statistic_medians_x/y *
            |       |   |- theoretical_quantities_info *
            |       |- error_bias
            |       |- other_analytics *
            |- A_bunch_of_stuff_that's_empty * **

        * These are not the actual key names, and may categorize multiple keys
        ** These are list items

        :param data: JSON string object containing data
        """

        data = json.loads(data)

        data = data['metrics'][0]

        # regression_metrics
        self.clear_table('regression_metrics')
        for i in range(2):
            cur_ref = 'current' if i == 0 else 'reference'

            mean_error = data['result'][cur_ref]['mean_error']
            error_std = data['result'][cur_ref]['error_std']
            mean_abs_error = data['result'][cur_ref]['mean_abs_error']
            abs_error_std = data['result'][cur_ref]['abs_error_std']
            mean_abs_perc_error = data['result'][cur_ref]['mean_abs_perc_error']
            abs_perc_error_std = data['result'][cur_ref]['abs_perc_error_std']
            majority_mean_error = data['result'][cur_ref]['underperformance']['majority']['mean_error']
            majority_std_error = data['result'][cur_ref]['underperformance']['majority']['std_error']
            underestimation_mean_error = data['result'][cur_ref]['underperformance']['underestimation']['mean_error']
            underestimation_std_error = data['result'][cur_ref]['underperformance']['underestimation']['std_error']
            overestimation_mean_error = data['result'][cur_ref]['underperformance']['overestimation']['mean_error']
            overestimation_std_error = data['result'][cur_ref]['underperformance']['overestimation']['std_error']

            self.write_to_table('regression_metrics', cur_ref, mean_error, error_std, mean_abs_error, abs_error_std, mean_abs_perc_error, abs_perc_error_std, majority_mean_error, majority_std_error, underestimation_mean_error, underestimation_std_error, overestimation_mean_error, overestimation_std_error)

        # error_normality_line, error_normality_data
        error_normality = data['result']['error_normality']

        # I have no idea what r is for
        self.clear_table('error_normality_line')
        self.write_to_table('error_normality_line', error_normality['slope'], error_normality['intercept'], error_normality['r'])

        x = error_normality['order_statistic_medians_x']
        y = error_normality['order_statistic_medians_y']
        for i in range(len(x)):
            self.write_to_table('error_normality_data', x[i], y[i])

        # error_bias
        self.clear_table('error_bias')
        error_bias = data['result']['error_bias']
        for column in error_bias:
            self.write_to_table('error_bias', column, error_bias[column]['feature_type'], error_bias[column]['current_majority'], error_bias[column]['current_under'], error_bias[column]['current_over'], error_bias[column]['current_range'], error_bias[column]['ref_majority'], error_bias[column]['ref_under'], error_bias[column]['ref_over'], error_bias[column]['ref_range'])

    def classification_model_report(self, data: str) -> None:
        """
        Data drift report structure (ignoring some keys):

        metrics
            |- ClassificationQualityMetric **
            |   |- result
            |       |- current
            |       |- reference
            |- ClassificationClassBalance ** <- empty
            |- ClassificationConfusionMatrix **
            |   |- result
            |       |- current_matrix
            |       |- reference_matrix
            |- ClassificationQualityByClass **
            |- ClassificationQualityByFeatureTable ** <- empty

       * These are not the actual key names, and may categorize multiple keys
       ** These are list items

       :param data: JSON string object containing data
       """

        data = json.loads(data)

        data = data['metrics']

        # classification_metrics
        self.clear_table('classification_metrics')
        for i in range(2):
            cur_ref = 'current' if i == 0 else 'reference'
            sql_cur_ref = 'cur' if i == 0 else 'ref'
            accuracy = data[0]['result'][cur_ref]['accuracy']
            precision = data[0]['result'][cur_ref]['precision']
            recall = data[0]['result'][cur_ref]['recall']
            f1 = data[0]['result'][cur_ref]['f1']

            self.write_to_table('classification_metrics', sql_cur_ref, accuracy, precision, recall, f1)

        # confusion_matrix
        self.clear_table('confusion_matrix')
        cur_matrix = data[2]['result']['current_matrix']
        ref_matrix = data[2]['result']['reference_matrix']
        self.write_to_table('confusion_matrix', 'cur', cur_matrix['labels'][0], cur_matrix['labels'][1], cur_matrix['values'][0][0], cur_matrix['values'][0][1], cur_matrix['values'][1][0], cur_matrix['values'][1][1])
        self.write_to_table('confusion_matrix', 'ref', ref_matrix['labels'][0], ref_matrix['labels'][1], ref_matrix['values'][0][0], ref_matrix['values'][0][1], ref_matrix['values'][1][0], ref_matrix['values'][1][1])

    def setup_database(self) -> None:
        """
        There are two copies of each table made, one for all-time data (table names beginning with "t_") and one for
        monthly data (table names beginning with "m_")
        """

        for db in self.database:
            query = f"CREATE TABLE {self.schema}.{self.prefix}{self.model_name}_{db} ("

            for col in self.database[db]:
                query += f"{col} {self.database[db][col]}, "

            # remove extra ", "
            query = query[:-2]
            query += ")"

            self.cursor.execute(query)
            self.conn.commit()

    def write_to_table(self, table: str, *values) -> None:
        """
        Writes data to a SQL table

        :param table: table name
        :param values: values to write
        """
        column_names = ', '.join(self.database[table].keys())
        placeholders = ', '.join(['?'] * len(self.database[table]))

        def filter_nan(vals: tuple):
            out = []
            for v in vals:
                try:
                    if math.isnan(v):
                        v = None
                except TypeError:
                    pass
                out.append(v)
            return out

        query = f'INSERT INTO {self.schema}.{self.prefix}{self.model_name}_{table} ({column_names}) VALUES ({placeholders})'

        values = filter_nan(values)
        #print(query, values)
        self.cursor.execute(query, values)
        self.conn.commit()

    def clear_table(self, table: str) -> None:
        """
        Clears the monthly SQL table

        :param table: name of SQL table
        """

        self.cursor.execute(f'DELETE FROM {self.schema}.{self.prefix}{self.model_name}_{table};')
        self.conn.commit()

    def target_by_feature_report(self, data: dict) -> None:
        """
        This is a custom generated report without evidently, data structure is as follows:

        Database is only able to support binary models

        feature *
            |- ref
            |   |- 0
            |   |   |- feature_value *
            |   |       |- count
            |   |- 1
            |       |- feature_value *
            |           |- count
            |- cur
                |- 0
                |   |- feature_val
                |       |- count
                |- 1
                    |- feature_val
                        |- count

        * These are not the actual key names, and may categorize multiple keys

        :param data:
        """

        def to_xy_array(xy_dict: dict) -> (list, list):
            """
            converts target-value dictionary into an x array and y array

            :param xy_dict: target-value dictionary
            :return: x array, y array
            """

            x, y = list(), list()
            for x_val in xy_dict:
                x.append(x_val)
                y.append(xy_dict[x_val])
            return x, y

        self.clear_table('target_by_feature')
        for feature in data:
            ft = data[feature]

            for i in range(2):
                cur_ref = 'cur' if i == 0 else 'ref'

                try:
                    x_0, y_0 = to_xy_array(ft[cur_ref]['0'])
                except KeyError:
                    x_0, y_0 = list(), list()
                try:
                    x_1, y_1 = to_xy_array(ft[cur_ref]['1'])
                except KeyError:
                    x_1, y_1 = list(), list()

                self.write_to_table('target_by_feature', feature, cur_ref, json.dumps(x_0), json.dumps(y_0), json.dumps(x_1), json.dumps(y_1))
