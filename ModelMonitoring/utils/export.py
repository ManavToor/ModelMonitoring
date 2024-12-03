import json
import pyodbc

class Export:
    def __init__(self, workspace: str):
        """
        methods for taking saved JSON files from evidently and uploading them to SQL databases for Power BI

        example usage:
            export = Export('workspace/model1')
            export.connect(r'C:/Absolute_path_to_database.accdb')
            export.setup_database()
            export.drift_report('drift.json')
            export.target_drift_report('target.json', 'cnt', 'prediction')
            export.disconnect()

        :param workspace: path to folder with JSON files
        """

        self.workspace = workspace

        self.database = {
            'data_drift_report': {                      # general information regarding data drift
                'drift_share': 'FLOAT',                 # dataset drift detection threshold (percent)
                'number_of_columns': 'INT',
                'number_of_drifted_columns': 'INT',
                'share_of_drifted_columns': 'FLOAT',    # percentage of columns that are drifted
                'dataset_drift': 'FLOAT'                # is drift detected (0=No, -1=Yes)
            },
            'data_drift_columns_report': {              # dataset drift information regarding each individual column
                'column_name': 'VARCHAR',
                'stattest_name': 'VARCHAR',             # type of test used to detect drift (ie K-S p_value)
                'drift_score': 'FLOAT',
                'drift_detected': 'VARCHAR'             # is drift detected (0=No, -1=Yes)
            },
            'dataset_summary': {                        # general information about dataset
                'ref_cur': 'CHAR(3)',                   # reference data (ref) or current data (cur)
                'target': 'VARCHAR',                    # name of target column
                'prediction': 'VARCHAR',                # name of prediction column
                'date_column': 'VARCHAR',               # name of date column
                'id_column' : 'VARCHAR',                # name of id column
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
            'column_summary': {                         #  general information about each column
                'column_name': 'VARCHAR',
                'cur_ref': 'CHAR(3)',                   # reference data (ref) or current data (cur)
                'column_type': 'VARCHAR',               # data type of column (ie numeric, categorical, datetime)
                'count': 'INT',                         # number of datapoints
                'missing': 'INT',                       # number of missing datapoints
                'missing_percentage': 'FLOAT',
                'mean': 'FLOAT',                        # mean value
                'std': 'FLOAT',                         # standard deviation
                'min': 'FLOAT',                         # smallest value
                'p25': 'FLOAT',                         # 25% smallest value
                'p50': 'FLOAT',                         # median value
                'p75': 'FLOAT',                         # 75% smallest value
                'max': 'FLOAT',                         # largest value
                'unique_values': 'INT',
                'unique_percentage': 'FLOAT',
                'most_common': 'FLOAT',
                'most_common_percentage': 'FLOAT',
                'new_in_current_values_count': 'INT',   # categorical only, number of new values in current data
                'unused_in_current_values_count': 'INT' # categorical only, number of unused values in current data
            },
            'target_drift_correlation': {               # information regarding target drift correlation
                'cur_ref': 'CHAR(3)',                   # reference data (ref) or current data (cur)
                'correlation_type': 'VARCHAR',          # correlation type (pearson, spearman, kendall)
                'tgt_prd': 'CHAR(3)',                   # target (tgt) or prediction (prd)
                'column_name': 'VARCHAR',
                'col_value': 'FLOAT'
            },
            'regression_metrics': {                     # general information regarding regression model
                'cur_ref': 'CHAR(3)',                   # reference data (ref) or current data (cur)
                'mean_error': 'FLOAT',
                'error_std': 'FLOAT',
                'mean_abs_error': 'FLOAT',
                'abs_error_std': 'FLOAT',
                'mean_abs_perc_error': 'FLOAT',         # mean absolute percentage error
                'abs_perc_error_std': 'FLOAT',          # 1 standard deviation of absolute percentage error
                'majority_mean_error': 'FLOAT',
                'majority_std_error': 'FLOAT',
                'underestimation_mean_error': 'FLOAT',
                'underestimation_std_error': 'FLOAT',
                'overestimation_mean_error': 'FLOAT',
                'overestimation_std_error': 'FLOAT'
            },
            'error_normality_line': {                   # identity line for Q-Q plot
                'slope': 'FLOAT',
                'intercept': 'FLOAT',
                'r': 'FLOAT'
            },
            'error_normality_data': {                   # values for Q-Q plot
                'x': 'FLOAT',                           # theoretical quantities
                'y': 'FLOAT'                            # dataset quantities
            },
            'error_bias': {                             # general error bias data for regression model
                'column_name': 'VARCHAR',
                'feature_type': 'VARCHAR',              # data type of column (ie numeric, categorical, datetime)
                'current_majority': 'FLOAT',            # 90% of predictions in current data
                'current_under': 'FLOAT',               # top-5% of the predictions with underestimation in current data
                'current_over': 'FLOAT',                # top-5% of predictions with overestimation in current data
                'current_range': 'FLOAT',               # current range percentage
                'ref_majority': 'FLOAT',                # 90% of predictions in reference data
                'ref_under': 'FLOAT',                   # top-5% of the predictions with underestimation in reference data
                'ref_over': 'FLOAT',                    # top-5% of predictions with overestimation in reference data
                'ref_range': 'FLOAT'                    # reference range percentage
            },
            'classification_metrics': {                 # general information regarding classification model
                'cur_ref': 'CHAR(3)',                   # reference data (ref) or current data (cur)
                'accuracy': 'FLOAT',                    # correct classification / total classification
                'precision': 'FLOAT',                   # true positives / (true positives + false positives)
                'recall': 'FLOAT',                      # true positives / (true positives + false negatives)
                'f1': 'FLOAT'                           # f1 score
            },
            'confusion_matrix': {                       # confusion matrix for classification model
                'cur_ref': 'CHAR(3)',                   # reference data (ref) or current data (cur)
                'label0': 'VARCHAR',                    # negative label
                'label1': 'VARCHAR',                    # positive label
                '00': 'INT',                            # true negatives
                '01': 'INT',                            # false positive
                '10': 'INT',                            # false negative
                '11': 'INT'                             # true positive
            }
            # column_dist:                              data distribution of each parameter
            #   ref_x FLOAT                             reference data x values
            #   ref_y FLOAT                             reference data y values
            #   cur_x FLOAT                             current data x values
            #   cur_y FLOAT                             current data y values

        }

        self.path_to_db = None
        self.conn = None
        self.cursor = None

    def connect(self, path: str) -> None:
        """
        establish connection to SQL database

        :param path: absolute path to database
        """

        self.path_to_db = path

        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            rf'DBQ={self.path_to_db};'
        )

        self.conn = pyodbc.connect(conn_str)
        self.cursor = self.conn.cursor()

    def disconnect(self) -> None:
        """
        disconnect from SQL database
        """

        self.conn.close()

    def data_drift_report(self, filename: str) -> None:
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

        :param filename: name of JSON file containing data (include .json)
        """

        with open(self.workspace + '/' + filename) as file:
            data = json.load(file)

        # data_drift_report
        report_general = data['metrics'][0]['result']
        self.write_to_table('data_drift_report', report_general['drift_share'], report_general['number_of_columns'], report_general['number_of_drifted_columns'], report_general['share_of_drifted_columns'], report_general['dataset_drift'])

        data = data['metrics'][1]['result']['drift_by_columns']

        # data_drift_columns_report, tables with names of columns
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

            # create table if it does not already exist
            try:
                self.setup_database(f'{column}_dist')
            except pyodbc.ProgrammingError:
                pass

            for i in range(len(x_cur)):
                self.cursor.execute(f'INSERT INTO {column}_dist (ref_x, ref_y, cur_x, cur_y) VALUES (?, ?, ?, ?)', x_ref[i], y_ref[i], x_cur[i], y_cur[i])
            self.conn.commit()

            self.write_to_table('data_drift_columns_report', column, col['stattest_name'], col['drift_score'], col['drift_detected'])

    def quality_report(self, filename: str) -> None:
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

        :param filename: name of JSON file containing data (include .json)
        """

        with open(self.workspace + '/' + filename) as file:
            data = json.load(file)

        data = data['metrics']

        # dataset_summary
        current = data[0]['result']['current']
        reference = data[0]['result']['reference']

        self.write_to_table('dataset_summary', 'cur', current['target'], current['prediction'], current['date_column'], current['id_column'], current['number_of_columns'], current['number_of_rows'], current['number_of_missing_values'], current['number_of_categorical_columns'], current['number_of_numeric_columns'], current['number_of_text_columns'], current['number_of_datetime_columns'], current['number_of_constant_columns'], current['number_of_almost_constant_columns'], current['number_of_duplicated_columns'], current['number_of_almost_duplicated_columns'], current['number_of_empty_rows'], current['number_of_empty_columns'], current['number_of_duplicated_rows'])
        self.write_to_table('dataset_summary', 'ref', reference['target'], reference['prediction'], reference['date_column'], reference['id_column'], reference['number_of_columns'], reference['number_of_rows'], reference['number_of_missing_values'], reference['number_of_categorical_columns'], reference['number_of_numeric_columns'], reference['number_of_text_columns'], reference['number_of_datetime_columns'], reference['number_of_constant_columns'], reference['number_of_almost_constant_columns'], reference['number_of_duplicated_columns'], reference['number_of_almost_duplicated_columns'],reference['number_of_empty_rows'], reference['number_of_empty_columns'], reference['number_of_duplicated_rows'])

        # column_summary
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
                self.write_to_table('column_summary', column['result']['column_name'], 'ref', column['result']['column_type'], current['count'], current['missing'], current['missing_percentage'], None, None, None, None, None, None, None, current['unique'], current['unique_percentage'], current['most_common'], current['most_common_percentage'], current['new_in_current_values_count'], current['unused_in_current_values_count'])

    def target_drift_report(self, filename: str, target: str, prediction: str) -> None:
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

        :param filename: name of JSON file containing data (include .json)
        :param target: name of target column
        :param prediction: name of prediction column
        """

        with open(self.workspace + '/' + filename) as file:
            data = json.load(file)

        # list to hold names of all columns
        columns = []

        # target_drift_correlation
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

    def regression_model_report(self, filename: str) -> None:
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

        :param filename: name of JSON file containing data (include .json)
        :return:
        """

        with open(self.workspace + '/' + filename) as file:
            data = json.load(file)

        data = data['metrics'][0]

        # regression_metrics
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
        self.write_to_table('error_normality_line', error_normality['slope'], error_normality['intercept'], error_normality['r'])

        x = error_normality['order_statistic_medians_x']
        y = error_normality['order_statistic_medians_y']
        for i in range(len(x)):
            self.write_to_table('error_normality_data', x[i], y[i])

        # error_bias
        # I don't fully know what this table is for
        error_bias = data['result']['error_bias']
        for column in error_bias:
            self.write_to_table('error_bias', column, error_bias[column]['feature_type'], error_bias[column]['current_majority'], error_bias[column]['current_under'], error_bias[column]['current_over'], error_bias[column]['current_range'], error_bias[column]['ref_majority'], error_bias[column]['ref_under'], error_bias[column]['ref_over'], error_bias[column]['ref_range'])

    def classification_model_report(self, filename: str) -> None:
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

       :param filename: name of JSON file containing data (include .json)
       """

        with open(self.workspace + '/' + filename) as file:
            data = json.load(file)

        data = data['metrics']

        # classification_metrics
        for i in range(2):
            cur_ref = 'current' if i == 0 else 'reference'

            accuracy = data[0]['result'][cur_ref]['accuracy']
            precision = data[0]['result'][cur_ref]['precision']
            recall = data[0]['result'][cur_ref]['recall']
            f1 = data[0]['result'][cur_ref]['f1']

            self.write_to_table('classification_metrics', cur_ref, accuracy, precision, recall, f1)

        # confusion_matrix
        cur_matrix = data[2]['result']['current_matrix']
        ref_matrix = data[2]['result']['reference_matrix']
        self.write_to_table('confusion_matrix', 'cur', cur_matrix['labels'][0], cur_matrix['labels'][1], cur_matrix['values'][0][0], cur_matrix['values'][0][1], cur_matrix['values'][1][0], cur_matrix['values'][1][1])
        self.write_to_table('confusion_matrix', 'ref', ref_matrix['labels'][0], ref_matrix['labels'][1], ref_matrix['values'][0][0], ref_matrix['values'][0][1], ref_matrix['values'][1][0], ref_matrix['values'][1][1])

    def setup_database(self, table_with_column_name=None) -> None:
        """
        .accdb file must be created manually using MS Access. Once an empty file has been created, this method will
        populate it with the appropriate tables

        :param table_with_column_name: for creating a table with the name of a column, this table holds data distribution, value is string with name of the column
        """

        if table_with_column_name is not None:
            self.cursor.execute(f"CREATE TABLE {table_with_column_name} (ID AUTOINCREMENT PRIMARY KEY, ref_x FLOAT, ref_y FLOAT, cur_x FLOAT, cur_y FLOAT)")
            self.conn.commit()

            return

        for db in self.database:
            query = f"CREATE TABLE {db} (ID AUTOINCREMENT PRIMARY KEY"

            for col in self.database[db]:
                query += f", {col} {self.database[db][col]}"

            query += ")"

            self.cursor.execute(query)
            self.conn.commit()

    def write_to_table(self, table: str, *values) -> None:
        column_names = ', '.join(self.database[table].keys())
        placeholders = ', '.join(['?'] * len(self.database[table]))

        query = f'INSERT INTO {table} ({column_names}) VALUES ({placeholders})'

        self.cursor.execute(query, values)
        self.conn.commit()

