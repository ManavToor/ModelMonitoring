from ModelMonitoring.ModelMonitoring.core import juvenile_standard_life, ya_standard_life
from datetime import datetime, timedelta

if __name__ == '__main__':
    # == Last month == #

    # get last month in format YYYY-MM
    now = datetime.now()
    first_day_of_current_month = datetime(now.year, now.month, 1)
    last_day_of_last_month = first_day_of_current_month - timedelta(days=1)
    last_month = last_day_of_last_month.strftime('%Y-%m')

    juvenile_standard_life.juvenile_standard_life(last_month)
    ya_standard_life.ya_standard_life(last_month)

    # == All time == #
    juvenile_standard_life.juvenile_standard_life()
    ya_standard_life.ya_standard_life()