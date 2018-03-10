import pandas as pd
from pandas_highcharts.core import serialize
from pandas.compat import StringIO
from datetime import datetime, timedelta, date
from models import *
from enums import *
from date_utils import *

CHART_DIV_ID = "outputchart"
OUTPUT_DATE_FORMAT = "%Y-%m-%d %H:%M"
agg_mapping = {LAST_7_DAYS :  "Last 7 days", LAST_30_DAYS : "Last 30 days", CURRENT_MONTH : "Current Month" , LAST_12_MONTHS : "Last 12 Months", CURRENT_YEAR : "Current Year", ALL_DATA : "All Data"}

def _get_readings(sites_arr, cperiod):
  start_date, end_date = get_date_range(cperiod)
  rq = Reading.query.filter(Reading.device_id.in_(sites_arr))
  if start_date and end_date:
    rq = rq.filter(Reading.rdate >= start_date).filter(Reading.rdate <= end_date)
  readings = rq.order_by(Reading.device_id, Reading.rdate).all()
  print(sites_arr, start_date, end_date, rq)
  return readings, start_date, end_date

def _get_op_arr(readings, sites_arr, start_date, end_date, agg_period, metric):
  op_arr = []
  ridx = 0
  time_steps = delta_steps(start_date, end_date, agg_period)
  for i in range(len(time_steps) -1 ):
    grouped = []
    for site_id in sites_arr:
      records = [r.total_kwh for r in readings if r.device_id == site_id and r.rdate >= time_steps[i] and r.rdate <= time_steps[i+1]] #TODO: Kwh/sqm, etc
      grouped.append(sum(records)/len(records) if len(records) > 0 else 0.0)
    op_arr.append(grouped)
  return op_arr, time_steps

def get_chart_data(sites_arr, ctype, cperiod, agg_period, metric = "kwh"):
  readings, start_date, end_date = _get_readings(sites_arr, cperiod)
  op_arr, time_steps = _get_op_arr(readings, sites_arr, start_date, end_date, agg_period, metric)
  devices = Device.query.filter(Device.id.in_(sites_arr)).all()
  df = pd.DataFrame(op_arr, columns = [x.device_id for x in devices])
  df['time'] = [time_steps[i].strftime(OUTPUT_DATE_FORMAT) for i in range(len(time_steps) -1 )]
  df  = df.set_index('time')
  print(df.head())
  sd_str, ed_str = start_date.strftime(OUTPUT_DATE_FORMAT), end_date.strftime(OUTPUT_DATE_FORMAT)
  agg_str = agg_mapping[cperiod]
  chart = serialize(df, render_to = CHART_DIV_ID, title = "Average KwH between {} and {} for the {}".format(sd_str, ed_str, agg_str))
  return chart
