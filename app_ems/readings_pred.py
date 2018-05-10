import pandas as pd, numpy as np
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from pandas.compat import StringIO
from datetime import datetime, timedelta, date
from models import *
from enums import *
from sqlalchemy import func, extract, cast, Integer
from app import db

epoch = datetime.utcfromtimestamp(0)

DF_COLUMNS = ['rdate', 'metric', 'reading_type']
CHART_DIV_ID = "outputchart"
OUTPUT_DATE_FORMAT = "%Y-%m-%d %H:%M"
MAX_DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7:31, 8:31, 9:30, 10:31, 11:31, 12:31}
agg_mapping = {LAST_7_DAYS :  "Last 7 days", LAST_30_DAYS : "Last 30 days", CURRENT_MONTH : "Current Month" , LAST_12_MONTHS : "Last 12 Months", CURRENT_YEAR : "Current Year", ALL_DATA : "All Data"}

def _get_chart(df, color_cut_idx, dot_cut_idx, chart_title, site_id):
  print(df.shape, color_cut_idx, dot_cut_idx)
  chart_json = {"chart":{"zoomType":"x","type":"line","renderTo":CHART_DIV_ID},"title":{"text":"Forecast"},"xAxis":{"categories":[]},"responsive":{"rules":[{"condition":{"maxWidth":500},"chartOptions":{"legend":{"layout":"horizontal","align":"center","verticalAlign":"bottom"}}}]}}

  data = df.metric.values.tolist()
  categories = df.rdate.values.tolist()
  categories = [(epoch + timedelta(seconds = x)).strftime("%A %d. %B %Y %H:%M:%S") for x in categories]
  chart_json['xAxis']['categories'] = categories
  series = {'zoneAxis' : 'x', 'name' : "Site Id: {}".format(site_id)}
  series['data'] = data
  series['zones'] = [{'value': color_cut_idx, 'color' : '#90ed7d'}, {'value': dot_cut_idx, 'dashStyle' : 'solid'}, {'value':df.shape[0]-1, 'dashStyle' : 'dot'}]#[ {'value' : color_cut_idx, 'color' : '#f7a35c'}, {'value' : dot_cut_idx, 'dashStyle' : 'dot'}]
  chart_json['series'] = [series]
  return chart_json

def _get_existing_readings(site_id): 
  site_ids = [site_id]
  today = datetime.today()
  start_date = today - timedelta(days = 90) 
  readings = db.session.query(Reading.rdate, Reading.total_kwh).filter(Reading.device_id.in_(site_ids)).filter(Reading.rdate >= start_date).order_by(Reading.rdate).all()
  df = pd.DataFrame(readings, columns = DF_COLUMNS[:-1])
  df['rdate'] = df['rdate'].astype(str).apply(lambda x : int((datetime.strptime(x, "%Y-%m-%d %H:%M:%S") - epoch).total_seconds()))
  df['reading_type'] = 'ACTUAL'
  return df

def _predict(site_id, df):
  x = df['rdate'].values
  y = df.metric.values

  slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
  print(slope, intercept, r_value, p_value, std_err)

  start_pred_epoch = df['rdate'].max() + 5 #PREDICT VALUES EVERY 5 SECONDS
  today = datetime.today()
  end_pred_epoch = int((datetime(today.year, today.month, MAX_DAYS[today.month]) - epoch).total_seconds()*1)
  time_steps = [dep for dep in range(start_pred_epoch, end_pred_epoch, 5)]

  new_df = pd.DataFrame(time_steps, columns = ['rdate']) 
  new_df['reading_type'] = 'PREDICTION'
  new_df['metric'] = slope * new_df['rdate'] + intercept

  op_df = pd.concat([df, new_df]) 
  op_df = op_df.reset_index(drop = True)
  return op_df, df['rdate'].max().item()

def _process_df(df1, agg_period, r1, r2):
  df = df1[(df1['rdate']>=r1) & (df1['rdate']<r2)]
  pmin, pmax = df.rdate.min(), df.rdate.max()
  prange = []

  try:
    prange = np.arange(pmin, pmax, agg_period)  
  except:
    print("Empty DF")
    return pd.DataFrame()

  df = df.groupby(pd.cut(df["rdate"], np.arange(df.rdate.min(), df.rdate.max(), agg_period))).sum()
  df.rdate = [x.left for x in df.index.values]
  df = df.reset_index(drop = True)
  return df[['rdate', 'metric']] 

def _group_by_period(df, predict_for, agg_period):
  today = datetime.today()
  today_start = datetime(today.year, today.month, today.day, 0, 0, 0)
  now = datetime.now()

  r1 = today_start - timedelta(days = today.weekday(), weeks = 1) if predict_for == WEEK else today_start - timedelta(days = 1)
  r2 = today_start - timedelta(days = today.weekday()) if predict_for == WEEK else today_start
  r3 = now
  r4 = today_start + timedelta(days = 7-today.weekday()) if predict_for == WEEK else today_start + timedelta(days =1)

  r1 = (r1 - epoch).total_seconds()
  r2 = (r2 - epoch).total_seconds()
  r3 = (r3 - epoch).total_seconds()
  r4 = (r4 - epoch).total_seconds()

  past_df = _process_df(df, agg_period, r1, r2)
  current_df = _process_df(df, agg_period, r2, r3)
  pred_df = _process_df(df, agg_period, r3, r4) 

  op_df = pd.concat([past_df, current_df, pred_df])
  op_df = op_df.sort_values(by=["rdate"])
  op_df = op_df.reset_index(drop = True)

  graph_cut1 = op_df[op_df['rdate']>=r2]['rdate'].idxmin()
  graph_cut2 = op_df[op_df['rdate']>=r3]['rdate'].idxmin()
  return op_df, graph_cut1.item(), graph_cut2.item()

def get_predictions_chart(site_id, predict_for, agg_period): #, chart_type):
  chart_title_fs = "{} minutes aggregation of {} data with predictions till end of the {}"
  aps, period, end_of = "15" if agg_period == FIFTEEN_MINUTES else "60", "weekly" if predict_for == WEEK else "daily", "week" if predict_for == WEEK else "day"
  chart_title = chart_title_fs.format(aps, period, end_of)
  df = _get_existing_readings(site_id) #GET 2 MONTHS DATA AND PREDICT FOR THEM. GROUP BY WEEK LATER. 
  df_pred, dt_end = _predict(site_id, df)
  cdf = pd.concat([df, df_pred])
  df, color_cut, dot_cut = _group_by_period(cdf, predict_for, agg_period)
  chart = _get_chart(df, color_cut, dot_cut, chart_title, site_id)
  return chart
