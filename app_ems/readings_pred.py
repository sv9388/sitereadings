import pandas as pd, numpy as np
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from pandas.compat import StringIO
from datetime import datetime, timedelta, date
from models import *
from enums import *
from sqlalchemy import func, extract, cast, Integer
from app import db

epoch = datetime.utcfromtimestamp(0)

DF_COLUMNS = ['date', 'metric', 'reading_type']
CHART_DIV_ID = "outputchart"
OUTPUT_DATE_FORMAT = "%Y-%m-%d %H:%M"
MAX_DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7:31, 8:31, 9:30, 10:31, 11:31, 12:31}
agg_mapping = {LAST_7_DAYS :  "Last 7 days", LAST_30_DAYS : "Last 30 days", CURRENT_MONTH : "Current Month" , LAST_12_MONTHS : "Last 12 Months", CURRENT_YEAR : "Current Year", ALL_DATA : "All Data"}

def _get_chart(df, color_cut_idx, dot_cut_idx, chart_title, site_id):
  print(df.shape, color_cut_idx, dot_cut_idx)
  chart_json = {"chart":{"zoomType":"x","type":"line","renderTo":CHART_DIV_ID},"title":{"text":"Forecast"},"xAxis":{"categories":[]},"responsive":{"rules":[{"condition":{"maxWidth":500},"chartOptions":{"legend":{"layout":"horizontal","align":"center","verticalAlign":"bottom"}}}]}}

  data = df.metric.values.tolist()
  categories = df.date.values.tolist()
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
  start_date = today - timedelta(days = 365)
  readings = db.session.query(Reading.rdate, Reading.total_kwh).filter(Reading.device_id.in_(site_ids)).filter(Reading.rdate >= start_date).order_by(Reading.rdate).all()
  df = pd.DataFrame(readings, columns = DF_COLUMNS[:-1])
  df['date'] = pd.to_datetime(df['date'])
  df['date'] = df['date'].apply(lambda x : (datetime(x.year, x.month, x.day, x.hour) - epoch).total_seconds())
  df['reading_type'] = 'ACTUAL'
  df['metric'] = df['metric'].diff()
  df.iloc[0]['metric'] = df.iloc[1]['metric']

  df.to_csv("./kwh.csv", index = False)
  weather = db.session.query(Weather.wdate, Weather.temp).filter(Weather.device_id.in_(site_ids)).filter(Weather.wdate >= start_date).order_by(Weather.wdate).all()
  wdf = pd.DataFrame(weather, columns = ["date", "temperature"])
  wdf['date'] = pd.to_datetime(wdf['date'])
  wdf['date'] = wdf['date'].apply(lambda x : (datetime(x.year, x.month, x.day, x.hour) - epoch).total_seconds())
  wdf['weather_type'] = 'ACTUAL'
  wdf.to_csv("./weather.csv", index = False)
  opdf = pd.merge(df, wdf, on = 'date')
  return opdf

def _predict(site_id, df):
  x = df['date'].values
  y = df.metric.values

  slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
  #print(slope, intercept, r_value, p_value, std_err)

  start_pred_epoch = int(df['date'].max().item() + 3600) #PREDICT VALUES EVERY 1 hour    
  today = datetime.today()
  end_pred_epoch = int((datetime(today.year, today.month, MAX_DAYS[today.month]) - epoch).total_seconds()*1)
  time_steps = [dep for dep in range(start_pred_epoch, end_pred_epoch, 3600)]

  new_df = pd.DataFrame(time_steps, columns = ['date']) 
  new_df['reading_type'] = 'PREDICTION'
  new_df['metric'] = slope * new_df['date'] + intercept

  op_df = pd.concat([df, new_df]) 
  op_df = op_df.reset_index(drop = True)
  return op_df, df['date'].max().item()

def _process_df(df1, aps, sd, ed):
  print(sd, ed)
  df = df1[(df1['date']>=sd) & (df1['date']<ed)]
  pmin, pmax = df.date.min(), df.date.max()
  prange = []

  try:
    prange = np.arange(pmin, pmax, int(aps)*60)
  except:
    print("Empty DF")
    return pd.DataFrame()

  df.date = df.date.apply(lambda epoch_diff : epoch + timedelta(seconds = epoch_diff))
  df.date = pd.to_datetime(df.date)
  df = df.set_index('date')
  df = df.groupby(pd.Grouper(freq='{}Min'.format(aps))).aggregate(np.sum) #df.groupby(pd.cut(df.date, np.arange(sd, ed, agg_period))).sum()
  df['date'] = df.index.tolist() #df.index.values]
  df['date'] = df.date.apply(lambda rd : (rd - epoch).total_seconds())
  df = df.reset_index(drop = True)
  print(df.head())
  return df[['date', 'metric']] 

def _group_by_period(df, predict_for, aps):
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

  past_df = _process_df(df, aps, r1, r2)
  current_df = _process_df(df, aps, r2, r3)
  pred_df = _process_df(df, aps, r3, r4) 

  op_df = pd.concat([past_df, current_df, pred_df])
  op_df = op_df.sort_values(by=["date"])
  op_df = op_df.reset_index(drop = True)

  graph_cut1 = op_df[op_df['date']>=r2]['date'].idxmin()
  graph_cut2 = op_df[op_df['date']>=r3]['date'].idxmin()
  return op_df, graph_cut1.item(), graph_cut2.item()

def get_predictions_chart(site_id, predict_for, agg_period): #, chart_type):
  chart_title_fs = "{} minutes aggregation of {} data with predictions till end of the {}"
  aps, period, end_of = "60", "weekly" if predict_for == WEEK else "daily", "week" if predict_for == WEEK else "day"
  chart_title = chart_title_fs.format(aps, period, end_of)
  df = _get_existing_readings(site_id) #GET 2 MONTHS DATA AND PREDICT FOR THEM. GROUP BY WEEK LATER. 
  df_pred, dt_end = _predict(site_id, df)
  cdf = pd.concat([df, df_pred])
  df, color_cut, dot_cut = _group_by_period(cdf, predict_for, aps)
  chart = _get_chart(df, color_cut, dot_cut, chart_title, site_id)
  return chart
