import pandas as pd, numpy as np
from pandas.compat import StringIO
from datetime import datetime, timedelta, date
from models import *
from enums import *
from sqlalchemy import func, extract, cast, Integer
from app import db

epoch = datetime.utcfromtimestamp(0)

CHART_DIV_ID = "outputchart"
OUTPUT_DATE_FORMAT = "%Y-%m-%d %H:%M"
agg_mapping = {LAST_7_DAYS :  "Last 7 days", LAST_30_DAYS : "Last 30 days", CURRENT_MONTH : "Current Month" , LAST_12_MONTHS : "Last 12 Months", CURRENT_YEAR : "Current Year", ALL_DATA : "All Data"}


def _get_chart(df):
  chart_json = {"chart" : {"zoomType": 'x', "type" : "line", "renderTo" : CHART_DIV_ID}, "title" : {"text" : ""}, "xAxis" : { "type" : "datetime" }, "yAxis" : {"title" : {"text" : "dummy"}}, "series" : [], "responsive" : { "rules" : [{"condition" : {"maxWidth" : 500}, "chartOptions": {"legend" : { "layout" : "horizontal", "align" : "center", "verticalAlign" : "bottom"}}}]}}
  chart_json["series"] = [{"name" : str(x), "data" : df[['rdate', 'metric']].values.tolist()} for x in df.device_id.unique()]  
  return chart_json

def _get_existing_readings(site_ids): 
  today = datetime.today()
  start_date = datetime(today.year, today.month, 1)
  readings = db.session.query(Reading.device_id, Reading.rdate, Reading.total_kwh).filter(Reading.device_id.in_(site_ids)).filter(Reading.rdate >= start_date).order_by(Reading.device_id, Reading.rdate).all()
  df = pd.DataFrame(readings, columns = ['device_id', 'rdate', 'metric'])
  df.rdate = df.rdate.astype(str).apply(lambda x : (datetime.strptime(x, "%Y-%m-%d %H:%M:%S") - epoch).total_seconds())
  print(df.head())
  return df

def _predict_for_site_id(site_id, df):
  x = df.rdate.values
  y = df.metric.values
  eq = np.polyfit(x, y, 2)
  print(eq) #id, slope, intercept, r_value, p_value, std_err)
  #TODO: Fit the curve, project to end of month and send. 
  return df #pe, intercept, r_value, p_value, std_err)

def get_predictions(site_ids): #, chart_type):
  df = _get_existing_readings(site_ids)
  predictions = {}
  for x in site_ids:
    predictions[x] = _predict_for_site_id(x, df[df.device_id == x])
  chart = _get_chart(df)
  return chart
