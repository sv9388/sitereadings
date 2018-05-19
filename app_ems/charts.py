import pandas as pd
from pandas.compat import StringIO
from datetime import datetime, timedelta, date
from models import *
from enums import *
from date_utils import *
from sqlalchemy import func, extract
from app import db

epoch = datetime.utcfromtimestamp(0)

CHART_DIV_ID = "outputchart"
OUTPUT_DATE_FORMAT = "%Y-%m-%d %H:%M"
agg_mapping = {LAST_7_DAYS :  "Last 7 days", LAST_30_DAYS : "Last 30 days", CURRENT_MONTH : "Current Month" , LAST_12_MONTHS : "Last 12 Months", CURRENT_YEAR : "Current Year", ALL_DATA : "All Data"}
 
def serialize(df, df2, qtype, metric, sd_str, ed_str, sd_str2, ed_str2, kind): 
  # msites = Time vs Metric (multiple sites per time graph). mtimerange = Site vs Metric (2 tr per range graph)
  chart_json = {"chart" : {"zoomType": 'x', "type" : "", "renderTo" : CHART_DIV_ID}, "title" : {"text" : ""}, "yAxis" : {"title" : {"text" : ""}}, "series" : [], "responsive" : { "rules" : [{"condition" : {"maxWidth" : 500}, "chartOptions": {"legend" : { "layout" : "horizontal", "align" : "center", "verticalAlign" : "bottom"}}}]}}
  chart_json["yAxis"]["title"]["text"] = metric
  chart_json["chart"]["type"] = kind

  if qtype == QUERY_MSITES:
    chart_json['title']['text'] = "{} between {} and {}".format(metric, sd_str, ed_str)
    chart_json["xAxis"] = { "type" : "datetime" }
    chart_json["series"] = [{"name" : x, "data" : df[['time', x]].values.tolist()} for x in df.columns if x != "time"]
  else:
    chart_json['title']['text'] = "{} comparison for timeranges {} - {} and {} - {}".format(metric, sd_str, ed_str, sd_str2, ed_str2)
    categories =  [x for x in df.columns if x != "time"]
    chart_json["xAxis"] = { "categories" : categories }
    if df2 is None or df is None:
      return chart_json, "No valid data found for other the chosen timerange"
    opj = []
    opj.append({"name" : "{} - {}".format(sd_str, ed_str), "data" : [[x, df[x].sum().item()] for x in categories]})
    opj.append({"name" : "{} - {}".format(sd_str2, ed_str2), "data" : [[x, df2[x].sum().item()] for x in categories]})
    chart_json["series"] = opj #me" : x, "data" : df[['time', x]].values.tolist()} for x in df.columns if x != "time"]

  return chart_json, ""

def _get_readings(sites_arr, start_date, end_date, agg_period, metric):
  print(start_date, end_date)
  if not start_date or not end_date:
    return []
  rq = db.session.query(Reading.device_id, (func.floor(extract('epoch', Reading.rdate)/agg_period)).label('grouped'), (func.max(Reading.total_kwh)-func.min(Reading.total_kwh)).label("value"))
  rq = rq.filter(Reading.device_id.in_(sites_arr))
  rq = rq.filter(Reading.rdate >= start_date).filter(Reading.rdate <= end_date)
  rq = rq.group_by(Reading.device_id, (func.floor(extract('epoch', Reading.rdate)/agg_period)).label('grouped'))
  rq = rq.order_by(Reading.device_id, (func.floor(extract('epoch', Reading.rdate)/agg_period)).label('grouped'))
  print(rq)
  readings = rq.all()
  return readings

def _get_df(op_arr, devices, agg_period, metric):
  df = pd.DataFrame(op_arr, columns = ["device_id", "grouped", "value"])
  df = df.pivot(index="grouped", columns="device_id", values="value")
  if metric == "kwh_psqm":
    for x in df.columns:
      df[x] /= devices[x][1]
  df.columns = [devices[x][0] for x in df.columns]
  df['time'] = df.index * agg_period * 1000 if not df.empty else 0
  return df

def get_chart_data(sites_arr, ctype, start_date, end_date, agg_period, start_date2 = None, end_date2 = None, metric = "total_kwh", qtype = QUERY_MSITES ):
  readings = _get_readings(sites_arr, start_date, end_date, agg_period, metric)
  readings2 = _get_readings(sites_arr, start_date2, end_date2, agg_period, metric)

  devices = Device.query.filter(Device.id.in_(sites_arr)).all()
  devices = { x.id : ["{} | {} | {} ".format(x.device_id, x.distributer_name, x.project), x.sqm] for x in devices}

  df = _get_df(readings, devices, agg_period, metric) 
  df2 = None
  if qtype == QUERY_MTIMERANGES:
    df2 = _get_df(readings2, devices, agg_period, metric)

  sd_str, ed_str = start_date.strftime(OUTPUT_DATE_FORMAT), end_date.strftime(OUTPUT_DATE_FORMAT)
  sd_str2, ed_str2 = None, None
  if qtype == QUERY_MTIMERANGES:
    sd_str2, ed_str2 = start_date2.strftime(OUTPUT_DATE_FORMAT), end_date2.strftime(OUTPUT_DATE_FORMAT)
  chart, errors = serialize(df, df2, qtype, metric, sd_str, ed_str, sd_str2, ed_str2, kind = ctype)
  return chart, errors
