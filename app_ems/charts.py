import pandas as pd
from pandas.compat import StringIO
from datetime import datetime, timedelta, date
from models import *
from enums import *
from date_utils import *

epoch = datetime.utcfromtimestamp(0)

CHART_DIV_ID = "outputchart"
OUTPUT_DATE_FORMAT = "%Y-%m-%d %H:%M"
agg_mapping = {LAST_7_DAYS :  "Last 7 days", LAST_30_DAYS : "Last 30 days", CURRENT_MONTH : "Current Month" , LAST_12_MONTHS : "Last 12 Months", CURRENT_YEAR : "Current Year", ALL_DATA : "All Data"}
 
def serialize(df, df2, qtype, metric, sd_str, ed_str, sd_str2, ed_str2, kind): ##msites = Time vs Metric (multiple sites per time graph). mtimerange = Site vs Metric (2 tr per range graph)
  chart_json = {"chart" : {"type" : "", "renderTo" : CHART_DIV_ID}, "title" : {"text" : ""}, "yAxis" : {"title" : {"text" : ""}}, "legend" : {"layout" : "vertical", "align" : "right", "verticalAlign" : "middle" }, "series" : [], "responsive" : { "rules" : [{"condition" : {"maxWidth" : 500}, "chartOptions": {"legend" : { "layout" : "horizontal", "align" : "center", "verticalAlign" : "bottom"}}}]}}
  chart_json["yAxis"]["title"]["text"] = metric
  chart_json["chart"]["type"] = kind
  if qtype == QUERY_MSITES:
    chart_json['title']['text'] = "{} between {} and {}".format(metric, sd_str, ed_str)
    chart_json["xAxis"] = { "type" : "datetime" }
    print(df.columns, df.index)
    chart_json["series"] = [{"name" : x, "data" : df[['time', x]].values.tolist()} for x in df.columns if x != "time"]
    print(chart_json["series"])
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
    print(chart_json["series"])
  return chart_json, ""

def _get_readings(sites_arr, start_date, end_date, start_date2 = None, end_date2 = None):
  readings, readings2 = [], []
  if start_date and end_date:
    rq = Reading.query.filter(Reading.device_id.in_(sites_arr))
    rq = rq.filter(Reading.rdate >= start_date).filter(Reading.rdate <= end_date)
    readings = rq.order_by(Reading.device_id, Reading.rdate).all()
  if start_date2 and end_date2:
    rq = Reading.query.filter(Reading.device_id.in_(sites_arr))
    rq = rq.filter(Reading.rdate >= start_date).filter(Reading.rdate <= end_date)
    readings2 = rq.order_by(Reading.device_id, Reading.rdate).all()
  return readings, readings2, ""

def _get_op_arr(readings, readings2, sites_arr, start_date, end_date,  agg_period, metric, qtype = QUERY_MSITES, start_date2 =  None, end_date2 = None):
  print(len(readings), len(readings2), sites_arr, start_date, end_date,  agg_period, metric, start_date2, end_date2)
  op_arr, op_arr2 = [], []
  ridx = 0
  time_steps = delta_steps(start_date, end_date, agg_period)
  time_steps2 = delta_steps(start_date2, end_date2, agg_period)

  if qtype == QUERY_MTIMERANGES and len(time_steps) != len(time_steps2):
    return [], [], time_steps, time_steps2, "Time Steps should match for timerange comparison charts"

  if metric not in ["total_kwh", "kwh_psqm"]:
    return [], [], time_steps, time_steps2, "{} is not a valid metric. Choose from {}".format(metric, ["total_kwh", "kwh_psqm"].join(", "))

  for i in range(len(time_steps) -1 ):
    grouped = []
    for site_id in sites_arr:
      device = Device.query.filter(id == site_id).first()
      records = [r.__dict__[metric] for r in readings if r.device_id == site_id and r.rdate >= time_steps[i] and r.rdate <= time_steps[i+1]]
      div_factor = 1 if metric == "total_kwh" else device.sqm 
      grouped.append(sum(records)/div_factor if div_factor > 0 else 0) #/len(records) if len(records) > 0 else 0.0)
    op_arr.append(grouped)
  for i in range(len(time_steps2) -1 ):
    grouped = []
    for site_id in sites_arr:
      device = Device.query.filter(id == site_id).first()
      records = [r.__dict__[metric] for r in readings2 if r.device_id == site_id and r.rdate >= time_steps2[i] and r.rdate <= time_steps2[i+1]] 
      div_factor = 1 if metric == "total_kwh" else device.sqm
      grouped.append(sum(records)/div_factor if div_factor > 0 else 0) #/len(records) if len(records) > 0 else 0.0)
    op_arr2.append(grouped)
  return op_arr, op_arr2, time_steps, time_steps2, ""

def get_chart_data(sites_arr, ctype, start_date, end_date, agg_period, start_date2 = None, end_date2 = None, metric = "total_kwh", qtype = QUERY_MSITES ):
  print(sites_arr, ctype, start_date, end_date, agg_period, start_date2, end_date2, metric, qtype)
  readings, readings2, errors = _get_readings(sites_arr, start_date, end_date, start_date2, end_date2)
  if len(errors)>0:
    return serialize(pd.DataFrame(), qtype, render_to = CHART_DIV_ID, title = "NotGenerated", kind = ctype), errors

  op_arr, op_arr2, time_steps, time_steps2, errors = _get_op_arr(readings, readings2, sites_arr, start_date, end_date, agg_period, metric, qtype = qtype, start_date2 = start_date2, end_date2 = end_date2)
  print(errors)
  if len(errors)>0:
    return serialize(pd.DataFrame(), qtype, render_to = CHART_DIV_ID, title = "NotGenerated", kind = ctype), errors

  devices = Device.query.filter(Device.id.in_(sites_arr)).all()
  df = pd.DataFrame(op_arr, columns = [x.device_id for x in devices])
  df['time'] = [(time_steps[i] - epoch).total_seconds() * 1000 for i in range(len(time_steps) -1 )]
  print(df.head())

  df2 = None
  if qtype == QUERY_MTIMERANGES:
    df2 = pd.DataFrame(op_arr2, columns = [x.device_id for x in devices])
    df2['time'] = [(time_steps2[i] - epoch).total_seconds() * 1000 for i in range(len(time_steps2) -1 )]

  sd_str, ed_str = start_date.strftime(OUTPUT_DATE_FORMAT), end_date.strftime(OUTPUT_DATE_FORMAT)
  sd_str2, ed_str2 = None, None
  if qtype == QUERY_MTIMERANGES:
    sd_str2, ed_str2 = start_date2.strftime(OUTPUT_DATE_FORMAT), end_date2.strftime(OUTPUT_DATE_FORMAT)
  chart, errors = serialize(df, df2, qtype, metric, sd_str, ed_str, sd_str2, ed_str2, kind = ctype)
  return chart, errors
