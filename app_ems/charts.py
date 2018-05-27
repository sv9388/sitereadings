import pandas as pd, numpy as np
from pandas.compat import StringIO
from datetime import datetime, timedelta, date
from models import *
from enums import *
from date_utils import *
from sqlalchemy import func, extract, or_
from app import db

epoch = datetime.utcfromtimestamp(0)

CHART_DIV_ID = "outputchart"
OUTPUT_DATE_FORMAT = "%Y-%m-%d %H:%M"
agg_mapping = {LAST_7_DAYS :  "Last 7 days", LAST_30_DAYS : "Last 30 days", CURRENT_MONTH : "Current Month" , LAST_12_MONTHS : "Last 12 Months", CURRENT_YEAR : "Current Year", ALL_DATA : "All Data"}
 
def serialize(df, df2, qtype, data_type, metric, sd_str, ed_str, sd_str2, ed_str2,  tags, kind): 
  # msites = Time vs Metric (multiple sites per time graph). mtimerange = Site vs Metric (2 tr per range graph)
  chart_json = {"chart" : {"zoomType": 'x', "type" : "", "renderTo" : CHART_DIV_ID}, "title" : {"text" : ""}, "yAxis" : {"title" : {"text" : ""}}, "series" : [], "responsive" : { "rules" : [{"condition" : {"maxWidth" : 500}, "chartOptions": {"legend" : { "layout" : "horizontal", "align" : "center", "verticalAlign" : "bottom"}}}]}}
  chart_json["yAxis"]["title"]["text"] = metric
  chart_json["chart"]["type"] = kind

  if qtype == QUERY_MSITES:
    if data_type == SITE:
      print(df.columns, df.shape)
      chart_json['title']['text'] = "{} between {} and {}".format(metric, sd_str, ed_str)
      chart_json["xAxis"] = { "type" : "datetime" }
      chart_json["series"] = [{"name" : x, "data" : df[['time', x]].values.tolist()} for x in df.columns if x not in ["time", "tag"]]
    elif data_type == TAG:
      chart_json['title']['text'] = "{} comparison for tags {} and {} between dates {} and {}".format(metric, tags[0], tags[1], sd_str, ed_str)        
      chart_json["xAxis"] = { "type" : "datetime" }
      series = []
      device_cols = [x for x in df.columns if x not in ["time", "tag"]]
      df['chart_value'] = df[device_cols].apply(sum, axis = 1)
      device_cols = [x for x in df2.columns if x not in ["time", "tag"]]
      df2['chart_value'] = df2[device_cols].apply(sum, axis = 1)
      series.append({"name" : tags[0], "data" : df[['time', 'chart_value']].values.tolist()})
      series.append({"name" : tags[1], "data" : df2[['time', 'chart_value']].values.tolist()})
      chart_json["series"] = series      
  else:
    chart_json['title']['text'] = "{} comparison for timeranges {} - {} and {} - {}".format(metric, sd_str, ed_str, sd_str2, ed_str2)
    categories =  [x for x in df.columns if x not in [ "time", "tag"]]
    chart_json["xAxis"] = { "categories" : categories }
    if df2 is None or df is None:
      return chart_json, "No valid data found for other the chosen timerange"
    opj = []
    opj.append({"name" : "{} - {}".format(sd_str, ed_str), "data" : [[x, df[x].sum().item()] for x in categories]})
    opj.append({"name" : "{} - {}".format(sd_str2, ed_str2), "data" : [[x, df2[x].sum().item()] for x in categories]})
    chart_json["series"] = opj #me" : x, "data" : df[['time', x]].values.tolist()} for x in df.columns if x != "time"]

  return chart_json, ""

def _get_devices_by_tags(tag):
    q = db.session.query(Device.id).filter(Device.tag_size == tag)
    if tag == "None":
      q = db.session.query(Device.id).filter(Device.tag_size.is_(None))
    size_devices = q.all()
    q = db.session.query(Device.device_id).filter(Device.tag_site_type == tag)
    if tag == "None":
      q = db.session.query(Device.device_id).filter(Device.tag_site_type.is_(None))
    site_type_devices = q.all()
    devices_arr = size_devices + site_type_devices
    devices_arr = [x[0] for x in devices_arr]
    return devices_arr

def _get_readings(sites_arr, tag, start_date, end_date, agg_period, metric, data_type):
  print("####################################################################################################")
  print(sites_arr, tag, start_date, end_date, agg_period, metric, data_type)
  devices_arr = sites_arr
  if not start_date and not end_date:
    if data_type != TAG:
      return []

  if data_type == TAG:
    devices_arr = _get_devices_by_tags(tag)

  print("AggregatedReadings arr = ", devices_arr)
  rq = db.session.query(AggregatedReading.device_id, (func.floor(extract('epoch', AggregatedReading.dayhour)/agg_period)).label('grouped'), (func.max(AggregatedReading.hourly_kwh)-func.min(AggregatedReading.hourly_kwh)).label("diff_kwh"))
  rq = rq.filter(AggregatedReading.device_id.in_(devices_arr))
  rq = rq.filter(AggregatedReading.dayhour >= start_date).filter(AggregatedReading.dayhour < end_date)
  rq = rq.group_by(AggregatedReading.device_id, (func.floor(extract('epoch', AggregatedReading.dayhour)/agg_period)).label('grouped'))
  rq = rq.order_by(AggregatedReading.device_id, (func.floor(extract('epoch', AggregatedReading.dayhour)/agg_period)).label('grouped'))
  readings = rq.all()
  return readings

def _get_weather(sites_arr, tag, start_date, end_date, agg_period, metric, data_type):
  print("####################################################################################################")
  print(sites_arr, tag, start_date, end_date, agg_period, metric, data_type)
  devices_arr = sites_arr
  if not start_date and not end_date:
    if data_type != TAG:
      return []

  if data_type == TAG:
    devices_arr = _get_devices_by_tags(tag)

  print("Weather arr = ", devices_arr)
  rq = db.session.query(Weather.device_id, (func.floor(extract('epoch', Weather.wdate)/agg_period)).label('grouped'), (func.max(Weather.temp)-func.min(Weather.temp)).label("diff_kwh"))
  rq = rq.filter(Weather.device_id.in_(devices_arr))
  rq = rq.filter(Weather.wdate >= start_date).filter(Weather.wdate < end_date)
  rq = rq.group_by(Weather.device_id, (func.floor(extract('epoch', Weather.wdate)/agg_period)).label('grouped'))
  rq = rq.order_by(Weather.device_id, (func.floor(extract('epoch', Weather.wdate)/agg_period)).label('grouped'))
  readings = rq.all()
  return readings

def _get_params(metric):
  print(metric, "kwh" in metric,  "temperature" in  metric, "sqm" in  metric)
  op = []
  if "kwh" in  metric:
    op.append("kwh")
  if "temperature" in  metric:
    op.append("temperature")
  if "sqm" in  metric:
    op.append("sqm")
  if "customvar" in metric:
    op.append("customvar")
  return op

def _get_df(rarr, warr, devices, agg_period, metric, param_file):
  params = _get_params(metric)
  wdf = pd.DataFrame(warr, columns = ["device_id", "grouped", "kwh"])
  rdf = pd.DataFrame(rarr, columns = ["device_id", "grouped", "temperature"])
  df = pd.merge(wdf, rdf, on=["device_id", "grouped"])
  df['sqm'] = 1
  df['customvar'] = 0
  if "customvar" in params:
    custom_df = pd.read_csv(param_file)
    custom_df.columns = ["device_id", "customvalue"]
    cdict = {x.device_id : x.customvalue for x in custom_df.iterrows()}
    for k, v in cdict.items():
      df[df['device_id'] == k]['customvar'] = v
  for k, v in devices.items():
    df[df['device_id'] == k]['sqm'] = v[1] # /= devices[x][1]

  eval_metric = metric
  dkeys = "_".join([str(x) for x in devices.keys()])
  for x in params:
    eval_metric = eval_metric.replace(x, "df['{}']".format(x))
  for x in ["sqrt", "power"]:
    eval_metric = eval_metric.replace(x, "np.{}".format(x))

  df['op_value'] = eval(eval_metric) 
  df.to_csv("kwh_by_temp_{}.csv".format(dkeys))
  print(devices)
  df = df.pivot(index="grouped", columns="device_id", values="op_value")
  df = df.rename(columns = {k : v[0] for k, v in devices.items()})
  print("Devices = ", df.columns.tolist())
  df['time'] = df.index * agg_period * 1000 if not df.empty else 0
  return df


def _get_raw_data(func, sites_arr, tag1, tag2, start_date, end_date, start_date2, end_date2, agg_period, metric, qtype, data_type):
  readings = func(sites_arr, tag1, start_date, end_date, agg_period, metric, data_type)
  readings2 = []
  print( "QUERY_MTIMERANGES"  if qtype == QUERY_MTIMERANGES else "QUERY_SITES", "TAG" if data_type == TAG else "SITE")
  if qtype == QUERY_MTIMERANGES:
    readings2 = _get_readings(sites_arr, tag1, start_date2, end_date2, agg_period, metric, data_type)
  elif data_type == TAG:
    readings2 = _get_readings(sites_arr, tag2, start_date, end_date, agg_period, metric, data_type)
  return readings, readings2

def get_chart_data(sites_arr, ctype, start_date, end_date, agg_period, start_date2 = None, end_date2 = None, tag1 = None, tag2 = None, metric = "kwh", param_file = None, qtype = QUERY_MSITES, data_type = SITE ):
  metric = metric.lower()
  readings, readings2 = _get_raw_data(_get_readings, sites_arr, tag1, tag2, start_date, end_date, start_date2, end_date2, agg_period, metric, qtype, data_type)
  weather, weather2 = _get_raw_data(_get_weather, sites_arr, tag1, tag2, start_date, end_date, start_date2, end_date2, agg_period, metric, qtype, data_type)
 
  devices = Device.query.filter(Device.id.in_(sites_arr)).all() if data_type == SITE else Device.query.filter(or_(Device.tag_site_type == tag1, Device.tag_size == tag1)).all()
  devices = { x.id : [x.device_unique_name, x.sqm] for x in devices}
  print("First set of devices" , devices)
  print("Len of first readings set = ", len(readings))
  print("Len of second readings set = ", len(readings2)) 
  df = _get_df(readings, weather, devices, agg_period,  metric, param_file) 
  df2 = None
  if qtype == QUERY_MTIMERANGES:
    df2 = _get_df(readings2, weather2, devices,agg_period, metric, param_file)
  elif data_type == TAG:
    devices2 = Device.query.filter(or_(Device.tag_site_type == tag2, Device.tag_size == tag2)).all()
    devices2 = { x.id : [x.device_unique_name, x.sqm] for x in devices2}
    df2 = _get_df(readings2, weather2, devices2, agg_period, metric, param_file)
    print(devices2)

  sd_str, ed_str = start_date.strftime(OUTPUT_DATE_FORMAT), end_date.strftime(OUTPUT_DATE_FORMAT)
  sd_str2, ed_str2 = None, None
  if qtype == QUERY_MTIMERANGES:
    sd_str2, ed_str2 = start_date2.strftime(OUTPUT_DATE_FORMAT), end_date2.strftime(OUTPUT_DATE_FORMAT)

  tags = []
  if data_type == TAG:
    tags = [tag1, tag2]
  chart, errors = serialize(df, df2, qtype, data_type, metric, sd_str, ed_str, sd_str2, ed_str2, tags, kind = ctype)
  return chart, errors
