import pandas as pd
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

def _get_readings(sites_arr, tag, start_date, end_date, agg_period, metric, data_type):
  print("####################################################################################################")
  print(sites_arr, tag, start_date, end_date, agg_period, metric, data_type)
  devices_arr = sites_arr
  if not start_date and not end_date:
    if data_type != TAG:
      return []

  if data_type == TAG:
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

  print("Readings arr = ", devices_arr)
  rq = db.session.query(Reading.device_id, (func.floor(extract('epoch', Reading.rdate)/agg_period)).label('grouped'), (func.max(Reading.total_kwh)-func.min(Reading.total_kwh)).label("value"))
  rq = rq.filter(Reading.device_id.in_(devices_arr))
  rq = rq.filter(Reading.rdate >= start_date).filter(Reading.rdate < end_date)
  rq = rq.group_by(Reading.device_id, (func.floor(extract('epoch', Reading.rdate)/agg_period)).label('grouped'))
  rq = rq.order_by(Reading.device_id, (func.floor(extract('epoch', Reading.rdate)/agg_period)).label('grouped'))
  readings = rq.all()
  return readings

def _get_df(op_arr, devices, agg_period, metric):
  df = pd.DataFrame(op_arr, columns = ["device_id", "grouped", "value"])
  df = df.pivot(index="grouped", columns="device_id", values="value")
  if metric == "kwh_psqm":
    for x in df.columns:
      df[x] /= devices[x][1]
  print("Devices = ", df.columns.tolist())
  #df.columns = [devices[x][0] for x in df.columns.tolist()]
  df['time'] = df.index * agg_period * 1000 if not df.empty else 0
  return df

def get_chart_data(sites_arr, ctype, start_date, end_date, agg_period, start_date2 = None, end_date2 = None, tag1 = None, tag2 = None, metric = "total_kwh", qtype = QUERY_MSITES, data_type = SITE ):
  readings = _get_readings(sites_arr, tag1, start_date, end_date, agg_period, metric, data_type)
  readings2 = []
  print( "QUERY_MTIMERANGES"  if qtype == QUERY_MTIMERANGES else "QUERY_SITES", "TAG" if data_type == TAG else "SITE")
  if qtype == QUERY_MTIMERANGES:
    readings2 = _get_readings(sites_arr, tag1, start_date2, end_date2, agg_period, metric, data_type)
  elif data_type == TAG:
    readings2 = _get_readings(sites_arr, tag2, start_date, end_date, agg_period, metric, data_type)   
 
  devices = Device.query.filter(Device.id.in_(sites_arr)).all() if data_type == SITE else Device.query.filter(or_(Device.tag_site_type == tag1, Device.tag_size == tag1)).all()
  devices = { x.id : ["{} | {} | {} ".format(x.device_id, x.distributer_name, x.project), x.sqm] for x in devices}
  print("First set of devices" , devices)
  print("Len of first readings set = ", len(readings))
  print("Len of second readings set = ", len(readings2)) 
  df = _get_df(readings, devices, agg_period, metric) 
  df2 = None
  if qtype == QUERY_MTIMERANGES:
    df2 = _get_df(readings2,  devices, agg_period, metric)
  elif data_type == TAG:
    devices2 = Device.query.filter(or_(Device.tag_site_type == tag2, Device.tag_size == tag2)).all()
    devices2 = { x.id : ["{} | {} | {} ".format(x.device_id, x.distributer_name, x.project), x.sqm] for x in devices2}
    df2 = _get_df(readings2, devices2, agg_period, metric)
    print(devices2)
  print(qtype, data_type, len(readings2), df2)

  sd_str, ed_str = start_date.strftime(OUTPUT_DATE_FORMAT), end_date.strftime(OUTPUT_DATE_FORMAT)
  sd_str2, ed_str2 = None, None
  if qtype == QUERY_MTIMERANGES:
    sd_str2, ed_str2 = start_date2.strftime(OUTPUT_DATE_FORMAT), end_date2.strftime(OUTPUT_DATE_FORMAT)

  tags = []
  if data_type == TAG:
    tags = [tag1, tag2]
  chart, errors = serialize(df, df2, qtype, data_type, metric, sd_str, ed_str, sd_str2, ed_str2, tags, kind = ctype)
  return chart, errors
