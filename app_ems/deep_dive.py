from models import *
import pandas as pd
from app import db

def serialize(df):
  return {}, "TODO"

def get_dd_chart_data(site_id):
  print(site_id)
  device = db.session.query(Device).get(site_id)
  print(device)
  if device is None:
    return {}, "No assosciated device found"

  readings = device.readings
  print(readings[0].__dict__.keys())
  df = pd.DataFrame.from_records([x.__dict__ for x in readings])
  chart, errors = serialize(df)
  return chart, errors
