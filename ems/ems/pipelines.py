from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from .models import *
from .settings import DATABASE

class EmsPipeline(object):
  def __init__(self):
    print("BB", DATABASE)
    engine = create_engine(DATABASE)
    session = sessionmaker()
    session.configure(bind=engine)
    self.session = session()

  def process_item(self, item, spider):
    print(spider.name, item.keys())
    if not "device_id" in item.keys():
      return {}

    if spider.name != "psines":
      return item

    device_id = item["device_id"]
    device_entity = self.session.query(Device).filter(Device.device_id == device_id).first()

    if not device_entity:
      device_entity = Device(device_id = device_id)
      try:
        self.session.add(device_entity)
        self.session.commit()
      except:
        self.session.rollback()

    device_dbid = device_entity.id
    readings = item["readings"]
    for i in range(len(readings)):
      readings[i]["device_id"] = device_dbid
    reading_dbents = [Reading(**x) for x in readings]
    try:
      self.session.add_all(reading_dbents)
      self.session.commit()
      print("Added")
    except Exception as e:
      print("Rolled back", e)
      self.session.rollback()
    return {}

class WeatherPipeline(object):
  def __init__(self):
    print("BB", DATABASE)
    engine = create_engine(DATABASE)
    session = sessionmaker()
    session.configure(bind=engine)
    self.session = session()

  def process_item(self, item, spider):
    print(spider.name, item.keys())
    if not "device_id" in item.keys():
      return {}

    if spider.name != "weather":
      return {}

    device_id = item["device_id"]
    print(device_id)
    device_entity = self.session.query(Device).filter(Device.device_id == device_id).first()
    print(device_id, device_entity)
 
    if not device_entity:
      return {} 

    device_dbid = device_entity.id
    weather_readings = item["weather_readings"]
    for i in range(len(weather_readings)):
      weather_readings[i]["device_id"] = device_dbid
    reading_dbents = [Weather(**x) for x in weather_readings]
    print(reading_dbents[0])
    try:
      self.session.add_all(reading_dbents)
      self.session.commit()
      print("Added")
    except Exception as e:
      print("Rolled back", e)
      self.session.rollback()
    return {} 
