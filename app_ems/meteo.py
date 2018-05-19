import os
import pandas as pd
from models import *

EN_COLS = "weather_city_code	date	weather_symbol_id	rainfall	rainfall_intensity	rainfall_prob	rainfall_uom	temperature	perceived_temperature	freezing_level	snow_level	wind_direction	wind_intensity	sea_power	sea_temperature	flurry	humidity	pressure	uv_index	uv_description	accumulation	windchill	solar_radiation	effective".split("\t")

def load_meteo_data(site_id):
  device = Device.query.filter_by(id = site_id).first()
  if not device:
    return {}, "No such device with site id {} found".format(site_id)
  weathers = device.weather
  arr = [[x.wdate, x.temp] for x in weathers]
  meteo_data = pd.DataFrame(arr, columns = ["date", "temp"])
  meteo_data.to_csv("./wt_ezhavu.csv")
  meteo_data["data"] = pd.to_datetime(meteo_data["date"]).dt.date
  meteo_data["hour"] = pd.to_datetime(meteo_data["date"]).dt.hour
  return meteo_data

