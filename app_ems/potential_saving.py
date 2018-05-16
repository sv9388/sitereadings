import pandas as pd
import bokeh
import meteo
import dd_settings as settings
import dd_utils as utils
import warnings
warnings.filterwarnings('ignore')

from dd_utils import ComputeAverages, FilterFunc, gaussian_smooth

TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,save"
#settings.set_setting("DATA_DIR", "/mnt/hdd_data/innowatio/20170315 - Energy - new")
#settings.set_setting("METEO_DIR", "/mnt/hdd_data/innowatio/20170320 - Meteo")

settings.set_setting("DATA_DIR", "data/pod_energy")
settings.set_setting("CACHE_DIR", "data/cache/preprocessed/")
settings.set_setting("POD_INFO_FILEPATH", "data/pod_location.csv")

settings.set_setting("METEO_DIR", "data/meteo")
settings.set_setting("METEO_NEAREST_STATION_FILEPATH", "data/pod_nearest_weather_station.csv")
settings.set_setting("METEO_STATION_METHODS", ("commune", "nearest")) # look first if we can find a weather station in the same commune, if no weather station look in file for nearest weather station
POD_ID = "IT001E15022242" # OVS Curno

# options:
#   temperatura_percepita
#   temperatura
TEMPERATURE = "temperatura_percepita"

# base analysis on energy consumption between these hours
HOUR_SPANS = [(10, 12), (16,18)]
CUT_DATE = '2017-08-01 00:00'

# to speed up data loading process use cache
# if data has changed in DATA_DIR or METEO_DIR, set to False to load new changes
USE_CACHE = True
def _get_all_graphs(POD_ID):
  bins = 16
  palette = bokeh.palettes.viridis(bins)
  merged, df = utils.get_merged(POD_ID, HOUR_SPANS, cut_date=CUT_DATE, temperature_col=TEMPERATURE, use_cache=USE_CACHE)
  from dd_utils import ComputeAverages, FilterFunc, gaussian_smooth
  
  METHOD = "dynamic" # 'dynamic' or 'fixed'
  MIN_SAMPLE = 15      # minumum samples needed to calculate an average for each temperature
  MAX_WINDOW_SIZE = 9  # maximum window to consider. if reached, uses only samples found in within MAX_WINDOW_SIZE
  WINDOW_SIZE = 5 # no matter how many samples are found in the window this is the window size that will be used
  SIGMA_FACTOR = 0.5     # STDs below average energy usage for baseline
  OUTLIER_FACTOR = 1.5   # computes outliers as daily average lower than 25th percentile * OUTLIER_FACTOR,
  AVG_METHOD = "median"  # choices are: mean, median
  LIN_REG_FROM = 25      # applicate linear regression between this temperature and max temperature
  filter_func = FilterFunc(OUTLIER_FACTOR)
  smooth_func = gaussian_smooth
  calc_avg = ComputeAverages(METHOD, avg_method=AVG_METHOD, min_sample=MIN_SAMPLE, max_window_size=MAX_WINDOW_SIZE, window_size=WINDOW_SIZE, filter_fn=filter_func, smooth_fn=gaussian_smooth, sigma=SIGMA_FACTOR)
  YEARS = "all"
  PAIRS = [[("energy_std", "energy_mean"),("energy_std", "samples")],
          [("energy_std", "win_size"),("energy_mean", "samples")]]
  
  chart_op = {}
  chart_op["Energy Consumption Dispersity Plots"] = utils.plot_energy_dispersity(merged, calc_avg, PAIRS, years=YEARS)
  
  YEARS = "all"
  MONTHS = "all"
  DAYS_OF_WEEK = "all"
  PERIODS = "whole" #[ ("2012-01-01", "2017-12-31"), ("2016-01-01", "2017-12-31")]
  PLOT_EACH_YEAR = True
  COMPARE_DAYS = ["working", "saturdays", "sundays"]
  
  PRINT_TABLE = False # Print data for each temperature
  calc_avg = ComputeAverages(METHOD, min_sample=MIN_SAMPLE, max_window_size=MAX_WINDOW_SIZE, window_size=WINDOW_SIZE, filter_fn=filter_func, smooth_fn=gaussian_smooth, sigma=SIGMA_FACTOR, lin_reg_from=LIN_REG_FROM)
  
  filters = utils.gen_filters(YEARS, MONTHS, DAYS_OF_WEEK, PERIODS)
  op = utils.plot_consumption_curve(merged, palette, bins, filters=filters, avg_fn=calc_avg, plot_each_year=PLOT_EACH_YEAR, compare_days=COMPARE_DAYS, print_table=PRINT_TABLE)
  print(len(op))
  chart_op["Ideal Energy Consumption Curves"] = op
  
  palette = bokeh.palettes.Inferno256[::16]
  op = utils.plot_heatmap(df, palette=palette)
  chart_op["True Energy Consumption Heatmap"] = op
  
  df_reconstructed = utils.reconstruct_by_weekday(df, merged, calc_avg)
  op = utils.plot_heatmap(df_reconstructed, palette=palette, filters=filters, by_year=PLOT_EACH_YEAR)
  chart_op["Reconstructed Energy Consumption Heatmap"] = op
  
  ENERGY2EURO = 0.17
  BY = "year"
  PERIOD = (None, "2017-12-31")
  MONTHS = "all"
  WEEKDAYS = "all"
  op = utils.plot_saving(df, df_reconstructed, by=BY, periods=PERIODS, months=MONTHS, weekdays=WEEKDAYS, energy2euro=ENERGY2EURO)
  chart_op["Potential Saving"] = op
  return chart_op
 
def get_dd_chart_data(site_id):
  chart_op = _get_all_graphs(POD_ID)
  for k, v in chart_op.items():
    print(k)
    for sd in v:
      print(k, len(v), type(sd))
  return chart_op
