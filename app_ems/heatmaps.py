import bokeh 
import hm_utils as utils
import warnings
warnings.filterwarnings('ignore')
from hm_utils import ComputeAverages, FilterFunc, gaussian_smooth

TEMPERATURE = "temp"

# base analysis on energy consumption between these hours
HOUR_SPANS = [(10, 12), (16,18)]
CUT_DATE = '2018-09-01 00:00'

def _get_all_graphs(site_id):
  chart_op = {}
  bins = 16
  palette = bokeh.palettes.viridis(bins)
  merged, df = utils.get_merged(site_id, HOUR_SPANS, cut_date=CUT_DATE, temperature_col=TEMPERATURE)
  print("MErged = ", merged.shape, merged.columns)
  print("df w d = ", df.shape, df.columns) 
  METHOD = "dynamic" # 'dynamic' or 'fixed'
  MIN_SAMPLE = 15      # minumum samples needed to calculate an average for each temperature
  MAX_WINDOW_SIZE = 9  # maximum window to consider. if reached, uses only samples found in within MAX_WINDOW_SIZE
  WINDOW_SIZE = 5 # no matter how many samples are found in the window this is the window size that will be used
  SIGMA_FACTOR = 0.5     # STDs below average energy usage for baseline
  OUTLIER_FACTOR = 1.5   # computes outliers as daily average lower than 25th percentile * OUTLIER_FACTOR,
  LIN_REG_FROM = 25      # applicate linear regression between this temperature and max temperature
  filter_func = FilterFunc(OUTLIER_FACTOR)
  
  YEARS = "all"
  MONTHS = "all"
  DAYS_OF_WEEK = "all"
  PERIODS = "whole" #[ ("2012-01-01", "2017-12-31"), ("2016-01-01", "2017-12-31")]
  PLOT_EACH_YEAR = True
  
  calc_avg = ComputeAverages(METHOD, min_sample=MIN_SAMPLE, max_window_size=MAX_WINDOW_SIZE, window_size=WINDOW_SIZE, filter_fn=filter_func, smooth_fn=gaussian_smooth, sigma=SIGMA_FACTOR, lin_reg_from=LIN_REG_FROM)
 
  filters = utils.gen_filters(YEARS, MONTHS, DAYS_OF_WEEK, PERIODS)
  
  palette = bokeh.palettes.Inferno256[::16]
  op = utils.plot_heatmap(df, palette=palette)
  chart_op["True Energy Consumption Heatmap"] = op
  
  df_reconstructed = utils.reconstruct_by_weekday(df, merged, calc_avg)
  op = utils.plot_heatmap(df_reconstructed, palette=palette, filters=filters, by_year=PLOT_EACH_YEAR)
  chart_op["Reconstructed Energy Consumption Heatmap"] = op
  
  ENERGY2EURO = 0.17
  BY = "year"
  MONTHS = "all"
  WEEKDAYS = "all"
  op = utils.plot_saving(df, df_reconstructed, by=BY, periods=PERIODS, months=MONTHS, weekdays=WEEKDAYS, energy2euro=ENERGY2EURO)
  chart_op["Potential Saving"] = op
  return chart_op
 
def get_dd_chart_data(site_id):
  chart_op = _get_all_graphs(site_id)
  for k, v in chart_op.items():
    print("########################################################################################################")
    print(k)
    for script, div in v:
      with open("./bkh_op_{}.txt".format(k), "w") as f:
        f.write(script)
        f.write(div)
  return chart_op, ""

#get_dd_chart_data(1)
