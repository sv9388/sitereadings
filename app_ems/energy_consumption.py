import bokeh
import ec_utils as utils
import warnings
warnings.filterwarnings('ignore')

from ec_utils import ComputeAverages, FilterFunc, gaussian_smooth

#TOOLS="hover,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,save"
#   temperatura_percepita
#   temperatura
TEMPERATURE = "temp" #eratura_percepita"

# base analysis on energy consumption between these hours
HOUR_SPANS = [(10, 12), (16,18)]
CUT_DATE = '2018-02-01'

def _get_iecc_chart(site_id):
  bins = 16
  palette = bokeh.palettes.viridis(bins)
  merged, _ = utils.get_merged(site_id, HOUR_SPANS, temperature_col=TEMPERATURE, cut_date=CUT_DATE)

  METHOD = "dynamic" # 'dynamic' or 'fixed'
  MIN_SAMPLE = 15      # minumum samples needed to calculate an average for each temperature
  MAX_WINDOW_SIZE = 9  # maximum window to consider. if reached, uses only samples found in within MAX_WINDOW_SIZE
  WINDOW_SIZE = 5 # no matter how many samples are found in the window this is the window size that will be used
  SIGMA_FACTOR = 0.5     # STDs below average energy usage for baseline
  OUTLIER_FACTOR = 1.5   # computes outliers as daily average lower than 25th percentile * OUTLIER_FACTOR,
  #AVG_METHOD = "median"  # choices are: mean, median
  LIN_REG_FROM = 25      # applicate linear regression between this temperature and max temperature
  filter_func = FilterFunc(OUTLIER_FACTOR)
  
  YEARS = "all"
  MONTHS = "all"
  DAYS_OF_WEEK = "all"
  PERIODS = "whole" #[ ("2012-01-01", "2017-12-31"), ("2016-01-01", "2017-12-31")]
  PLOT_EACH_YEAR = True
  COMPARE_DAYS = ["working", "saturdays", "sundays"]
  
  PRINT_TABLE = False # Print data for each temperature
  calc_avg = ComputeAverages(METHOD, min_sample=MIN_SAMPLE, max_window_size=MAX_WINDOW_SIZE, window_size=WINDOW_SIZE, filter_fn=filter_func, smooth_fn=gaussian_smooth, sigma=SIGMA_FACTOR, lin_reg_from=LIN_REG_FROM)
  
  filters = utils.gen_filters(YEARS, MONTHS, DAYS_OF_WEEK, PERIODS)
  merged = merged.dropna()
  op = utils.plot_consumption_curve(merged, palette, bins, filters=filters, avg_fn=calc_avg, plot_each_year=PLOT_EACH_YEAR, compare_days=COMPARE_DAYS, print_table=PRINT_TABLE)
  return op  
 
def get_ideal_energy_consumption_curves_chart(site_id):
  op =  _get_iecc_chart(site_id)
  chart_op = {"Ideal Energy Consumption Curves" : op}
  return chart_op, ""

#get_ideal_energy_consumption_curves_chart(1)
