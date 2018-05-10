# coding: utf-8

import calendar
import datetime
import datetime as dt
import math
import os
import time

import numpy as np
import pandas as pd

from skimage import exposure
from scipy import stats
from scipy import ndimage

import bokeh
from bokeh.io import output_notebook, output_file, show, push_notebook, save
from bokeh.plotting import figure
#import bokeh.palettes
from bokeh.layouts import gridplot, row, column
from bokeh.embed import components

#from bkcharts import HeatMap
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    FixedTicker,
    PrintfTickFormatter,
    DatetimeTickFormatter,
    ColorBar,
    FuncTickFormatter,
)


from dd_settings import get_setting

import sys
EPSILON = sys.float_info.epsilon  # smallest possible difference

def convert_to_rgb(minval, maxval, val, colors):
    fi = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    i = int(fi)
    f = fi - i
    if f < EPSILON:
        return "#%02x%02x%02x" % (colors[i])
    (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
    return "#%02x%02x%02x" % (int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1)))

def normalize(values):
    minval, maxval = min(values), max(values)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # [BLUE, GREEN, RED]
    op_arr = [convert_to_rgb(minval, maxval, val, colors) for val in values]
    print(op_arr)
    return op_arr

def get_pod_info(pod_id, pod_info_filepath=None):
    if pod_info_filepath is None:
        pod_info_filepath = get_setting("POD_INFO_FILEPATH")
    locations = pd.read_csv(pod_info_filepath, sep=",", index_col="pod")
    return locations.loc[pod_id].to_dict()

def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        return True
    return False

def load_dataset(path, normalize=False, cut=False, use_cache=True):

    name = "%s_%s_%s" % (os.path.basename(path).split(".")[0], normalize, cut)

    cache_dir = os.path.join(get_setting("CACHE_DIR"), "dataset")
    ensure_dir(cache_dir)
    preprocessed_filepath = os.path.join(cache_dir, "%s.csv" % name)

    if use_cache and os.path.exists(preprocessed_filepath):
        res = pd.read_csv(preprocessed_filepath, index_col=0, parse_dates=["date", "data", "ora"])
        return res

    if not os.path.exists(path):
        raise Exception("Dataset not found: %s" % (path))

    df = pd.read_csv(path, sep=",").rename(columns={'timestamp': 'date', 'active_power': 'active', 'metric_value': 'active'}) #TODO: Get data from reading table here
    if 'anomaly_level' not in df.columns:
        df.loc[:, 'anomaly_level'] = 0
    else:
        df.loc[:, 'anomaly_level'] = df['anomaly_level'].apply(lambda x: 1 if x == 'HIGH' else 0)
    df.loc[:, 'date'] = pd.to_datetime(df['date'])

    if normalize:
        max_ac, min_ac = df['active'].max(), df['active'].min()
        df.loc[:, 'active'] = (df.loc[:, 'active'] - min_ac) / (max_ac - min_ac)
    df = df.set_index('date')
    df = df.resample("15T").asfreq()
    df = df.reset_index()
    res = df.loc[:, ['date', 'active', 'pod', 'anomaly_level']]

    if cut:
        res["reconstructed"] = 1
        valid_dates = get_valid_dates(res)
        res.loc[res.date.apply(lambda x: x.date() in valid_dates), 'reconstructed'] = 0

    res.to_csv(preprocessed_filepath)
    return res



def get_labels(values, hist_bins, bins=10):
    '''
    Get bin centers, cdf, palette binning and return the corresponding real energy bins as labels.
    '''
    cdf, bin_centers = exposure.cumulative_distribution(values, nbins=bins)
    labels = []
    for el in hist_bins:
        index = np.where(cdf<=el)[0][-1]
        if index >= len(cdf)-1:
            labels.append(values.max())
        else:
            labels.append((el - cdf[index])/(cdf[index+1]-cdf[index]) *
                          (bin_centers[index+1]-bin_centers[index]) +
                          bin_centers[index])
        labels[0] = 0
    return labels


TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

def heatmap(dataset, pod_name, palette):
    '''
    Plot heat map of consumption with colorbar.
    '''
    bins = len(palette)
    dataset.loc[:, 'tooltip'] = dataset['date'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") + ' ' + calendar.day_name[x.weekday()])
    dataset.loc[:, 'data'] = dataset['date'].apply(lambda x: pd.to_datetime(x.date()))
    dataset.loc[:, 'data+1'] = dataset['data'].apply(lambda x: x + np.timedelta64(1, 'D'))
    dataset.loc[:, 'tempo'] = dataset['date'].apply(lambda x: datetime.datetime(2000, 1, 1, x.hour, x.minute))
    dataset.loc[:, 'tempo+1'] = dataset['tempo'].apply(lambda x: x + np.timedelta64(15, 'm'))
    dataset.loc[:, 'active_hist'] = exposure.equalize_hist(dataset['active'].values, nbins=bins)
    mapper = LinearColorMapper(palette=palette,
                               low=dataset['active_hist'].min(),
                               high=dataset['active_hist'].max())

    source = ColumnDataSource(dataset)

    p = figure(title='Energy consumption for {}'.format(pod_name),
               x_axis_type='datetime',
               x_axis_location='below', plot_width=900, plot_height=400,
               tools=TOOLS, toolbar_location='above')

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Time'
    p.yaxis[0].formatter = DatetimeTickFormatter()

    p.quad(left="data", right="data+1",
       bottom="tempo", top="tempo+1",
       source=source,
       fill_color={'field': 'active_hist', 'transform': mapper},
       line_alpha={'field': 'anomaly_level'},
       line_color='black',
       fill_alpha=1,
       line_width=3)

    ticker = FixedTicker()
    ticker.ticks = list(np.histogram(dataset['active_hist'], bins=bins)[1])
    bin_labels = [round(a, 1) for a in get_labels(dataset['active'].values, ticker.ticks, bins=bins)]
    label_dict = {}

    max_label = max(bin_labels)
    if max_label >= 100:
        bin_labels = ["%03d" % s for s in bin_labels]
    else:
        bin_labels = ["%04.1f" % s for s in bin_labels]

    for i, s in zip(ticker.ticks, bin_labels):
        label_dict[i] = s

    formatter = FuncTickFormatter(code="""
                                        var labels = %s;
                                        return labels[tick];
                                       """ % label_dict)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                         ticker=ticker,
                         formatter=formatter,
                         label_standoff=10, border_line_color=None, location=(-10, 10))
    color_bar.title = '[KWh]'
    p.add_layout(color_bar, 'right')

    p.select_one(HoverTool).tooltips = [
         ('date', '@tooltip'),
         ('energy(KWh)', '@active'),
    ]
    return p


BASE_YEAR = 2016

def conv_plus_one(x):
    if x.month == 1 and x.day == 1:
        return datetime.date(BASE_YEAR+1, x.month, x.day)
    else:
        return datetime.date(BASE_YEAR, x.month, x.day)

def conv(x):
    return datetime.date(BASE_YEAR, x.month, x.day)

def heatmap(dataset, pod_name, palette, x_range=None, y_range=None, active_low=None, active_high=None):
    '''
    Plot heat map of consumption with colorbar.
    '''

    bins = len(palette)

    dataset['tooltip'] = dataset['date'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") + ' ' + calendar.day_name[x.weekday()])

    dataset["tempo"] = dataset['date'].apply(lambda x: datetime.datetime(2000, 1, 1, x.hour, x.minute))
    dataset["tempo+1"] = dataset["tempo"] + np.timedelta64(15, "m")
    dataset["tempo-1"] = dataset["tempo"] - np.timedelta64(15, "m")

    dataset['data'] = pd.to_datetime(dataset['date'].dt.date)
    dataset['data+1'] = dataset['data'] + pd.to_timedelta(np.timedelta64(1, 'D'))

    dataset["data"] = dataset["data"].apply(conv)
    dataset["data+1"] = dataset["data+1"].apply(conv_plus_one)

    dataset = dataset.dropna(how='any', subset=["active"])

    if active_low is None:
        active_low = dataset['active'].min()
    if active_high is None:
        active_high = dataset['active'].max()

    mapper = LinearColorMapper(palette=palette,
                               low=active_low,
                               high=active_high)

    source = ColumnDataSource(dataset)

    fig_params = dict(title='Energy consumption for {}'.format(pod_name),
               x_axis_type='datetime',
               y_axis_type="datetime",
               x_axis_location='below', plot_width=900, plot_height=400,
               tools=TOOLS, toolbar_location='above')
    if x_range:
        fig_params["x_range"] = x_range
    if y_range:
        fig_params["y_range"] = y_range

    p = figure(**fig_params)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Time'
    p.yaxis[0].formatter = DatetimeTickFormatter()

    p.quad(left="data", right="data+1",
        bottom="tempo", top="tempo+1",
       #bottom="tempo-1", top="tempo",
       source=source,
       fill_color={'field': 'active', 'transform': mapper},
       line_alpha={'field': 'anomaly_level'},
       line_color='lime',
       fill_alpha=1,
       line_width=3)

    template = ""
    if active_high > 100:
        template = "%03d"
    else:
        template = "%04.1f"

    ticker = FixedTicker()
    ticker.ticks = np.linspace(active_low, active_high, bins)
    label_dict = dict([(v, template % round(v)) for v in ticker.ticks])

    formatter = FuncTickFormatter(code="""
                                        var labels = %s;
                                        return labels[tick];
                                       """ % label_dict)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                         ticker=ticker,
                         formatter=formatter,
                         label_standoff=10, border_line_color=None, location=(-10, 10))
    color_bar.title = '[KWh]'
    p.add_layout(color_bar, 'right')

    p.select_one(HoverTool).tooltips = [
         ('date', '@tooltip'),
         ('energy(KWh)', '@active'),
    ]
    return p


def heatmap1(dataset, pod_name, palette):
    '''
    Plot heat map of consumption with colorbar.
    '''
    bins = len(palette)



    dataset.loc[:, 'tooltip'] = dataset['date'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") + ' ' + calendar.day_name[x.weekday()])
    dataset.loc[:, 'data'] = dataset['date'].apply(lambda x: pd.to_datetime(x.date()))
    dataset.loc[:, 'tempo'] = dataset['date'].apply(lambda x: datetime.datetime(2000, 1, 1, x.hour, x.minute))

    mapper = LinearColorMapper(palette=palette,
                               low=dataset['active'].min(),
                               high=dataset['active'].max())

    source = ColumnDataSource(dataset)

    p = figure(title='Energy consumption for {}'.format(pod_name),
               x_axis_type='datetime',
               x_axis_location='below', plot_width=900, plot_height=400,
               tools=TOOLS, toolbar_location='above')

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Time'
    p.yaxis[0].formatter = DatetimeTickFormatter()

    p.square(x="data", y="tempo",
       source=source,
       fill_color={'field': 'active', 'transform': mapper},
       line_alpha={'field': 'anomaly_level'},
       line_color='black',
       fill_alpha=1,
       line_width=0)

    ticker = FixedTicker()
    ticker.ticks = list(np.histogram(dataset['active'], bins=bins)[1])
    #bin_labels = [round(a, 1) for a in get_labels(dataset['active'].values, ticker.ticks, bins=bins)]
    bin_labels = [round(a, 1) for a in ticker.ticks]
    label_dict = {}


    max_label = max(bin_labels)
    if max_label >= 100:
        bin_labels = ["%03d" % s for s in bin_labels]
    else:
        bin_labels = ["%04.1f" % s for s in bin_labels]

    for i, s in zip(ticker.ticks, bin_labels):
        label_dict[i] = s

    formatter = FuncTickFormatter(code="""
                                        var labels = %s;
                                        return labels[tick];
                                       """ % label_dict)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                         ticker=ticker,
                         formatter=formatter,
                         label_standoff=10, border_line_color=None, location=(-10, 10))
    color_bar.title = '[KWh]'
    p.add_layout(color_bar, 'right')

    p.select_one(HoverTool).tooltips = [
         ('date', '@tooltip'),
         ('energy(KWh)', '@active'),
    ]
    return p

def get_plot(path, locations, palette=bokeh.palettes.Spectral10, normalize=False):
    df = load_dataset(path, normalize=normalize)
    file_name = os.path.basename(path)
    return plot(df, locations, file_name, palette)

def plot(df, locations, file_name, palette=bokeh.palettes.Spectral10):

    try:
        name, description, province = locations.loc[df['pod'].iloc[0], ['company', 'description', 'province_shortname']].values
        title = " ".join([name, description, province, " "])
    except:
        title = "Unknown "

def plot_heatmap(df, palette=bokeh.palettes.Spectral10, filters=None, by_year=True):
    op = None

    df = apply_filters(df, filters, date_col="date")

    active_low = df["active"].min()
    active_high = df["active"].max()

    if by_year:
        plots = []
        x_range = None
        y_range = None
        for year, df_year in df.groupby(by="year"):
            try:
                pod_id = df.iloc[0]["pod"]
                pod_info = get_pod_info(pod_id)
                title = " ".join([pod_id, pod_info["company"], pod_info["description"], pod_info["province_shortname"]])
                title = "%s, year %d" % (title, year)
            except:
                title = "Unknown "

            p = heatmap(df_year, pod_name=title, palette=palette, x_range=x_range, y_range=y_range, active_low=active_low, active_high=active_high)
            plots.append([p])

            if x_range is None:
                x_range = p.x_range
                y_range = p.y_range

        op = [components(gridplot(plots))]
    else:
        try:
            pod_id = df.iloc[0]["pod"]
            pod_info = get_pod_info(pod_id)
            title = " ".join([pod_id, pod_info["company"], pod_info["description"], pod_info["province_shortname"]])

        except:
            title = "Unknown "

        op = [components(heatmap(df, pod_name=title, palette=palette, active_low=active_low, active_high=active_high))]
    return op

def get_month(ds, year_i, month_i):
    g = ds.groupby(ds['date'].map(lambda x: x.year))
    year = sorted(g.groups.keys())[year_i]
    ds_year = ds.loc[g.groups[year]]
    g_month = ds_year.groupby(ds_year['date'].map(lambda x: x.month))
    month = sorted(g_month.groups.keys())[month_i]
    ds_month = ds_year.loc[g_month.groups[month]]
    return ds_month.active.values

def get_period(df, timestamp1, timestamp2):
    d1 = datetime.datetime.strptime(timestamp1, "%Y-%m-%d")
    d3 = datetime.datetime.strptime(timestamp2, "%Y-%m-%d")
    d4 = d3 + datetime.timedelta(days=1)
    res = df.loc[(df['date'] >= d1) & (df['date'] < d4)]
    return res

def day_present(df, timestamp):
    d1 = datetime.datetime.strptime(timestamp, "%Y-%m-%d")
    d2 = d1 + datetime.timedelta(days=1)
    return df.loc[(df['date'] >= d1) & (df['date'] < d2)].shape[0] == 96

MAX = 20

def filter_datasets(directory, periods):
    datasets = []
    count = 0
    for f in os.listdir(directory):
        if f.endswith(".csv") and len(f) == len("IT001E00000004.csv") and f.startswith("IT") and count < (MAX) :
            try:
                df = load_dataset(os.path.join(directory, f), normalize=True)
                for start, stop in periods:
                    if not (day_present(df, start) and day_present(df, stop)):
                        break
                    nulls = get_period(df, start, stop)['active'].isnull()
                    if sum(nulls) > 10:
                        print("Too many nan", f)
                        break
                else:
                    df.loc[:, 'active'] = df.loc[:, 'active'].fillna(0)
                    datasets.append(df)
                    count += 1
            except Exception as e:
                print("Exception", e)
    return datasets

def get_correlation(datasets, periods):
    res = []
    for p in periods:
        months = np.array([get_period(d, p[0], p[1])['active'] for d in datasets])
        cc = np.corrcoef(months)
        res.append(cc)
    final = sum(res) / float(len(res))
    toprint = [[int(c*100) for c in r] for r in final]
    return pd.DataFrame(toprint)

def cluster(df, threshold=75):
    data = np.triu(df.values, k=1)
    res = np.zeros(df.values.shape[0]) - 1
    for r, row in enumerate(data):
        for c, value in enumerate(row):
            if value >= threshold:
                if res[r] == -1:
                    res[r] = r
                if res[c] == -1:
                    target = res[r]
                else:
                    target = res[c]
                    for i, e in enumerate(res):
                        if e == r:
                            res[i] = target
                res[c] = target
    return res

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, EasterMonday, Easter
from pandas.tseries.offsets import Day, CustomBusinessDay

class ITBusinessCalendar(AbstractHolidayCalendar):
    """ Custom Holiday calendar for France based on
        https://en.wikipedia.org/wiki/Public_holidays_in_France
      - 1 January: New Year's Day
      - Moveable: Easter Monday (Monday after Easter Sunday)
      - 1 May: Labour Day
      - 8 May: Victory in Europe Day
      - Moveable Ascension Day (Thursday, 39 days after Easter Sunday)
      - 14 July: Bastille Day
      - 15 August: Assumption of Mary to Heaven
      - 1 November: All Saints' Day
      - 11 November: Armistice Day
      - 25 December: Christmas Day
    """
    rules = [
        Holiday('New Years Day', month=1, day=1),
        Holiday('Abbbefana', month=1, day=6),
        EasterMonday,
        Holiday('Immacolata concezione', month=12, day=8),
        Holiday('Labour Day', month=5, day=1),
        Holiday('Victory in Europe Day', month=4, day=25),
        Holiday('Assumption of Mary to Heaven', month=8, day=15),
        Holiday('All Saints Day', month=11, day=1),
        Holiday('ARrepubbblica', month=6, day=2),
        Holiday('Christmas Day', month=12, day=25),
        Holiday('Stefanzilla day', month=12, day=26)
    ]



def filter_holidays(df):
    c = ITBusinessCalendar()
    holidays = c.holidays()
    # non elimina le domeniche
    # wdays = df.loc[(df.date.apply(lambda x: x.weekday()) <= 5) & df.date.apply(lambda x: not (x.date() in holidays))]
    wdays = df.loc[df.date.apply(lambda x: not (x.date() in holidays))]
    if "reconstructed" in wdays.columns:
        wdays = wdays.loc[wdays.reconstructed == 0]
    return wdays

def get_bands(df):
    wdays = filter_holidays(df)
    first_band = wdays.loc[((df.date.apply(lambda x: x.hour) < 18) &
       (df.date.apply(lambda x: x.hour) >= 16)) |
       ((df.date.apply(lambda x: x.hour) >= 10) &
       (df.date.apply(lambda x: x.hour) < 12))
      ]

    second_band = wdays.loc[((df.date.apply(lambda x: x.hour) < 4) &
       (df.date.apply(lambda x: x.hour) >= 2))
      ]

    return first_band, second_band


def get_middle_bands(df):
    c = ITBusinessCalendar()
    holidays = c.holidays()
    wdays = df.loc[(df.date.apply(lambda x: x.weekday()) <= 5) & df.date.apply(lambda x: not (x.date() in holidays)) ]

    mid_band = wdays.loc[((df.date.apply(lambda x: x.hour) < 14) &
       (df.date.apply(lambda x: x.hour) >= 13))
      ]

    return mid_band

def get_bands(df):
    first_band = df.loc[((df.date.apply(lambda x: x.hour) < 18) &
       (df.date.apply(lambda x: x.hour) >= 16)) |
       ((df.date.apply(lambda x: x.hour) >= 10) &
       (df.date.apply(lambda x: x.hour) < 12))
      ]

    second_band = df.loc[((df.date.apply(lambda x: x.hour) < 4) &
       (df.date.apply(lambda x: x.hour) >= 2))
      ]

    mid_band = df.loc[((df.date.apply(lambda x: x.hour) < 14) &
       (df.date.apply(lambda x: x.hour) >= 13))
      ]

    return first_band, mid_band, second_band


def entropy(data):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entro = stats.entropy(p_data)  # input probabilities to get the entropy
    return entro

def scan_pods(paths, threshold=0):
    for p in paths:
        ds = load_dataset(p)
        if ds.active.max() <= threshold:
            continue
        yield calculate_levels(ds)

def calculate_levels(ds):
    first_band, second_band = get_bands(ds)
    day_entropy = entropy(first_band.active)
    night_entropy = entropy(second_band.active)
    day_energy = first_band.active.sum()
    night_energy = second_band.active.sum()
    pod = ds["pod"].iloc[0]
    return pod, day_energy, night_energy, day_entropy, night_entropy

"""
PODS = list(filter(lambda x: x.endswith(".csv") and
            x.startswith("IT") and
            "(" not in x and
            "old" not in x.lower() and
            "ex" not in x.lower(), os.listdir(DATA_DIR)))
"""


def compute_ener_entr_ratios(df):
    first_band, second_band = get_bands(df)
    mid_band = get_middle_bands(df)

    first_band['week_index'] = 100 * first_band['date'].dt.year + first_band['date'].dt.week
    mid_band['week_index'] = 100 * mid_band['date'].dt.year + mid_band['date'].dt.week
    second_band['week_index'] = 100 * second_band['date'].dt.year + second_band['date'].dt.week
    first_band_grouped_by_week = first_band.groupby(['week_index'])
    mid_band_grouped_by_week = mid_band.groupby(['week_index'])
    second_band_grouped_by_week = second_band.groupby(['week_index'])

    day_entropies = []
    mid_entropies = []
    night_entropies = []
    day_energies = []
    mid_energies = []
    night_energies = []


    for g in (first_band_grouped_by_week.groups.keys()):
        day_entropy = entropy(first_band_grouped_by_week.get_group(g)['active'])
        night_entropy = entropy(second_band_grouped_by_week.get_group(g)['active'])
        mid_entropy = entropy(mid_band_grouped_by_week.get_group(g)['active'])
        day_entropies.append(day_entropy)
        night_entropies.append(night_entropy)
        mid_entropies.append(mid_entropy)

        day_energy = (first_band_grouped_by_week.get_group(g)['active'].sum())
        night_energy = (second_band_grouped_by_week.get_group(g)['active'].sum())
        mid_energy = (mid_band_grouped_by_week.get_group(g)['active'].sum())

        day_energies.append(day_energy)
        night_energies.append(night_energy)
        mid_energies.append(mid_energy)


    av_day_entropy = np.nanmean(day_entropies)
    av_night_entropy = np.nanmean(night_entropies)
    av_mid_entropy = np.nanmean(mid_entropies)

    av_day_energy = np.nanmean(day_energies)
    av_night_energy = np.nanmean(night_energies)
    av_mid_energy = np.nanmean(mid_energies)



    return av_mid_entropy, av_mid_energy, av_day_entropy, av_night_entropy, av_day_energy, av_night_energy



def compute_var_med_range(df):
    c = ITBusinessCalendar()
    holidays = c.holidays()
    wdays = df.loc[(df.date.apply(lambda x: x.weekday()) < 5) & df.date.apply(lambda x: not (x.date() in holidays))]
    wdays.loc[:, 'week_index'] = 100 * wdays['date'].dt.year + wdays['date'].dt.week
    wdays_grouped_by_week = wdays.groupby(['week_index'])
    groups = wdays_grouped_by_week.groups
    ranges = []


    for g in (groups.keys()):
        active = wdays.loc[groups[g]].active
        en_range = active.max() - active.min()
        ranges.append(en_range)

    sigma = np.nanstd(ranges)
    median = np.nanmedian(ranges)
    max_en = wdays.active.max()
    min_en = wdays.active.min()

    return sigma, median, max_en, min_en

def compute_features(df):
    df = filter_holidays(df)
    df['week_index'] = 100 * df['date'].dt.year + df['date'].dt.week
    first_band, mid_band, second_band = get_bands(df)
    #mid_band = get_middle_bands(df)

    #first_band['week_index'] = 100 * first_band['date'].dt.year + first_band['date'].dt.week
    #mid_band['week_index'] = 100 * mid_band['date'].dt.year + mid_band['date'].dt.week
    #second_band['week_index'] = 100 * second_band['date'].dt.year + second_band['date'].dt.week
    first_band_grouped_by_week = first_band.groupby(['week_index'])
    mid_band_grouped_by_week = mid_band.groupby(['week_index'])
    second_band_grouped_by_week = second_band.groupby(['week_index'])
    wdays_grouped_by_week = df.groupby(['week_index'])
    groups = wdays_grouped_by_week.groups

    day_entropies = []
    mid_entropies = []
    night_entropies = []
    day_energies = []
    mid_energies = []
    night_energies = []
    ranges = []


    for g in (first_band_grouped_by_week.groups.keys()):
        day_entropy = entropy(first_band_grouped_by_week.get_group(g)['active'])
        night_entropy = entropy(second_band_grouped_by_week.get_group(g)['active'])
        mid_entropy = entropy(mid_band_grouped_by_week.get_group(g)['active'])
        day_entropies.append(day_entropy)
        night_entropies.append(night_entropy)
        mid_entropies.append(mid_entropy)

        day_energy = (first_band_grouped_by_week.get_group(g)['active'].sum())
        night_energy = (second_band_grouped_by_week.get_group(g)['active'].sum())
        mid_energy = (mid_band_grouped_by_week.get_group(g)['active'].sum())

        day_energies.append(day_energy)
        night_energies.append(night_energy)
        mid_energies.append(mid_energy)

        active = df.loc[groups[g]].active
        en_range = active.max() - active.min()
        ranges.append(en_range)


    av_day_entropy = np.nanmean(day_entropies)
    av_night_entropy = np.nanmean(night_entropies)
    av_mid_entropy = np.nanmean(mid_entropies)

    av_day_energy = np.nanmean(day_energies)
    av_night_energy = np.nanmean(night_energies)
    av_mid_energy = np.nanmean(mid_energies)

    sigma = np.nanstd(ranges)
    median = np.nanmedian(ranges)
    max_en = df.active.max()
    min_en = df.active.min()



    return av_mid_entropy, av_mid_energy, av_day_entropy, av_night_entropy, av_day_energy, av_night_energy, sigma, median, max_en, min_en

def cut_reconstructed(dataset):
    dataset['data'] = dataset['date'].apply(lambda x: x.date())
    dataset['ora'] = dataset['date'].apply(lambda x: datetime.datetime(2000, 1, 1, x.hour, x.minute))
    pivoted_dataset = dataset.pivot_table('active', 'data', 'ora')

    k = np.array([[-1], [0], [1]])
    # edges = ndimage.convolve(pivoted_dataset.values, filter_fun)
    edges = ndimage.convolve(pivoted_dataset.values, k.T)

    prova = pd.DataFrame(edges, columns=pivoted_dataset.columns)
    prova['data'] = pivoted_dataset.index
    prova = pd.melt(prova, id_vars=['data'], value_name='filtered')
    prova['date'] = prova.apply(lambda x: pd.datetime.combine(x['data'], x['ora'].time()), axis=1)
    prova.set_index('date', inplace=True)
    prova.sort_index(inplace=True)

    grouped = prova.groupby(prova['data'])
    x = pd.DataFrame()
    valid_dates = []
    for g in sorted(grouped.groups):
        if not ((grouped.get_group(g)['filtered'] == 0) | grouped.get_group(g)['filtered'].isnull()).sum() >= 70:
            x = pd.concat([x, dataset[dataset['data'] == g]])
            #valid_dates.append(g)

    #return dataset.loc[dataset.data.apply(lambda x: x in valid_dates)]

    return x

def get_valid_dates(dataset):
    dataset['data'] = dataset['date'].apply(lambda x: x.date())
    dataset['ora'] = dataset['date'].apply(lambda x: datetime.datetime(2000, 1, 1, x.hour, x.minute))
    pivoted_dataset = dataset.pivot_table('active', 'data', 'ora')

    k = np.array([[-1], [0], [1]])
    # edges = ndimage.convolve(pivoted_dataset.values, filter_fun)
    edges = ndimage.convolve(pivoted_dataset.values, k.T)

    prova = pd.DataFrame(edges, columns=pivoted_dataset.columns)
    prova['data'] = pivoted_dataset.index
    prova = pd.melt(prova, id_vars=['data'], value_name='filtered')
    prova['date'] = prova.apply(lambda x: pd.datetime.combine(x['data'], x['ora'].time()), axis=1)
    prova.set_index('date', inplace=True)
    prova.sort_index(inplace=True)

    grouped = prova.groupby(prova['data'])
    #x = pd.DataFrame()
    valid_dates = []
    for g in sorted(grouped.groups):
        if not ((grouped.get_group(g)['filtered'] == 0) | grouped.get_group(g)['filtered'].isnull()).sum() >= 70:
            #x = pd.concat([x, dataset[dataset['data'] == g]])
            valid_dates.append(g)

    return valid_dates

    #return x


###########################################################################

#TODO: Hybrid prop in models for hour
def filter_by_hours(df, o, c, col="hour"):
    return df[(df[col] >= o) & (df[col] < c)]


def get_merged(pod_id, spans, cut_date=None, use_cache=True, temperature_col="temperatura", ):

    data_dir = get_setting("DATA_DIR")
    meteo_dir = get_setting("METEO_DIR")
    cache_dir = os.path.join(get_setting("CACHE_DIR"), "merged")

    ensure_dir(cache_dir)
    merged_filepath = os.path.join(cache_dir, "merged_%s_%s_%s.csv" % (pod_id, temperature_col, cut_date))
    df_w_d_filepath = os.path.join(cache_dir, "df_w_d_%s_%s_%s.csv" % (pod_id, temperature_col, cut_date))

    if use_cache and os.path.exists(merged_filepath) and os.path.exists(df_w_d_filepath):
        merged = pd.read_csv(merged_filepath, index_col=0, parse_dates=["data"])
        print("loaded %s from cache" % merged_filepath)
        df_w_d = pd.read_csv(df_w_d_filepath, index_col=0, parse_dates=["date","data", "ora"])
        print("loaded %s from cache" % df_w_d_filepath)
        return merged, df_w_d


    pod_info = get_pod_info(pod_id)
    LAST_DAY = 6

    df_single_pod = load_dataset(os.path.join(data_dir, "%s.csv" % pod_id), cut=True, use_cache=True)

    if cut_date is not None:
        df_single_pod = df_single_pod[df_single_pod['date'] < pd.to_datetime(cut_date)]

    df_w_d = filter_holidays(df_single_pod)
    df_w_d.loc[:, 'data'] = pd.to_datetime(df_w_d['data'])
    df_w_d.loc[:, 'active'] = df_w_d['active'] * 4
    df_w_d.loc[:, 'tooltip'] = df_w_d['data'].apply(lambda x: x.strftime("%Y-%m-%d") + ' ' + calendar.day_name[x.weekday()])

    df_w_d["year"] = df_w_d["date"].dt.year
    df_w_d["month"] = df_w_d["date"].dt.month
    df_w_d["hour"] = df_w_d["date"].dt.hour
    df_w_d["weekday"] = df_w_d["date"].dt.weekday

    first_band = df_w_d[df_w_d["weekday"] <= LAST_DAY]
    dfs = []
    for o,c in spans:
        dfs.append(filter_by_hours(first_band, o, c, "hour"))
    first_band = pd.concat(dfs).sort_values(by="date")

    #
    # first_band = df_w_d.loc[(((df_w_d["hour"] < 18) & (df_w_d["hour"] >= 16)) |
    #    ((df_w_d["hour"] >= 10) & (df_w_d["hour"] < 12))) & (df_w_d["weekday"] <= LAST_DAY)
    #   ]
    # print(first_band.shape)


    av_energy = first_band.groupby("data")[["active"]].mean().rename(columns={"active": "daily_av_energy"})

    #
    # name, description, province = locations.loc[df_w_d['pod'].iloc[0], ['company', 'description', 'province_shortname']].values
    # file_name = str(province + '-' + description + '.csv')
    # meteo_file = list(filter(lambda x: x.endswith(".csv") and
    #                          file_name in x, os.listdir(meteo_dir)))
    #
    # print(file_name)
    # print(meteo_file)
    # assert False
    # meteo_data = pd.read_csv(os.path.join(meteo_dir, meteo_file[0]), sep=",", parse_dates=['data', 'ora'])
    # meteo_data["hour"] = meteo_data["ora"].dt.hour

    import meteo
    meteo_data = meteo.load_meteo_data(pod_id)

    dfs = []
    for o,c in spans:
        dfs.append(filter_by_hours(meteo_data, o, c, "hour"))
    first_band_meteo = pd.concat(dfs).sort_values(by="data")
    # print("fbm", first_band_meteo.shape)
    # first_band_meteo = meteo_data.loc[((meteo_data['ora'].apply(lambda x: x.hour) <= 18) &
    #                                    (meteo_data['ora'].apply(lambda x: x.hour) >= 16)) |
    #                                   ((meteo_data['ora'].apply(lambda x: x.hour) >= 10) &
    #                                    (meteo_data['ora'].apply(lambda x: x.hour) <= 12))
    #                                  ]
    # print("fbm", first_band_meteo.shape)
    first_band_meteo['temperatura'] = first_band_meteo[temperature_col]


    av_meteo = pd.DataFrame(first_band_meteo.groupby('data')['temperatura'].mean())
    merged = pd.merge(av_meteo, av_energy, left_index=True, right_index=True)

    merged.reset_index(inplace=True)
    merged.rename(columns={'index':'data'}, inplace=True)
    merged['date_index'] = merged.index
    merged.loc[:, 'tooltip'] = merged['data'].apply(lambda x: x.strftime("%Y-%m-%d") + ' ' + calendar.day_name[x.weekday()])
    merged.loc[:, 'colorbar'] = merged['data'].apply(lambda x: x.strftime("%Y-%m-%d"))

    merged["pod"] = pod_id
    merged['location'] = pod_info["description"]
    merged['province'] = pod_info["province_shortname"]
    merged['company'] = pod_info["company"]

    merged['rounded_temp'] = merged['temperatura'].apply(lambda x: int(round(x)))

    merged["month"] = merged["data"].dt.month
    merged["weekday"] = merged["data"].dt.weekday
    merged["year"] = merged["data"].dt.year

    merged.to_csv(merged_filepath)
    df_w_d.to_csv(df_w_d_filepath)

    return merged, df_w_d

#########################################################################


COLORING = {
    "month": {
        "colormap": {1: 'dodgerblue', 2: 'blue', 3: 'slateblue', 12: 'cyan',
                    5: 'mediumvioletred', 4:'fuchsia', 11:'green', 6:'red', 10: 'lime',
                    7: 'darkorange', 9:'yellow', 8: 'darkgoldenrod'},
        "labels": {1: 'Jan.', 2: 'Feb.', 3: 'Mar.', 12: 'Dec.',
                5: 'May', 4:'Apr.', 11:'Nov.', 6:'June', 10: 'Oct.',
                7: 'July', 9:'Sept.', 8: 'Aug.'}
             },
    "year": {
        "colormap": {
        2011: 'dodgerblue',
        2012: 'blue',
        2013: 'slateblue',
        #12: 'cyan',
                    2015: 'mediumvioletred', 2014:'fuchsia', 11:'green', 2016:'red', 2010: 'lime',
                    2017: 'darkorange', 2019:'yellow', 2018: 'darkgoldenrod'},
        "labels": {1: 'Jan.', 2: 'Feb.', 3: 'Mar.', 12: 'Dec.',
                5: 'May', 4:'Apr.', 11:'Nov.', 6:'June', 10: 'Oct.',
                7: 'July', 9:'Sept.', 8: 'Aug.'}
             },
    "weekday": {
        "colormap": {1: 'dodgerblue', 2: 'blue', 3: 'slateblue', 0: 'cyan',
                    5: 'mediumvioletred', 4:'fuchsia', 11:'green', 6:'red', 10: 'lime',
                    7: 'darkorange', 9:'yellow', 8: 'darkgoldenrod'},
        "labels": {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun",}
    }
}

def get_coloring(name):
    if name == "year":
        colormap = COLORING["year"]["colormap"]
        labels = dict([(y,str(y)) for y in range(2000, 2018)])
    elif name in COLORING:
        colormap = COLORING[name]["colormap"]
        labels = COLORING[name]["labels"]
    else:
        raise Exception("No colormap defined for %s" % name)

    return colormap, labels

from bokeh.models import Legend


def energy_temp(merged, averages, palette, bins=16, legend_col="month", confront=None, extra_title=None, x_range=None, y_range=None, comparisons=None):

    MUTED_ALPHA = 0.1

    colormap, labels = get_coloring(legend_col)

    merged = merged.sort_values(by=legend_col)

    colors = [colormap[x] for x in merged[legend_col]]
    legend_label=[labels[x] for x in merged[legend_col]]
    merged['legend_label'] = legend_label

    pod = merged['pod'].values[0]
    if pod.endswith(".csv"):
        pod = pod[:-4]

    title = '{} {} {} {}'.format(merged['company'].values[0],
                                          merged['location'].values[0],
                                          merged['province'].values[0],
                                          pod)
    if extra_title:
        title = "%s %s" % (title, extra_title)

    p = figure(title=title,
               x_axis_location='below', plot_width=900, plot_height=600,
               toolbar_location='above', tools=TOOLS)



    if x_range:
        p.x_range = Range1d(*x_range)
    if y_range:
        p.y_range = Range1d(*y_range)

    legend_items = {}

    for label, dfg in merged.groupby(by=legend_col):
        color_l = colormap[label]
        source = ColumnDataSource(dfg)
        c = p.circle(source=source, x="temperatura", y="daily_av_energy",
             color= color_l,
             muted_color=color_l,
             muted_alpha=MUTED_ALPHA,
             line_alpha=0.6,
             fill_alpha=0.6, size=10,
             #legend='legend_label'
             )
        legend_items[labels[label]] = [c]

    # source = ColumnDataSource(merged)
    # p.circle(source=source, x="temperatura", y="daily_av_energy",
    #          color= colors,
    #          muted_color="gray",
    #          muted_alpha=0.2,
    #          line_alpha=0,
    #          fill_alpha=1, size=10, legend='legend_label')

    p.select_one(HoverTool).tooltips = [
            ('Date', '@tooltip'),
            ('Av. Energy', '@daily_av_energy'),
            ('Ext. Temp.', '@temperatura')
            ]

    source_prova = ColumnDataSource(averages)

    #legend_labels = defaultdict(list)


    x = averages["temp"].values
    upper = averages["energy_mean"].values
    lower = averages["mean-std"].values
    band_x = np.append(x, x[::-1])
    band_y = np.append(lower, upper[::-1])

    # p.patch(band_x, band_y, color="blue", fill_alpha=0.3, legend="all days", muted_color="blue", muted_alpha=0.2)
    # p.line(source=source_prova, x='temp', y='energy_mean', color='red', line_width=4, legend="all days", muted_color="blue", muted_alpha=0.2)
    # p.line(source=source_prova, x='temp', y='mean-std', color='blue', line_width=4, legend="all days", muted_color="blue", muted_alpha=0.2)

    pb = p.patch(band_x, band_y, color="blue", fill_alpha=0.3, muted_color="blue", muted_alpha=0.1)
    pl1 = p.line(source=source_prova, x='temp', y='energy_mean', color='blue', line_width=4, muted_color="blue", muted_alpha=0.1)
    pl2  = p.line(source=source_prova, x='temp', y='mean-std', color='blue', line_width=4, muted_color="blue", muted_alpha=0.1)
    legend_items["All Days"] = [pb, pl1, pl2]

    if not confront is None:
        source = ColumnDataSource(confront)
        x = confront["temp"].values
        upper = confront["energy_mean"].values
        lower = confront["mean-std"].values
        band_x = np.append(x, x[::-1])
        band_y = np.append(lower, upper[::-1])
        p.patch(band_x, band_y, color="gray", fill_alpha=0.3)

        p.line(source=source, x='temp', y='energy_mean', color='gray', line_width=2)
        p.line(source=source, x='temp', y='mean-std', color='gray', line_width=2)

    if not comparisons is None:
        comp_colors = ["green", "orange", "red"]
        for i, (label, comp_data, _) in enumerate(comparisons):

            color = comp_colors[i]
            source = ColumnDataSource(comp_data)
            x = comp_data["temp"].values
            upper = comp_data["energy_mean"].values
            lower = comp_data["mean-std"].values
            band_x = np.append(x, x[::-1])
            band_y = np.append(lower, upper[::-1])
            patch = p.patch(band_x, band_y, color=color, fill_alpha=0.3, #legend=label,
            muted_color=color, muted_alpha=MUTED_ALPHA)

            l1 = p.line(source=source, x='temp', y='energy_mean', color=color, line_width=4,
            #legend=label,
            muted_color=color, muted_alpha=MUTED_ALPHA)
            l2 = p.line(source=source, x='temp', y='mean-std', color=color, line_width=4,
            #legend=label,
            muted_color=color, muted_alpha=MUTED_ALPHA)
            legend_items[label] = [patch, l1, l2]



    p.xaxis.axis_label = 'TEMPERATURE'
    p.yaxis.axis_label = 'DAILY AVERAGE ENERGY'

    p.legend.click_policy="mute"

    legend = Legend(items=list(legend_items.items()), location = (0,0))
    legend.click_policy="mute"
    p.add_layout(legend, "right")
    return p




################################################################################

def filter_period(df, period=None, col="date"):

    if period is None:
        return df

    if not period[0] is None:
        d1 = datetime.datetime.strptime(period[0], "%Y-%m-%d")

        df = df[df[col] >= d1]

    if not period[1] is None:
        d2 = datetime.datetime.strptime(period[1], "%Y-%m-%d")

        df = df[df[col] <= d2]
    return df


def filter_months(df, months="all", col="month", date_col="date"):

    if months == "all":
        return df

    if not col in df.columns:
        df[col] = df[date_col].dt.month

    return df[df[col].isin(months)]

def filter_weekdays(df, weekday="all", col="weekday", date_col="date"):

    if weekday == "all":
        return df

    if not col in df.columns:
        df[col] = df[date_col].dt.weekday

    return df[df[col].isin(weekday)]

################################################################################


def find_fences(values, factor=1.5):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    return q1 - factor*iqr, q3 + factor*iqr

def remove_outliers(values, factor=1.5):
    l, h = find_fences(values, factor=factor)
    return values[(l <= values) & (values <= h)]

################################################################################

def average_white_band(merged):
    group_temp = merged.groupby('rounded_temp')
    temp = []
    energy_mean = []
    energy_std = []
    for g in group_temp.groups:
        temp.append(g)
        energy_mean.append(group_temp.get_group(g)['daily_av_energy'].mean())
        energy_std.append(group_temp.get_group(g)['daily_av_energy'].std())
    temp = np.array(temp)
    energy_mean = np.array(energy_mean)
    energy_std = np.array(energy_std)

    tmp = pd.DataFrame(np.vstack([temp, energy_mean, energy_std]).T, columns=['temp', 'energy_mean', 'energy_std'])
    tmp.fillna(0, inplace=True)
    tmp['mean-std'] = tmp['energy_mean'] - tmp['energy_std']
    return tmp

from scipy.ndimage.interpolation import shift
def remove_hills_positive(values):
    while True:
        diff = np.diff(values)
        if (diff < 0).sum() == 0:
            break
        for i, value in enumerate(diff):
            if value < 0:
                values[i] = values[i+1]

    return values

def do_remove_hills(values):
    if len(values) < 3:
        return values


    values = np.array(values).copy()

    min_idx = np.array(values).argmin()

    values[min_idx:] = remove_hills_positive(values[min_idx:])
    values[:min_idx+1] = remove_hills_positive(values[:min_idx+1][::-1])[::-1]

    return values

from sklearn.linear_model import LinearRegression

def average_window(df, method="fixed", avg_method="mean", min_sample=10, max_window_size=15, window_size=3, filter_fn=None, smooth_fn=None, sigma=1, remove_hills=True, lin_reg_from=None):

    avg_fns = {
        "mean": lambda x: x.mean(),
        "median": lambda x: np.median(x),
    }

    avg_fn = avg_fns[avg_method]


    METHODS = ("fixed", "dynamic")
    if not method in METHODS:
        raise Exception("Unknown method: %s\n available methods: %s" % (method, METHODS) )

    temps = sorted(df["rounded_temp"].unique())


    if not lin_reg_from is None:
        max_temp = df["rounded_temp"].max()
        temps = [t for t in temps if t < lin_reg_from]

    assert window_size > 0, "min_sample must be greater than 0"

    if method == "fixed":
        if window_size % 2 == 0: #
            offset_low = int((window_size - 1)/2)
            offset_high = offset_low + 1
        else:
            offset_low = int((window_size - 1)/2)
            offset_high = offset_low

    energy_values = []
    energy_std = []
    sample_count = []
    win_size = []
    for temp in temps:

        if method == "fixed":
            l = temp - offset_low
            h = temp + offset_high
            dft = df[(df["rounded_temp"] >= l) & (df["rounded_temp"] <= h)]
            values = dft["daily_av_energy"].values

            if filter_fn:
                values = filter_fn(values)

        elif method == "dynamic":
            offset_low = 0
            offset_high = 0
            for offset in range(max_window_size):
                if offset > 0 and offset % 2 == 0:
                    offset_high += 1
                elif offset % 2 == 1:
                    offset_low += 1

                l = temp - offset_low
                h = temp + offset_high
                dft = df[(df["rounded_temp"] >= l) & (df["rounded_temp"] <= h)]
                values = dft["daily_av_energy"].values

                if filter_fn:
                    values = filter_fn(values)

                if len(values) >= min_sample:
                    break



        energy_values.append(avg_fn(values))
        energy_std.append(values.std())
        sample_count.append(len(values))
        win_size.append(1 + offset_low + offset_high)

    lin_reg_formula = None
    if not lin_reg_from is None:
        X = df[df["rounded_temp"] >= lin_reg_from][["temperatura", "daily_av_energy"]].as_matrix()
        if X.shape[0] > 0:

            values = X[:,1].reshape(-1,1)
            lr = LinearRegression().fit(X[:,0].reshape(-1,1), values)
            res = lr.predict(np.array(range(lin_reg_from, max_temp+1)).reshape(-1,1))
            # print(dir(lr))
            # print(lr.coef_, lr.intercept_)
            if lr.intercept_ < 0:
                s_intercept = "- %.3f" % (abs(lr.intercept_))
            else:
                s_intercept = "+ %.3f" % (lr.intercept_)

            lin_reg_formula = "y = %.3fx %s" % (lr.coef_, s_intercept)
            std = values.std()
            n_samples = len(values)

            for i,value in enumerate(res):
                temps.append(lin_reg_from+i)
                energy_values.append(value[0])
                #energy_std.append(energy_std[-1])
                energy_std.append(std)
                sample_count.append(n_samples)
                win_size.append(-1)

    temp = np.array(temps)
    energy_values = np.array(energy_values)
    energy_std = np.array(energy_std)

    if remove_hills:
        energy_values = do_remove_hills(energy_values)

    sample_count = np.array(sample_count)
    win_size = np.array(win_size)

    tmp = pd.DataFrame(np.vstack([temp, energy_values, energy_std, sample_count, win_size]).T, columns=['temp', 'energy_mean', 'energy_std', "samples", "win_size"])
    tmp.fillna(0, inplace=True)

    if smooth_fn:
        tmp = smooth_fn(tmp)

    tmp['mean-std'] = tmp['energy_mean'] - sigma*tmp['energy_std']#.mean()

    return tmp, lin_reg_formula

class FilterFunc(object):

    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, values):
        return remove_outliers(values, factor=self.factor)

class ComputeAverages(object):

    def __init__(self, method="fixed", avg_method="mean", min_sample=10, max_window_size=15, window_size=1, filter_fn=None, smooth_fn=None, sigma=1, remove_hills=True, lin_reg_from=None):
        self.method = method
        self.avg_method = avg_method
        self.min_sample = min_sample
        self.max_window_size = max_window_size
        self.window_size = window_size
        self.filter_fn = filter_fn
        self.smooth_fn = smooth_fn
        self.sigma = sigma
        self.remove_hills = remove_hills
        self.lin_reg_from = lin_reg_from

    def __call__(self, df):
        return average_window(df, self.method, self.avg_method, self.min_sample, self.max_window_size, self.window_size, self.filter_fn, self.smooth_fn, self.sigma, self.remove_hills, self.lin_reg_from)


###########################################################################

from bokeh.models import ContinuousTicker
from bokeh.models import FixedTicker
from bokeh.palettes import RdYlGn11
def plot_temp_disp(df, x_col, y_col, colors, low, high):
    colormapper = LinearColorMapper(colors, low=low, high=high)
    #colormapper = LinearColorMapper(palette=RdYlGn11, low=0, high=100)
    ticker = FixedTicker(ticks=[5,15,25,35,45,55])
    formatter = FuncTickFormatter(code="""
    return tick.toString() + ' °C';
    """)
    colorbar = ColorBar(color_mapper=colormapper, location=(0,0),
                        major_tick_out=35, major_tick_in=0, major_label_text_align='right',
                        #ticker=ticker,
                        formatter=formatter,
                        label_standoff=2, major_label_text_font_size='10pt')

    fig = figure(x_axis_label=x_col, y_axis_label=y_col, plot_width=460, plot_height=400)
    fig.scatter(df[x_col], df[y_col], marker="circle", size=12, fill_color=colors)
    fig.add_layout(colorbar, "right")
    return fig

def plot_energy_dispersity(merged, avg_fn, pairs, years="all"):

    merged["year"] = merged["data"].dt.year
    merged = merged.sort_values(by="year")

    if not years == "all":
        merged = merged[merged["year"].isin(years)]

    avg, _ = avg_fn(merged)

    labels = avg["temp"].astype(int)
    colors = normalize(labels) #["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*get_cmap("jet")(mpl.colors.Normalize()(labels))]

    min_temp = labels.min()
    max_temp = labels.max()

    plots = []
    for row in pairs:
        row_plots = []

        for x,y in row:
            plot = plot_temp_disp(avg, x,y, colors, min_temp, max_temp)
            #plot.add_layout(colorbar, "right")
            row_plots.append(plot)
        plots.append(row_plots)

    grid = gridplot(plots)
    op  = []
    op.append(components(grid))
    return op

###########################################################################


def gen_filters(years="all", months="all", days_of_week="all", periods=None):
    return {
        "years": years,
        "months": months,
        "days_of_week": days_of_week,
        "periods": periods,
    }
def apply_filters(df, filters, date_col="data"):
    if isinstance(filters, dict):
        for key, values in filters.items():
            if not values in ("all", "whole", "complete"):
                if key == "years": key = "year"
                elif key == "months": key = "month"
                elif key == "days_of_week":
                    key = "weekday"
                    values = [v-1 for v in values]

                elif key == "periods":
                    df = filter_periods(df, values, col=date_col)
                    continue

                df = df[df[key].isin(values)]
    return df

WEEKDAY_FILTERS = {"days_of_week": [1,2,3,4,5]}
SAT_FILTERS = {"days_of_week": [6]}
SUN_FILTERS = {"days_of_week": [7]}

def plot_consumption_curve(merged, palette, bins, avg_fn, filters=None, plot_each_year=True, compare_days=None, print_table=False, print_std=False):
    ops = []

    merged = apply_filters(merged, filters)

    avg, lin_reg_formula = avg_fn(merged)

    x_range = (avg["temp"].min(), avg["temp"].max())
    y_range = (merged["daily_av_energy"].min(), merged["daily_av_energy"].max())


    comps = []
    if compare_days is None:
        comps = []
    else:
        if "working" in compare_days:
            comps.append(("Working Days", *avg_fn(apply_filters(merged, WEEKDAY_FILTERS))))
        if "saturdays" in compare_days:
            comps.append(("Saturdays", *avg_fn(apply_filters(merged, SAT_FILTERS))))
        if "sundays" in compare_days:
            comps.append(("Sundays", *avg_fn(apply_filters(merged, SUN_FILTERS))))


    if print_table:
        print("\nAll Days")
        print(avg)
        for comp_day, comp_df, _ in comps:
            print("\n%s" % (comp_day))
            print(comp_df)

    if print_std:
        #std_data = [("All Days", avg["energy_mean"].mean(), avg["energy_std"].mean())]
        std_data = []
        for comp_day, comp_df,_ in comps:

            std_data.append((comp_day, comp_df["energy_mean"].mean(), comp_df["energy_std"].mean()))

        print("All years")
        print(pd.DataFrame(std_data, columns=["selection", "mean energy_mean", "mean energy_std"]))

    print_lin_reg = False
    if print_lin_reg:
        print()
        print("Linear Regression formulas for ", len(comps))
        for comp_day, _, lin_reg_formula in comps:
            print("%15s  %s" % (comp_day, lin_reg_formula))

    ops.append(components(energy_temp(merged, avg, palette, bins, legend_col="year", x_range=x_range, y_range=y_range, comparisons=comps)))

    if plot_each_year:
        for year, dfyear in merged.groupby(by="year"):
            avg_year, _ = avg_fn(dfyear)

            if print_table:
                print("\nYear %d: All Days" % year)
                print(avg_year)

            comps = []
            if compare_days is None:
                comps = []
            else:
                if "working" in compare_days:
                    comps.append(("Working Days", *avg_fn(apply_filters(dfyear, WEEKDAY_FILTERS))))
                if "saturdays" in compare_days:
                    comps.append(("Saturdays", *avg_fn(apply_filters(dfyear, SAT_FILTERS))))
                if "sundays" in compare_days:
                    comps.append(("Sundays", *avg_fn(apply_filters(dfyear, SUN_FILTERS))))

            if print_table:
                for comp_day, comp_df, _ in comps:
                    print("\nYear %d: %s" % (year, comp_day))
                    print(comp_df)

            if print_std:
                #std_data = [("All Days", avg["energy_mean"].mean(), avg["energy_std"].mean())]
                std_data = []
                for comp_day, comp_df, _ in comps:

                    std_data.append((comp_day, comp_df["energy_mean"].mean(), comp_df["energy_std"].mean()))
                print("Year %d" % year)
                print(pd.DataFrame(std_data, columns=["selection", "mean energy_mean", "mean energy_std"]))

            if print_lin_reg:
                print()
                print("Linear Regression formulas")
                for comp_day, _, lin_reg_formula in comps:
                    print("%15s  %s" % (comp_day, lin_reg_formula))

            p = energy_temp(dfyear, avg_year, palette, bins, legend_col="month", extra_title=", year %d" % year, x_range=x_range, y_range=y_range, comparisons=comps)
            ops.append(components(p))
    return ops

################################################################################

def gen_plots(merged, averages, palette, bins, smooth_fn=None, legend_col="month", confront=None, extra_title=None):

    if smooth_fn:
        n_averages = smooth_fn(averages)
        if not confront is None:
            confront = smooth_fn(confront)
    else:
        n_averages = averages

    p = energy_temp(merged, n_averages, palette, bins, legend_col=legend_col, confront=confront, extra_title=extra_title)

    return p

def plot_energy(df, palette, bins, smooth_fn, months="all", weekdays="all", legend_col="month", averages=None, confront=None, extra_title=None, x_limits=None):

    avg_fun = average_white_band

    if isinstance(confront, bool) and confront == True:
        confront = avg_fun(df)

    if not months == "all":
        if isinstance(months, int):
            months = [months]
        df = df[df["month"].isin(months)]

    if not weekdays == "all":
        if isinstance(weekdays, int):
            weekdays = [weekdays]
        df = df[df["weekday"].isin(weekdays)]

    if df.shape[0] == 0:
        print("WARNING: no data for given period")
        return

    if averages is None:
        averages = avg_fun(df)

    # p = gen_plots(df, averages, palette, bins, smooth_fn, legend_col=legend_col, confront=confront, extra_title=extra_title)
    p = energy_temp(df, averages, palette, bins, legend_col=legend_col, confront=confront, extra_title=extra_title)

    show(p)


################################################################################
from scipy.ndimage import gaussian_filter1d

def gaussian_smooth(averages):
    nuovo_averages = averages.copy()
    nuovo_averages['energy_mean'] = gaussian_filter1d(averages['energy_mean'], sigma=1)
    nuovo_averages['energy_std'] = gaussian_filter1d(averages['energy_std'], sigma=1)
    nuovo_averages['mean-std'] = nuovo_averages['energy_mean'] - nuovo_averages['energy_std']
    return nuovo_averages

from scipy.signal import savgol_filter

def savgol(averages):
    nuovo_averages = averages.copy()
    window_length = 31
    poly_order = 2

    energy_mean = averages['energy_mean']
    if len(energy_mean) < window_length:
        window_length = len(energy_mean)
        if window_length % 2 == 0:
            window_length -= 1

    nuovo_averages['energy_mean'] = savgol_filter(energy_mean, window_length, poly_order)
    nuovo_averages['energy_std'] = savgol_filter(averages['energy_std'], window_length, poly_order)
    nuovo_averages['mean-std'] = nuovo_averages['energy_mean'] - nuovo_averages['energy_std']
    return nuovo_averages



################################################################################



def daily_temp(merged, averaged):
    tmp = {}
    for el in averaged.iterrows():
        a = merged[(merged['rounded_temp'] == el[1]['temp']) &
                   (merged['daily_av_energy'] <= el[1]['energy_mean']) &
                   (merged['daily_av_energy'] >= el[1]['mean-std'])]
        if a.empty:
            a = merged[(merged['rounded_temp'] == el[1]['temp']) &
                       (merged['daily_av_energy'] <= el[1]['energy_mean'] + el[1]['energy_std'])]
        if a.empty:
            a = merged[(merged['rounded_temp'] == el[1]['temp'])]
        tmp[el[1]['temp']] = a
    return tmp

def average_consumption(similar_days, df):
    curve_temp = {}
    plots = []
    for k in similar_days:
        similar_means = []
        p = figure(title='', x_axis_type='datetime', plot_width=900, plot_height=600,
                   toolbar_location='above', tools=TOOLS)
        for d in similar_days[k]['data'].values:
            tmp = df[df['data'] == d]
            similar_means.append(tmp['active'].values)
            #p.line(tmp['ora'], tmp['active'])
            p.line(source=ColumnDataSource(tmp), x='ora', y='active')
        similar_means = pd.DataFrame(np.array(similar_means).T)
        #similar_means['mean'] = similar_means.mean(axis=1)
        similar_means['mean'] = similar_means.median(axis=1)
        similar_means['ora'] = tmp['ora'].values
        p.line(similar_means['ora'], similar_means['mean'],color='red', line_width=4)
        p.select_one(HoverTool).tooltips = [
            ('Date', '@tooltip'),
            ('Energy', '@active'),
            ]
        p.title.text = 'Temperature: {}'.format(int(k))
        plots.append([p])
        curve_temp[k] = similar_means
    return plots, curve_temp

def reconstruct(df, merged, average_func=None):

    if average_func is None:
        average_func = average_white_band

    averages, _ = average_func(merged)

    #averages = average_window(merged, window_size=3, filter_fn=gaussian_smooth)
    similar_days = daily_temp(merged, averages)

    _, curve_temp = average_consumption(similar_days, df)

    data = []
    grouped = df.groupby('data')
    for g in sorted(grouped.groups):
        tmp = grouped.get_group(g)
        date = tmp['data'].iloc[0]
        try:
            rounded_temp = merged[merged['data'] == date]['rounded_temp'].values[0]
            active_mean = curve_temp[rounded_temp]['mean'].values
        except IndexError:
            active_mean = tmp['active'].values

        start_date = date.strftime('%Y-%m-%d') + ' 00:00'
        end_date = date.strftime('%Y-%m-%d') + ' 23:45'
        index_date = pd.date_range(start_date, end_date, freq='15T')
        data.append(pd.DataFrame({'active': active_mean, 'date': index_date}))

    reconstructed = pd.concat(data)
    reconstructed['pod'] = df['pod'].values
    reconstructed['anomaly_level'] = df['anomaly_level'].values
    return reconstructed

def reconstruct_by_weekday(df, merged, average_func=None):

    if average_func is None:
        average_func = average_white_band

    day_groups = [[0,1,2,3,4], [5], [6]]

    results = []
    for year, dfyear in df.groupby(by="year"):

        merged_year = merged[merged["year"] == year]

        for days in day_groups:
            res = reconstruct(
                dfyear[dfyear["weekday"].isin(days)],
                merged_year[merged_year["weekday"].isin(days)],
                average_func,
            )
            results.append(res)

    df_rec = pd.concat(results).sort_values(by="date")

    df_rec["year"] = df_rec["date"].dt.year
    df_rec["month"] = df_rec["date"].dt.month
    df_rec["weekday"] = df_rec["date"].dt.weekday

    return df_rec


################################################################################


def calculate_gain(df, df_ideal, by="month"):
    pod = df.iloc[0]["pod"]
    columns = {"active_x": "active_real", "active_y": "active_ideal"}

    df_m = df.merge(df_ideal, left_on="date", right_on="date").rename(columns=columns)

    df_m["gain"] = df_m["active_real"] - df_m["active_ideal"]
    mask = df_m["gain"] < 0
    df_m["gain"][mask] = 0
    df_m["active_real"][mask] = df_m["active_ideal"][mask]


    if "data" in df_m.columns:
        date_col = "data"
    elif "data_x" in df_m.columns:
        date_col = "data_x"
    elif "data_y" in df_m.columns:
        date_col = "data_y"
    else:
        raise Exception("can't find valid date column.\ncolumns: %s" % df_m.columns)

    data = []
    if by == "month":
        df_m["year"] = df_m[date_col].dt.year

        month_col = "month"
        if not month_col in df_m.columns:
            month_col = "month_x"

        for (year, month), dfg in df_m.groupby(by=["year", month_col]):
            data.append((pod, dfg["data_x"].iloc[0], int(year), month, dfg["active_real"].sum(), dfg["active_ideal"].sum()))

        df_gain = pd.DataFrame(data, columns=["pod", "date", "year", "month", "active_real", "active_ideal"])
        df_gain["label"] = df_gain["date"].apply(lambda x: "%4d-%02d" % (x.year, x.month))
        df_gain = df_gain.sort_values(by="label")

    elif by == "day of week":
        df_m["day of week"] = df_m[date_col].dt.weekday

        for weekday, dfg in df_m.groupby(by=["day of week"]):
            data.append((pod, weekday, dfg["active_real"].sum(), dfg["active_ideal"].sum()))

        df_gain = pd.DataFrame(data, columns=["pod", "day of week", "active_real", "active_ideal"])
        df_gain["label"] = df_gain["day of week"].apply(lambda x: calendar.day_abbr[int(x)])
        df_gain = df_gain.sort_values(by="day of week")

    elif by == "year":
        df_m["year"] = df_m[date_col].dt.year

        for year, dfg in df_m.groupby(by="year"):
            data.append((pod, year, dfg["active_real"].sum(), dfg["active_ideal"].sum()))

        df_gain = pd.DataFrame(data, columns=["pod", "year", "active_real", "active_ideal"])
        df_gain["label"] = df_gain["year"].apply(lambda x: str(x))
        df_gain = df_gain.sort_values(by="label")

    else:
        raise Exception("unknown value by=%s  \nValid values: month, day of week" % by)

    df_gain["gain"] = df_gain["active_real"] - df_gain["active_ideal"]
    mask = df_gain["gain"] < 0
    df_gain["gain"][mask] = 0
    df_gain["active_real"][mask] = df_gain["active_ideal"][mask]
    df_gain["gain_percentage"] = (df_gain["gain"] / df_gain["active_real"]) * 100
    df_gain["ideal_percentage"] = (df_gain["active_ideal"] / df_gain["active_real"]) * 100

    return df_gain

#from bkcharts.operations import blend
#from bkcharts.attributes import cat, color
#from bokeh.charts import Bar
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
from bokeh.models import LinearAxis, Range1d, NumeralTickFormatter

def filter_periods(df, periods, col="date"):
    if periods is None or periods == "whole":
        return df
    else:
        df_all = [filter_period(df, period, col=col) for period in periods]
        return pd.concat(df_all)


def plot_saving(df, df_ideal, by="month", periods=None, months="all", weekdays="all", energy2euro=0.17):

    plot_height = 500
    plot_width = 750

    df = filter_months(df, months)
    df = filter_weekdays(df, weekdays)
    df_ideal = filter_months(df_ideal, months)
    df_ideal = filter_weekdays(df_ideal, months)

    if not periods is None:
        df = filter_periods(df, periods)
        df_ideal = filter_periods(df_ideal, periods)

    period_start = df["date"].min().strftime("%Y-%m-%d")
    period_end = df["date"].max().strftime("%Y-%m-%d")

    df_gain = calculate_gain(df, df_ideal, by=by)

    source = df_gain[["label", "gain", "active_ideal", "active_real", "gain_percentage"]]
    source["tt_gain"] = source["gain"].astype(int)
    source["tt_gain_euro"] = (source["gain"] * energy2euro).astype(int)
    source["tt_gain_percentage"] = source["gain_percentage"].apply(lambda x: "%.1f%%" % x)


    #source["Gain"] = source["gain"].astype(int)
    #source["Ideal"] = source["active_ideal"].astype(int)
    #source = ColumnDataSource(source)


    tooltips = [
        (by, "@label"),
        ("Saving %", "@tt_gain_percentage"),
        ("Saving kWh", "@tt_gain"),
        ("Saving €", "@tt_gain_euro"),
        ]

    bar = figure(title="Ideal Consumption and Potential Saving",
        x_range=source["label"].tolist(),
            width=plot_width, height=plot_height,
            tools=TOOLS, toolbar_location='above')
    bar.vbar(x="label",
             bottom=0, top="active_ideal",
             width=0.8, color="green", alpha=0.6, source=source)
    bar.vbar(x="label",
             bottom="active_ideal", top="active_real",
             width=0.8, color="red", alpha=0.6, source=source)
    bar.select(HoverTool).tooltips = tooltips


    bar.yaxis.axis_label = 'Energy consumption [kWh]'
    bar.xaxis.axis_label = ''
    bar.xaxis.major_label_orientation = math.pi/3
    bar.xaxis.major_label_text_font_size = "12pt"
    
    source_power_low = df_gain["gain"].min()
    source_power_high = df_gain["active_real"].max()
    bar.y_range = Range1d(source_power_low, source_power_high)
    bar.extra_y_ranges['euro'] = Range1d(source_power_low*energy2euro, source_power_high*energy2euro)
    bar.add_layout(LinearAxis(y_range_name="euro", axis_label="Euro"), 'right')

    bar.yaxis[0].formatter=NumeralTickFormatter(format="0")
    bar.yaxis[1].formatter=NumeralTickFormatter(format="0")

    tot_gain = df_gain["gain"].sum()
    tot_gain_euro = tot_gain * energy2euro
    tot_ideal = df_gain["active_ideal"].sum()
    tot_real = tot_gain + tot_ideal
    tot_real_euro = tot_real * energy2euro


    source = [{
        "label": "",
        "active_ideal": tot_ideal,
        "active_real": tot_real,
        "tt_gain_percentage": "%.1f%%" % (100.0*tot_gain/tot_real),
        "tt_gain_euro": int(tot_gain_euro),
        "tt_gain": int(tot_gain),
    }]

    source = pd.DataFrame(source)
    tooltips = [
        ("Gain %", "@tt_gain_percentage"),
        ("Gain kWh", "@tt_gain"),
        ("Gain €", "@tt_gain_euro"),
        ]

    bar2 = figure(x_range=source["label"].tolist(),
        width=200, height=plot_height,
        tools=TOOLS, toolbar_location='above')
    bar2.vbar(x="label",
             bottom=0, top="active_ideal",
             width=0.8, color="green", alpha=0.6, source=source)
    bar2.vbar(x="label",
             bottom="active_ideal", top="active_real",
             width=0.8, color="red", alpha=0.6, source=source)
    bar2.select(HoverTool).tooltips = tooltips

    # data = {'sample': ['ideal', 'gain'],
    #     'label': ['consumption', 'consumption'],
    #     'values': [tot_ideal, tot_gain],
    # }

    # bar2 = Bar(data, values='values', stack='sample',
    #            height=plot_height, width=250, legend=False, title="Total",
    #           tooltips =[("value", "@height")])
    bar2.y_range = Range1d(0, tot_real)
    bar2.legend.visible = False
    bar2.yaxis.axis_label = 'Energy consumption [kWh]'
    bar2.extra_y_ranges['euro'] = Range1d(0, tot_real_euro)
    bar2.add_layout(LinearAxis(y_range_name="euro", axis_label="Euro"), 'right')
    #bar.toolbar_sticky = False
    #bar.toolbar_location = 'below'
    #bar.yaxis[0].ticker=FixedTicker(ticks=[0,
    #                                       energy_gain['cum_reconstr_active'].max(),
    #                                       energy_gain['cum_real_active'].max()])
    #bar.yaxis[1].ticker=FixedTicker(ticks=[0,
    #                                       energy_gain['cum_reconstr_active'].max() * 0.17,
    #                                       energy_gain['cum_real_active'].max()*0.17])

    bar2.yaxis[0].ticker=FixedTicker(ticks=[0, tot_real])
    bar2.yaxis[1].ticker=FixedTicker(ticks=[0, tot_real_euro])

    #p.yaxis[0].formatter=NumeralTickFormatter(format="0")
    bar2.yaxis[0].formatter=NumeralTickFormatter(format="0")
    bar2.yaxis[1].formatter=NumeralTickFormatter(format="0")

    #label = Label(x=78, y=plot_height - 30 , x_units='screen', y_units="screen", text='&nbsp€ %d&nbsp' % gain_euro, render_mode='css',
      #border_line_color='black', border_line_alpha=.0,
      #background_fill_color='white', background_fill_alpha=0.5)

    #bar.add_layout(label)

    return [components(gridplot([[bar, bar2]], toolbar_location = 'below'))]
    #show(bar)
    #p = figure(title='', x_axis_type='datetime', plot_width=700, plot_height=plot_height,
               #toolbar_location='above', tools=TOOLS)

    #p = Bar(df_gain)

