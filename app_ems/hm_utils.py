# coding: utf-8
from models import *

import calendar
import datetime
import datetime as dt
import math
from math import pi
import time

import numpy as np
import pandas as pd

from skimage import exposure
from scipy import stats
from scipy import ndimage

import bokeh
from bokeh.plotting import figure
#import bokeh.palettes
from bokeh.layouts import gridplot
from bokeh.embed import components

#from bkcharts import HeatMap
from bokeh.models import (
    ColumnDataSource,
    LinearAxis,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    FixedTicker,
    PrintfTickFormatter,
    DatetimeTickFormatter,
    ColorBar,
    FuncTickFormatter,
)


import sys
epoch = datetime.datetime.utcfromtimestamp(0)

def load_dataset(site_id, cut=False):
    device = Device.query.get(site_id)
    readings = Reading.query.filter_by(device_id = site_id).order_by(Reading.rdate).all() #device.readings
    arr = [[r.rdate, r.total_kwh] for r in readings]
    df = pd.DataFrame(arr, columns = ["date", "active"])

    df['active']  = df['active'].apply(lambda x: int(x))
    df['anomaly_level'] = 0
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    last_date = pd.Timestamp(df.date.max().date())
    print(last_date)
    df = df[df.date < pd.Timestamp('2018-05-01')]
    df = df.sort_values(by = "date")
    df = df.set_index('date')

    df = df.resample("60T").fillna("nearest") #) #TODO: FILL NA
    df.active = df.active.diff()
    df.iloc[0].active  = df.iloc[1].active
    df['device'] = device.device_id
    df['anomaly_level'] = 0
    df = df.reset_index()
    res = df.loc[:, ['date', 'active', 'device', 'anomaly_level']]
    res.loc[:, 'date'] = pd.to_datetime(res['date'])

    if cut:
        res["reconstructed"] = 1
        valid_dates = get_valid_dates(res)
        res.loc[res.date.apply(lambda x: x.date() in valid_dates), 'reconstructed'] = 0
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
    print(values, labels)
    return labels


TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

def heatmap(dataset, pod_name, palette):
    '''
    Plot heat map of consumption with colorbar.
    '''
    print("HEAT MAP")
    print(dataset.shape, dataset.columns)
    print(dataset.head())
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
               tools=TOOLS, toolbar_location='below')

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

    max_label = max(bin_labels)
    if max_label >= 100:
        bin_labels = ["%03d" % s for s in bin_labels]
    else:
        bin_labels = ["%04.1f" % s for s in bin_labels]

    label_dict_str = "";
    for i, s in zip(ticker.ticks, bin_labels):
        label_dict_str = label_dict_str + """labels[%d] = "%s"; """ % (i, s)#label_dict[i] = s

    formatter = FuncTickFormatter(code="""
                                        var labels = {}; %s
                                        return labels[tick];
                                       """ % label_dict_str)

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


BASE_YEAR = 2017

def heatmap(dataset, ct, pod_name, palette, x_range=None, y_range=None, active_low=None, active_high=None):
    df = dataset[['date', 'active']]
    df.date = pd.to_datetime(df.date)
    df = df[df.date>=pd.to_datetime(datetime.datetime(BASE_YEAR, 1, 1, 0, 0))]
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df.date.apply(lambda x: str(int((datetime.datetime(x.year, x.month, x.day, 0, 0, 0) - epoch).total_seconds() * 1000)))
    df['time'] = df.date.apply(lambda x: str(int((datetime.datetime(2000,1,1,x.hour, x.minute) - epoch).total_seconds()  * 1000)))
    df = df[['day', 'time', 'active']].sort_values(by = ['day', 'time'])
    df.active = pd.to_numeric(df.active, downcast = 'float')

    mapper = LinearColorMapper(palette=palette, low=df.active.min(), high=df.active.max())
    source = ColumnDataSource(df[['day', 'time', 'active']])

    days = list(df.day.drop_duplicates())
    hours = list(df.time.drop_duplicates())

    p = figure(title="{}".format(pod_name),
           #x_axis_type='datetime', y_axis_type='datetime',
           x_range = days, y_range = hours,
           x_axis_location="below", plot_width=900, plot_height=400,
           tools=TOOLS, toolbar_location='below')
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3
    p.xaxis.formatter = DatetimeTickFormatter(days = ['%m/%d', '%a%d'], months = ['%m/%Y', '%b %Y'], years = ['%Y'])
    p.yaxis.formatter = DatetimeTickFormatter(hourmin = ['%H:%M'], hours = ['%Hh', '%H:%M'])

    p.rect(x="day", y="time", width=1, height=1, source = source, fill_color={'field': 'active', 'transform': mapper}, line_color=None)

    tc = int(df.active.max() - df.active.min())
    tc = min(15, tc)
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10pt",
                     ticker=BasicTicker(desired_num_ticks=tc),
                     formatter=PrintfTickFormatter(format = "%d"),
                     label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    p.select_one(HoverTool).tooltips = [
     ('date', '@time @day'),
     ('active', '@active'),
    ]

    return p

def plot_heatmap(df, palette=bokeh.palettes.Spectral10, filters=None, by_year=True):
    op = None
    df = apply_filters(df, filters, date_col="date")
    active_low = df["active"].min()
    active_high = df["active"].max()

    ct = 0
    if by_year:
        plots = []
        x_range = None
        y_range = None
        for year, df_year in df.groupby(by="year"):
            title = ""
            try:
                #pod_id = df.iloc[0]['device']
                pod_info = {"company" : "TODO", "description" : "Get unique name from Emanuel", "province_shortname" : "IT"}
                title = " ".join([pod_id, pod_info["company"], pod_info["description"], pod_info["province_shortname"]])
                title = "%s, year %d" % (title, year)
            except Exception as e:
                print(e)
                title = "Unknown "

            p = heatmap(df_year, ct = ct, pod_name=title, palette=palette, x_range=x_range, y_range=y_range, active_low=active_low, active_high=active_high)
            ct += 1
            plots.append([p])

            if x_range is None:
                x_range = p.x_range
                y_range = p.y_range

        op = [components(gridplot(plots))]
    else:
        try:
            pod_id = df.iloc[0]['device']
            pod_info = {"company" : "TODO", "description" : "Get unique name from Emanuel", "province_shortname" : "IT"}
            title = " ".join([pod_id, pod_info["company"], pod_info["description"], pod_info["province_shortname"]])

        except:
            title = "Unknown "

        p = heatmap(df, ct = ct, pod_name=title, palette=palette, active_low=active_low, active_high=active_high)
        ct += 1
        op = [components(p)]
    return op

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

def entropy(data):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entro = stats.entropy(p_data)  # input probabilities to get the entropy
    return entro

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

def get_merged(site_id, spans, cut_date=None, temperature_col="temp", ):
    #device = Device.query.get(site_id)
    #pod_info = {"company" : device.system_name, "province_shortname" : "TODO", "description" : "Italy"}
    pod_info = {"company" : "TODO", "description" : "Get unique name from Emanuel", "province_shortname" : "IT"}
    LAST_DAY = 6

    df_single_pod = load_dataset(site_id, cut = True) #load_dataset(os.path.join(data_dir, "%s.csv" % site_id), cut=True, use_cache=True)
    if cut_date is not None:
        df_single_pod = df_single_pod[pd.to_datetime(df_single_pod['date']) < pd.to_datetime(cut_date)]
    df_w_d = filter_holidays(df_single_pod)
    df_w_d.loc[:, 'data'] = pd.to_datetime(df_w_d['data'])
    df_w_d.loc[:, 'active'] = df_w_d['active'] * 4
    df_w_d.loc[:, 'tooltip'] = df_w_d['data'].apply(lambda x: x.strftime("%Y-%m-%d") + ' ' + calendar.day_name[x.weekday()])

    df_w_d["year"] = pd.to_datetime(df_w_d["date"]).dt.year
    df_w_d["month"] = pd.to_datetime(df_w_d["date"]).dt.month
    df_w_d["hour"] = pd.to_datetime(df_w_d["date"]).dt.hour
    df_w_d["weekday"] = pd.to_datetime(df_w_d["date"]).dt.weekday

    first_band = df_w_d[df_w_d["weekday"] <= LAST_DAY]
    dfs = []
    for o,c in spans:
        dfs.append(filter_by_hours(first_band, o, c, "hour"))
    first_band = pd.concat(dfs).sort_values(by="date")
    av_energy = first_band.groupby("data")[["active"]].mean().rename(columns={"active": "daily_av_energy"})

    import meteo
    meteo_data = meteo.load_meteo_data(site_id)

    dfs = []
    for o,c in spans:
        dfs.append(filter_by_hours(meteo_data, o, c, "hour"))
    first_band_meteo = pd.concat(dfs).sort_values(by="data")
    first_band_meteo['temp'] = first_band_meteo[temperature_col]

    av_meteo = pd.DataFrame(first_band_meteo.groupby('data')['temp'].mean())
    merged = pd.merge(av_meteo, av_energy, left_index=True, right_index=True)

    merged.reset_index(inplace=True)
    merged.rename(columns={'index':'data'}, inplace=True)
    merged['date_index'] = merged.index
    merged.loc[:, 'tooltip'] = merged['data'].apply(lambda x: x.strftime("%Y-%m-%d") + ' ' + calendar.day_name[x.weekday()])
    merged.loc[:, 'colorbar'] = merged['data'].apply(lambda x: x.strftime("%Y-%m-%d"))

    merged["device"] = site_id
    merged['location'] = pod_info["description"]
    merged['province'] = pod_info["province_shortname"]
    merged['company'] = pod_info["company"]

    merged['rounded_temp'] = merged['temp'].apply(lambda x: int(round(x)))

    merged["month"] = pd.to_datetime(merged["data"]).dt.month
    merged["weekday"] = pd.to_datetime(merged["data"]).dt.weekday
    merged["year"] = pd.to_datetime(merged["data"]).dt.year

    
    return merged, df_w_d

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
        X = df[df["rounded_temp"] >= lin_reg_from][["temp", "daily_av_energy"]].as_matrix()
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

from bokeh.models import FixedTicker
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

from scipy.ndimage import gaussian_filter1d

def gaussian_smooth(averages):
    nuovo_averages = averages.copy()
    nuovo_averages['energy_mean'] = gaussian_filter1d(averages['energy_mean'], sigma=1)
    nuovo_averages['energy_std'] = gaussian_filter1d(averages['energy_std'], sigma=1)
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
    for k in similar_days.keys():
        similar_means = []
        p = figure(title='', x_axis_type='datetime', plot_width=900, plot_height=600,
                   toolbar_location='above', tools=TOOLS)
        tmp = pd.DataFrame([], columns = ["ora"])
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
        index_date = pd.date_range(start_date, end_date, freq='60T')
        if active_mean.shape[0] == index_date.shape[0]:
          data.append(pd.DataFrame({'active': active_mean, 'date': index_date}))

    reconstructed = pd.concat(data)
    reconstructed['device'] = df['device'].values[0]
    reconstructed['anomaly_level'] = df['anomaly_level'].values[0]
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
    pod = df.iloc[0]['device']
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

        df_gain = pd.DataFrame(data, columns=['device', "date", "year", "month", "active_real", "active_ideal"])
        df_gain["label"] = df_gain["date"].apply(lambda x: "%4d-%02d" % (x.year, x.month))
        df_gain = df_gain.sort_values(by="label")

    elif by == "day of week":
        df_m["day of week"] = df_m[date_col].dt.weekday

        for weekday, dfg in df_m.groupby(by=["day of week"]):
            data.append((pod, weekday, dfg["active_real"].sum(), dfg["active_ideal"].sum()))

        df_gain = pd.DataFrame(data, columns=['device', "day of week", "active_real", "active_ideal"])
        df_gain["label"] = df_gain["day of week"].apply(lambda x: calendar.day_abbr[int(x)])
        df_gain = df_gain.sort_values(by="day of week")

    elif by == "year":
        df_m["year"] = df_m[date_col].dt.year

        for year, dfg in df_m.groupby(by="year"):
            data.append((pod, year, dfg["active_real"].sum(), dfg["active_ideal"].sum()))

        df_gain = pd.DataFrame(data, columns=['device', "year", "active_real", "active_ideal"])
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

from bokeh.models import ColumnDataSource, Range1d
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

    df_gain = calculate_gain(df, df_ideal, by=by)

    source = df_gain[["label", "gain", "active_ideal", "active_real", "gain_percentage"]]
    source["tt_gain"] = source["gain"].astype(int)
    source["tt_gain_euro"] = (source["gain"] * energy2euro).astype(int)
    source["tt_gain_percentage"] = source["gain_percentage"].apply(lambda x: "%.1f%%" % x)

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
    bar2.y_range = Range1d(0, tot_real)
    bar2.legend.visible = False
    bar2.yaxis.axis_label = 'Energy consumption [kWh]'
    bar2.extra_y_ranges['euro'] = Range1d(0, tot_real_euro)
    bar2.add_layout(LinearAxis(y_range_name="euro", axis_label="Euro"), 'right')
    bar2.yaxis[0].ticker=FixedTicker(ticks=[0, tot_real])
    bar2.yaxis[1].ticker=FixedTicker(ticks=[0, tot_real_euro])
    bar2.yaxis[0].formatter=NumeralTickFormatter(format="0")
    bar2.yaxis[1].formatter=NumeralTickFormatter(format="0")
    return [components(gridplot([[bar, bar2]], toolbar_location = 'below'))]

    #p = Bar(df_gain)

"""
ComputeAverages()
FilterFunc()
gen_filters()
get_merged(1, [])
gaussian_smooth([])
plot_heatmap(pd.DataFrame())
reconstruct_by_weekday(pd.DataFrame(), pd.DataFrame(), None)
plot_saving(pd.DataFrame(), pd.DataFrame())
"""
