# coding: utf-8
from models import *

import calendar
import datetime
import datetime as dt
import time

import numpy as np
import pandas as pd

from scipy import stats
from scipy import ndimage
#from bokeh.io import output_notebook, output_file, show, push_notebook, save
from bokeh.plotting import figure
#import bokeh.palettes
from bokeh.embed import components

#from bkcharts import HeatMap
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
)


def load_dataset(site_id, cut=False):
    device = Device.query.get(site_id)
    readings = Reading.query.filter_by(device_id = site_id).order_by(Reading.rdate).all() #device.readings
    arr = [[r.rdate, r.total_kwh] for r in readings]
    #arr = [[arr[i][0], int(arr[i][1] - arr[i-1][1]), arr[i][2]] for i in range(len(arr))]
    df = pd.DataFrame(arr, columns = ["date", "active"])
    df.active = df.active.diff()
    df = df.iloc[1:]
    df['anomaly_level'] = 0
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df = df.resample("15T").fillna('ffill')
    #df.active = df.active.fillna(int(df.active.mean())) #) #TODO: FILL NA 
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

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, EasterMonday
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
    wdays = df[df.date.apply(lambda x: (x.date() not in holidays))]
    if "reconstructed" in wdays.columns:
        wdays = wdays.loc[wdays.reconstructed == 0]
    return wdays

def get_valid_dates(dataset):
    dataset['data'] = dataset['date'].apply(lambda x: x.date())
    dataset['ora'] = dataset['date'].apply(lambda x: datetime.datetime(2000, 1, 1, x.hour, x.minute))
    #dataset.to_csv("./resampled.csv", index = False)
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
        print("########################################################################################")
        print(g)
        print(grouped.get_group(g))
        if not ((grouped.get_group(g)['filtered'] == 0) | grouped.get_group(g)['filtered'].isnull()).sum() >= 70: #TODO: Outlier = 70 for 6 years. For 1 year ~= 11
            #x = pd.concat([x, dataset[dataset['data'] == g]])
            valid_dates.append(g)
    print("Valid Dates Length = ", len(valid_dates))
    return valid_dates

    #return x


###########################################################################

#TODO: Hybrid prop in models for hour
def filter_by_hours(df, o, c, col="hour"):
    return df[(df[col] >= o) & (df[col] < c)]


def get_merged(site_id, spans, cut_date=None, temperature_col="temp", ):
    device = Device.query.get(site_id)
    pod_info = {"company" : device.system_name, "province_shortname" : "TODO", "description" : "Italy"}
 
    LAST_DAY = 6

    df_single_pod = load_dataset(site_id, cut = True) #load_dataset(os.path.join(data_dir, "%s.csv" % site_id), cut=True, use_cache=True) 
    print("Loaded = ", df_single_pod.shape)
    if cut_date is not None:
        df_single_pod = df_single_pod[pd.to_datetime(df_single_pod['date']) < pd.to_datetime(cut_date)]
    print("Cut Date = ", df_single_pod.shape)
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
        labels = dict([(y,str(y)) for y in range(2000, 2020)])
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

    print(merged.head())
    pod = merged['device'].values[0]
    #if pod.endswith(".csv"):
    #    pod = pod[:-4]

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
        c = p.circle(source=source, x="temp", y="daily_av_energy",
             color= color_l,
             muted_color=color_l,
             muted_alpha=MUTED_ALPHA,
             line_alpha=0.6,
             fill_alpha=0.6, size=10,
             #legend='legend_label'
             )
        legend_items[labels[label]] = [c]


    p.select_one(HoverTool).tooltips = [
            ('Date', '@tooltip'),
            ('Av. Energy', '@daily_av_energy'),
            ('Ext. Temp.', '@temp')
            ]

    source_prova = ColumnDataSource(averages)

    x = averages["temp"].values
    upper = averages["energy_mean"].values
    lower = averages["mean-std"].values
    band_x = np.append(x, x[::-1])
    band_y = np.append(lower, upper[::-1])

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


def find_fences(values, factor=1.5):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    return q1 - factor*iqr, q3 + factor*iqr

def remove_outliers(values, factor=1.5):
    l, h = find_fences(values, factor=factor)
    return values[(l <= values) & (values <= h)]


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

from scipy.ndimage import gaussian_filter1d

def gaussian_smooth(averages):
    nuovo_averages = averages.copy()
    nuovo_averages['energy_mean'] = gaussian_filter1d(averages['energy_mean'], sigma=1)
    nuovo_averages['energy_std'] = gaussian_filter1d(averages['energy_std'], sigma=1)
    nuovo_averages['mean-std'] = nuovo_averages['energy_mean'] - nuovo_averages['energy_std']
    return nuovo_averages

from bokeh.models import ColumnDataSource, Range1d
from bokeh.models import Range1d

def filter_periods(df, periods, col="date"):
    if periods is None or periods == "whole":
        return df
    else:
        df_all = [filter_period(df, period, col=col) for period in periods]
        return pd.concat(df_all)


"""
ComputeAverages()
FilterFunc()
gen_filters()
get_merged(pd.DataFrame(), [])
gaussian_smooth([])
plot_consumption_curve(pd.DataFrame())
"""
