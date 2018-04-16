import os

import pandas as pd

from dd_settings import get_setting
from dd_utils import get_pod_info

def get_nearest_weather_station(pod_id, meteo_dir=None, filepath=None):

    if meteo_dir is None:
        meteo_dir = get_setting("METEO_DIR")


    if filepath is None:
        filepath = get_setting("METEO_NEAREST_STATION_FILEPATH")

    df = pd.read_csv(filepath, index_col=0)
    try:
        df_pod = df[df["pod"] == pod_id].iloc[0]
    except IndexError:
        raise Exception("Can't load weather data, pod %s not found in %s" % (pod_id, filepath))

    code = df_pod["station_code"]

    filename = None
    for filename in os.listdir(meteo_dir):
        if filename.startswith("%s-" % code):
            break

    return filename

def get_commune_weather_station(pod_id, meteo_dir=None):

    if meteo_dir is None:
        meteo_dir = get_setting("METEO_DIR")
    pod_info = get_pod_info(pod_id)
    file_name = str(pod_info["province_shortname"] + '-' + pod_info["description"] + '.csv')
    meteo_files = list(filter(lambda x: x.endswith(".csv") and
                             file_name in x, os.listdir(meteo_dir)))

    if len(meteo_files) > 0:
        return meteo_files[0]
    else:
        return None

EN_COLS = "weather_city_code	date	weather_symbol_id	rainfall	rainfall_intensity	rainfall_prob	rainfall_uom	temperature	perceived_temperature	freezing_level	snow_level	wind_direction	wind_intensity	sea_power	sea_temperature	flurry	humidity	pressure	uv_index	uv_description	accumulation	windchill	solar_radiation	effective".split("\t")
IT_COLS = "codice_localita	ora\tsimbolo_meteo	precipitazioni	intensita_precipitazioni	probabilita_precipitazioni	uom_precipitazioni	temperatura	temperatura_percepita	zero_termico	quota_neve	direzione_vento	intensita_vento	forza_mare	temperatura_mare	raffica	umidita	pressione	indice_uv	descrizione_uv	accumulo	windchill	radiazione_solare".split("\t")

EN_IT_MAP = dict(zip(EN_COLS, IT_COLS))

def is_in_english(filepath):
    with open(filepath) as f:
        line = f.readline()

    return "weather_city_code" in line
def load_meteo_data(pod_id, meteo_dir=None, methods=None):

    if meteo_dir is None:
        meteo_dir = get_setting("METEO_DIR")

    if methods is None:
        methods = get_setting("METEO_STATION_METHODS")

    for method in methods:
        if method == "nearest":
            fn = get_nearest_weather_station
        elif method == "commune":
            fn = get_commune_weather_station

        filename = fn(pod_id)
        if not filename is None:
            break

    if filename is None:
        raise Exception("Could not find a weather stations for pod %s" % pod_id)


    meteo_filepath = os.path.join(meteo_dir, filename)
    print("Loading meteo data from: %s" % meteo_filepath)

    if is_in_english(meteo_filepath):
        meteo_data = pd.read_csv(meteo_filepath, sep=",", parse_dates=['date'], index_col=0)
        meteo_data = meteo_data.rename(columns=EN_IT_MAP)
        #print(EN_IT_MAP)
        meteo_data["data"] = pd.to_datetime(meteo_data["ora"].dt.date)
        # print(meteo_data["data"].dtype)
        # print(meteo_data["ora"].dtype)
        # print(meteo_data.head())
        # assert False

    else:
        meteo_data = pd.read_csv(meteo_filepath, sep=",", parse_dates=['data', 'ora'], index_col=0)
        #
        # print(meteo_data["data"].dtype)
        # print(meteo_data["ora"].dtype)

        # print(meteo_data.head())
        # assert False
    meteo_data["hour"] = meteo_data["ora"].dt.hour

    return meteo_data

