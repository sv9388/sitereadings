import scrapy, json, logging, psycopg2
from datetime import date, datetime, timedelta

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr.strftime("%Y-%m-%d")
        curr += delta

def update_crawl_limit(start_date):
  try:
    conn=psycopg2.connect("dbname='siter' user='powersines' password='powersines' host='powersinesdb.cm6zndedailb.eu-central-1.rds.amazonaws.com'")
    cur = conn.cursor()
    q = """UPDATE crawl_limits set date_limit = '{}' where table_key = 'weather';""".format(start_date.strftime("%Y-%m-%d"))
    print(q)
    cur.execute(q)
    conn.commit()
    cur.close()
    conn.close()
  except Exception as e:
    print("TROUBLE at crawl limit update!!!!!", e)


def get_dateranges():
  start_date = datetime(2017, 1, 1)
  op = []
  try:
    conn=psycopg2.connect("dbname='siter' user='powersines' password='powersines' host='powersinesdb.cm6zndedailb.eu-central-1.rds.amazonaws.com'")
    cur = conn.cursor()
    cur.execute("""SELECT date_limit as max_date FROM crawl_limits where table_key = 'weather';""")
    rows = cur.fetchall()
    for row in rows:
      start_date = row[0]
    today = datetime.today()
    end_date = datetime(today.year, today.month, today.day)        
    while start_date <= end_date:
      op.append((start_date.strftime("%Y-%m-%d"), (start_date + timedelta(days = 7)).strftime("%Y-%m-%d")))
      start_date = start_date + timedelta(days = 7)
    conn.commit()
    cur.close()
    conn.close()
  except Exception as e:
    print("TROUBLE!!!!!", e)

  update_crawl_limit(start_date - timedelta(days = 7))
  return op

def get_devices():
  try:
    conn=psycopg2.connect("dbname='siter' user='powersines' password='powersines' host='powersinesdb.cm6zndedailb.eu-central-1.rds.amazonaws.com'")
    cur = conn.cursor()
    cur.execute("""SELECT device_id, latitude, longitude from device where latitude>=0.0""")
    rows = cur.fetchall()
    devices_arr = [[row[0], row[1] , row[2]] for row in rows]
    return devices_arr
  except Exception as e:
    print(e)
  return []

class Weather(scrapy.Spider):
  name = "weather"

  def get_req_list(self):
    config = {}
    with open("./weather_config.json", "r") as f:
      config  = json.load(f)
    api_key = config["api_key"]
    date_slices = get_dateranges()
    devices = get_devices()
    print(date_slices[0])
    params_tuples = []
    for start_date, end_date in date_slices:
      #self.log("{} - {}".format(start_date, end_date))
      for device in devices:
        device_id, latitude, longitude = device
        params_tuples.append((api_key, latitude, longitude, start_date, end_date, device_id))
    #self.log("Total devices = {}".format(len(devices)))
    #self.log("Total Dateranges = {}".format(len(date_slices)))
    #self.log("Found {} combinations".format(len(params_tuples)))
    return params_tuples

  def start_requests(self):
    url_fs = "http://api.weatherbit.io/v2.0/history/hourly?key={}&lat={}&lon={}&start_date={}&end_date={}&device_id={}"
    params_tuples = self.get_req_list()
    #self.log("Request list length = {}".format(len(params_tuples)))
    for pt in params_tuples:
      request = scrapy.http.Request(url = url_fs.format(*pt), callback = self.parse)
      yield request

  def parse(self, response):
    device_id = response.request.url.split("=")[-1]
    if response.status != 200:
      #self.log("No parseable JSON returned. Response was {}".format(json.loads(response.body.decode("utf-8"))))
      return {}
    jresp = response.body.decode("utf-8")
    jresp  = json.loads(jresp)
    warr = []
    for wdata in jresp["data"]:
      warr.append({"wdate" : datetime.strptime(wdata["datetime"], "%Y-%m-%d:%H"), "temp" : wdata["temp"]})
    op_json = { "device_id" : str(device_id), "weather_readings" : warr }
    return op_json
