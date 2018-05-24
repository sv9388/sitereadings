import scrapy, json, logging, psycopg2
from datetime import date, datetime, timedelta

INPUT_DATE_FORMAT = "%Y-%m-%d %H:%M"
UTC_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr.isoformat() + ".000Z"
        curr += delta

def get_dateranges(device_id, should_get_all):
  did = None
  conn=psycopg2.connect("dbname='siter' user='powersines' password='powersines' host='powersinesdb.cm6zndedailb.eu-central-1.rds.amazonaws.com'")

  try:
    cur = conn.cursor()
    q = """SELECT id from device where device_id = '{}';""".format(device_id)
    print(q)
    cur.execute(q)
    rec = cur.fetchone()
    print(rec)
    did = rec[0]
    print(did)
  except Exception as e:
    print(e)

  if not did:
    print("No device id found")
    return None, None

  start_date, end_date = None, None
  if not should_get_all:
    today = datetime.today()
    daypart = datetime(today.year, today.month, today.day, 0, 0)
    return daypart - timedelta(days = 1), daypart

  try:
    cur = conn.cursor()
    q = """select max(rdate) from reading where device_id = {}""".format(did)
    print(q)
    cur.execute(q)
    rec = cur.fetchone()
    print(type(rec[0]), rec[0].year, rec[0].month, rec[0].day, rec[0].hour, 0)
    start_date = datetime(rec[0].year, rec[0].month, rec[0].day, rec[0].hour, 0) + timedelta(hours = 1) if rec else datetime(2017, 1, 1)
    today = datetime.today()
    end_date = datetime(today.year, today.month, today.day, 0, 0) 
    print(start_date, end_date)
  except Exception as e:
    print(e)

  return start_date, end_date

db_col_mapping = { "Sum_of_Powers_on_all_3_phases" : "total_power_kw" , "Accumulated_kWh_from_the_beginning_of_installation" : "total_kwh" }

class PowerSines(scrapy.Spider):
  name = "psines"

  def get_req_list(self):
    frmdata = [] #{"user" : "salmona.emanuel@googlemail.com","password" : "salmona13579", "code" : "0000", "device" : "160084" }]
    config = {}
    with open("./config.json", "r") as f:
      config  = json.load(f)
    devices = config["devices"]
    usr, pwd, code = config["emsuser"], config["emspasswd"], config["emscode"]

    for did in devices:
      start_date, end_date = get_dateranges(did, config["get_all"])
      print(start_date, end_date)
      if not start_date and not end_date:
        continue
      dateranges = perdelta(start_date, end_date, timedelta(hours=1))
      for timeslice in dateranges:
        dt = {"user" : usr, "password" : pwd, "code" : code, "device" : did, "timestamp" : timeslice}
        frmdata.append(dt)
    self.log("Found {} combinations".format(len(frmdata)))
    return frmdata

  def start_requests(self):
    url = "http://ems.powersines.com:7272/rest/resources/reports/general/retrieve"
    frmdata = self.get_req_list()
    self.log("Request list length = {}".format(len(frmdata)))
    for fd in frmdata:
      self.log(fd)
      yield scrapy.http.FormRequest(url = url, formdata = fd, headers = {"Content-Type" : "application/x-www-form-urlencoded", "Accept" : "application/json"}, callback = self.parse)

  def parse(self, response):
    if response.status != 200:
      self.log("No parseable JSON returned. Response was {}".format(json.loads(response.body.decode("utf-8"))))
      return {}
    jresp = response.body.decode("utf-8")
    jresp  = json.loads(jresp)
    marr = []
    for measurement in jresp["measurements"]:
      mdict = {db_col_mapping[value["ParamName"]] : float(value["Value"]) for value in measurement["Values"] if value["ParamName"] in db_col_mapping.keys() }
      mdict["rdate"] = datetime.strptime(measurement["TimeStart"], UTC_TIME_FORMAT)        
      marr.append(mdict)
    op_json = { "device_id" : jresp["DeviceID"], "readings" : marr }
    return op_json
