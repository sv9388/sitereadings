import scrapy, json, logging, psycopg2
from datetime import date, datetime, timedelta

INPUT_DATE_FORMAT = "%Y-%m-%d %H:%M"
UTC_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr.isoformat() + ".000Z"
        curr += delta

def get_dateranges():
  try:
    conn=psycopg2.connect("dbname='siter' user='powersines' password='powersines' host='powersinesdb.cm6zndedailb.eu-central-1.rds.amazonaws.com'")
    cur = conn.cursor()
    cur.execute("""SELECT MAX(RDATE) as max_date FROM READING;""")
    rows = cur.fetchall()
    start_date = datetime(2017, 1, 1)
    for row in rows:
      start_date = row[0]
    end_date = datetime.today() - timedelta(days = 1)
    return start_date, end_date
  except Exception as e:
    print(e)
  return None, None

db_col_mapping = { "Sum_of_Powers_on_all_3_phases" : "total_power_kw" , "Accumulated_kWh_from_the_beginning_of_installation" : "total_kwh" }
#"Input_Voltage_on_phase_1" : "input_voltage_l1" , "Input_Voltage_on_phase_2" : "input_voltage_l2" , "Input_Voltage_on_phase_3" : "input_voltage_l3" ,"Output_voltage_lowest" : "output_voltage" , "Input_Current_on_phase_1" : "current_l1" ,"Input_Current_on_phase_2" : "current_l2" , "Input_Current_on_phase_3" : "current_l3" , "PF_L1_meaning" : "pf_l1" , "PF_L2_meaning" : "pf_l2" , "PF_L3_meaning" : "pf_l3" , "Input_Power_on_Phase_1" : "power_l1" , "Input_Power_on_Phase_2" : "power_l2" , "Input_Power_on_Phase_3" : "power_l3" , "kWh_in_Window_2_Initilized_each_day" : "window2_consumption" , "Accumulated_kWh_SAVED_for_today_Initialized_at_00_00" : "today_saved" ,"Accumulated_kWh_consumption_for_today_Initialized_at_00_00" :  "today_consumption" , "Total_kWh_SAVED_for_the_previous_day_Updated_at_00_00" : "yesterday_saved" , "Total_kWh_consumption_for_the_previous_day_Updated_at_00_00" : "yesterday_consumption" , "Accumulated_kWh_SAVED_for_this_week_Initialized_at_Mon_00_00" : "weekly_saved" , "Accumulated_kWh_consumption_for_this_week_Initialized_at_Mon_00_00" : "weekly_consumption" , "Total_kWh_SAVED_for_the_previous_week_Updated_on_Mon_00_00"  : "last_week_saved" , "Total_kWh_consumption_for_the_previous_week_Updated_on_Mon_00_00" : "last_week_consumption" , "Accumulated_kWh_SAVED_for_this_month" : "monthly_saved" , "Accumulated_kWh_consumption_for_this_month"  : "monthly_consumption" , "Total_kWh_SAVED_for_the_previous_month" : "last_month_saved" , "Total_kWh_consumption_for_the_previous_month" : "last_month_consumption" , "Accumulated_kWh_SAVED_for_this_year" : "yearly_saved" , "Accumulated_kWh_consumption_for_this_year" : "yearly_consumption" , "Total_kWh_SAVED_for_the_previous_year" : "last_year_saved" , "Total_kWh_consumption_for_the_previous_year"  : "last_year_consumption" , "Over_load_indication" : "over_load_indication" , "Phase_missing_indication" : "missing_phase_indication" , "Over_temperature_indication" : "over_temperature_indication" , "Trafo_temperature_on_L1" : "temperature_l1" , "Trafo_temperature_on_L2" : "temperature_l2" , "Trafo_temperature_on_L3" : "temperature_l3" , "CPU_Temperature" : "cpu_temperature" , "ComEC_mode_vout_status_meaing" : "device_mode" , "reduction_step_meaning" : "reduction_step" , "reset_counter_meaning" : "software_resets"}

class PowerSines(scrapy.Spider):
  name = "psines"

  def get_req_list(self):
    frmdata = [] #{"user" : "salmona.emanuel@googlemail.com","password" : "salmona13579", "code" : "0000", "device" : "160084" }]
    config = {}
    with open("./config.json", "r") as f:
      config  = json.load(f)
    devices = config["devices"]
    usr, pwd, code = config["emsuser"], config["emspasswd"], config["emscode"]

    if not "start" in config.keys() and not "end" in config.keys():
      for did in devices:
        frmdata.append({"user" : usr, "password" : pwd, "code" : code, "device" : did})
      self.log("Found {} combinations".format(len(frmdata)))
      return frmdata

    self.log("Timestamp specified")

    start_date, end_date = get_dateranges()

    if not start_date and not end_date:
      return []

    #For custom timeranges, update start and end in config.json, uncomment next line and comment the line after that.
    #dateranges = perdelta(datetime.strptime(config["start"], INPUT_DATE_FORMAT), datetime.strptime(config["end"], INPUT_DATE_FORMAT), timedelta(hours=1))
    dateranges = perdelta(start_date, end_date, timedelta(hours=1))

    for timeslice in dateranges:
      self.log(timeslice)
      for did in devices:
        frmdata.append({"user" : usr, "password" : pwd, "code" : code, "device" : did, "timestamp" : timeslice})     
    self.log("Found {} combinations".format(len(frmdata)))
    return frmdata

  def start_requests(self):
    url = "http://ems.powersines.com:7272/rest/resources/reports/general/retrieve"
    frmdata = self.get_req_list()
    self.log("Request list length = {}".format(len(frmdata)))
    for fd in frmdata:
      yield scrapy.http.FormRequest(url = url, formdata = fd, callback = self.parse)

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
