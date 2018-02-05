import pandas as pd, glob, csv, re, psycopg2, copy
from models import *
from settings import *
from io import StringIO
from sqlalchemy.orm import sessionmaker

reading_columns = ["rdate" , "input_voltage_l1" , "input_voltage_l2" , "input_voltage_l3" , "output_voltage" , "current_l1" , "current_l2" , "current_l3" , "pf_l1" , "pf_l2" , "pf_l3" , "power_l1" , "power_l2" , "power_l3" , "total_power_kw" , "total_kwh" , "window2_consumption" , "today_saved" , "today_consumption" , "yesterday_saved" , "yesterday_consumption" , "weekly_saved" , "weekly_consumption" , "last_week_saved" , "last_week_consumption" , "monthly_saved" , "monthly_consumption" , "last_month_saved" , "last_month_consumption" , "yearly_saved" , "yearly_consumption" , "last_year_saved" , "last_year_consumption" , "over_load_indication" , "missing_phase_indication" , "over_temperature_indication" , "temperature_l1" , "temperature_l2" , "temperature_l3" , "cpu_temperature" , "device_mode" , "reduction_step" , "software_resets"]

class Merger():
  def __init__(self,  merge_folder = "./sitereadings", del_successful_files = True):
    engine = create_engine(DB_URI)
    session = sessionmaker()
    session.configure(bind=engine)
    self.session = session()
    self.merge_folder = merge_folder
    self.del_successful_files = del_successful_files
    self.cluttered = False

  def _get_df_from_csv(self, fl):
    s = StringIO()
    with open(fl) as f:
      data = list(csv.reader(f))
      #data = [[d for d in row] for row in data]
      print("Csv data", data[0], len(data))
      edf = pd.DataFrame(data[2:], columns = reading_columns) 
      df = edf[reading_columns[1:]].apply(pd.to_numeric)
      df['rdate'] = edf['rdate']
      device_details = ' '.join(data[0]).strip()
      device_details = re.sub("\s+", " ", device_details)[len("Global report for device '"):-1]
      print("File", fl)
      print("Details", df.head(), df.shape)
      print("Device id", device_details)
      return [device_details, df]
    return None

  def _get_new_readings(self):
    print("Glob search string", self.merge_folder + "/**/*.csv")
    files = glob.glob(self.merge_folder + "/**/*.csv")
    print("Files", files)
    devs_dfs = [self._get_df_from_csv(x) for x in files]
    print("Dataframes", len(devs_dfs))
    return files, devs_dfs

  def _retrieve_reading(self, row_dict, devid):
    new_dict = copy.deepcopy(row_dict)
    if not self.cluttered:
      print("To be inserted dict", new_dict)
    reading_entity = self.session.query(Reading).filter(Reading.rdate == new_dict['rdate']).filter(Reading.device_id == devid).first()
    if reading_entity:
      if not self.cluttered:
        print("Reading exists for time")
      existing_dict = reading_entity.__dict__
      if not self.cluttered:
        print("Existing dict", existing_dict)
      for k, v in new_dict.items():
        if k != 'rdate':
          new_dict[k] = (new_dict[k] + existing_dict[k])*1./2
      self.session.delete(reading_entity)
      self.session.commit()
    if not self.cluttered:
      print("updated dict", new_dict)
      self.cluttered = True
    return Reading(**new_dict)
      
  def _update_db(self, device, df):
    success = True
    device_entity = self.session.query(Device).filter(Device.name == device).first()
    if not device_entity:
      device_entity = Device(name = device)
      self.session.add(device_entity)
      self.session.commit()  
    print(device, device_entity.id, device_entity)
    print(df.head())
    print(df.columns, df.shape)
    for idx, row in df.iterrows():
      #print("Reading dict", row.to_dict())
      reading_entity = self._retrieve_reading(row.to_dict(), device_entity.id)
      #print("New reading obj", reading_entity)
      reading_entity.readings_for_device = device_entity
      self.session.add(reading_entity)
      self.session.commit()
    return True

  def _delete_files_and_folders(self, files, did_succeed_arr):
    return 

  def merge(self):
    files, devs_dfs = self._get_new_readings()
    did_succeed_arr = []
    for device, df in devs_dfs:
      status = self._update_db(device, df) 
      did_succeed_arr.append(status)
    if self.del_successful_files:
      self._delete_files_and_folders(files, did_succeed_arr)

if __name__ == "__main__":
  merger = Merger()
  merger.merge()
