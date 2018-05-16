from wtforms import SelectField, Form, validators, SubmitField, DateTimeField, StringField, SelectField
from flask_wtf import FlaskForm
from enums import *
from models import Device
from datetime import datetime, timedelta
import random
from optgroup_select_field import OptgroupSelectWidget, OptgroupSelectField, OptgroupSelectMultipleField

class PredictionForm(Form):
  #site_ids = OptgroupSelectMultipleField("Site Ids", [validators.DataRequired()], coerce = int)
  site_id = SelectField("Site Id", [validators.DataRequired()], coerce = int)
  predict_for_timerange = SelectField("Predict values for", choices =  [(WEEK, "This week"), (DAY, "Today")], coerce = int, default = DAY)
  aggregate_unit = SelectField("Aggregate Period", [validators.Optional()], coerce = int, choices = [(FIFTEEN_MINUTES, "15 Minutes"), (HOURLY, "Hourly")], default = HOURLY)
 
  def __init__(self, *args, **kwargs):
    super(PredictionForm, self).__init__(*args, **kwargs)
    device_records = Device.query.all()
    devices = [(d.id, d.device_id) for d in device_records]
    #sizes, site_types = set([x.tag_size for x in device_records]), set([x.tag_site_type for x in device_records])
    #cdict = { }
    #for x in sizes:
    #  cdict[x] = tuple([(d.id, d.device_id) for d in device_records if d.tag_size == x])
    #for x in site_types:
    #  cdict[x] = tuple([(d.id, d.device_id) for d in device_records if d.tag_site_type == x])
    #self.site_id.choices = tuple(cdict.items())
    self.site_id.choices = devices
    self.site_id.default = 1

DEFAULT_DATESTR = "{} - {}".format((datetime.today() - timedelta(days = 7)).strftime("%Y-%m-%d"), datetime.today().strftime("%Y-%m-%d"))
print(DEFAULT_DATESTR)
class CustomSitesForm(Form):
  metric = SelectField("Metric", [validators.DataRequired()], choices = [("total_kwh", "KwH"), ("kwh_psqm", "Kwh/SqM")], default = "total_kwh")
  chart_type = SelectField("Chart Type", [validators.DataRequired()],  choices = [(BAR , "Bar"), (LINE , "Line"), (COLUMN , "Column") ], default = COLUMN)
  site_ids = OptgroupSelectMultipleField("Site Ids", [validators.DataRequired()], coerce = int, default = [1, 2, 3, 4, 5])
  date_range = StringField("Date Range", [validators.DataRequired()], default = DEFAULT_DATESTR)
  date_range2 = StringField("Second Date Range", [validators.Optional()], default = "")
  aggregate_unit = SelectField("Aggregate Period", [validators.Optional()], coerce = int, choices = [(FIFTEEN_MINUTES, "15 Minutes"), (HOURLY, "Hourly"), (DAILY, "Daily")], default = DAILY)

  def __init__(self, *args, **kwargs):
    super(CustomSitesForm, self).__init__(*args, **kwargs)
    device_records = Device.query.all()
    devices = [(d.id, d.device_id) for d in device_records]
    sizes, site_types = set([x.tag_size for x in device_records]), set([x.tag_site_type for x in device_records])
    cdict = { }
    for x in sizes:
      cdict[x] = tuple([(d.id, d.device_id) for d in device_records if d.tag_size == x])
    for x in site_types:
      cdict[x] = tuple([(d.id, d.device_id) for d in device_records if d.tag_site_type == x])

    self.site_ids.choices = tuple(cdict.items())

class DeepDiveForm(FlaskForm):
  site_id = SelectField("Site Id", coerce = int)

  def __init__(self, *args, **kwargs):
    super(DeepDiveForm, self).__init__(*args, **kwargs)
    device_records = Device.query.all()
    devices = [(d.id, d.device_id) for d in device_records if d.latitude]
    self.site_id.choices = devices
    if len(devices) > 0:
      random.shuffle(devices)
      default = devices[0]
      self.site_id.default = default[0]

