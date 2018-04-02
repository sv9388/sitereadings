from wtforms import SelectField, Form, validators, SubmitField, DateTimeField, StringField
from flask_wtf import FlaskForm
from enums import *
from models import Device
from datetime import datetime, timedelta
import random
from optgroup_select_field import OptgroupSelectWidget, OptgroupSelectField, OptgroupSelectMultipleField

class SitesForm(FlaskForm):
  metric = SelectField("Metric", choices = [("total_kwh", "KwH"), ("kwh_psqm", "Kwh/SqM")], default = "total_kwh")
  chart_type = SelectField("Chart Type", choices = [(BAR , "Bar"), (LINE , "Line"), (COLUMN , "Column") ], default = COLUMN, validators = [validators.DataRequired()])
  site_ids = OptgroupSelectMultipleField("Site Ids", coerce = int)
  date_range = StringField("Date Range", default = "{} - {}".format(datetime.today() - timedelta(days = 7), datetime.today()))
  aggregate_unit = SelectField("Aggregate Period", coerce = int, choices = [(FIFTEEN_MINUTES, "15 Minutes"), (HOURLY, "Hourly"), (DAILY, "Daily")], default = DAILY)

  def __init__(self, *args, **kwargs):
    super(SitesForm, self).__init__(*args, **kwargs)
    device_records = Device.query.all()
    devices = [(d.id, d.device_id) for d in device_records]
    sizes, site_types = set([x.tag_size for x in device_records]), set([x.tag_site_type for x in device_records])
    cdict = { }
    for x in sizes:
      cdict[x] = tuple([(d.id, d.device_id) for d in device_records if d.tag_size == x])
    for x in site_types:
      cdict[x] = tuple([(d.id, d.device_id) for d in device_records if d.tag_site_type == x])

    self.site_ids.choices = tuple(cdict.items())
    if len(devices) > 0:
      random.shuffle(devices)
      defaults = devices[:min(5, len(devices))]
      self.site_ids.process_data([x[0] for x in defaults])

class TimerangesForm(SitesForm):
  date_range2 = StringField("Second Date Range", default = "{} - {}".format(datetime.today() - timedelta(days = 14), datetime.today() - timedelta(days = 7)))

class DeepDiveForm(FlaskForm):
  site_id = SelectField("Site Id", coerce = int)

  def __init__(self, *args, **kwargs):
    super(DeepDiveForm, self).__init__(*args, **kwargs)
    device_records = Device.query.all()
    devices = [(d.id, d.device_id) for d in device_records]
    self.site_id.choices = devices
    if len(devices) > 0:
      random.shuffle(devices)
      default = devices[0]
      self.site_id.default = default[0]

