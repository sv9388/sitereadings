from wtforms import SelectField, Form, validators, SubmitField, SelectMultipleField
from flask_wtf import FlaskForm
from enums import *
from models import Device

class ReadingForm(FlaskForm):
  chart_type = SelectField("Chart Type", choices = [(BAR , "Bar"), (LINE , "Line"), (COLUMN , "Column"), (PIE , "Pie Chart")], default = BAR, validators = [validators.DataRequired()])
  site_ids = SelectMultipleField("Site Ids")
  period = SelectField("Period", coerce = int, choices = [(LAST_7_DAYS, "Last 7 days"), (LAST_30_DAYS, "Last 30 days"), (CURRENT_MONTH, "Current Month"), (LAST_12_MONTHS, "Last 12 Months"), (CURRENT_YEAR, "Current Year"), (ALL_DATA, "All Data")], default = "last7days")
  aggregate_unit = SelectField("Aggregate Period", coerce = int, choices = [(FIFTEEN_MINUTES, "15 Minutes"), (HOURLY, "Hourly"), (DAILY, "Daily")])

  def __init__(self, *args, **kwargs):
    super(ReadingForm, self).__init__(*args, **kwargs)
    devices = Device.query.all()
    devices = [(d.id, d.device_id) for d in devices]
    self.site_ids.choices = devices
    if len(devices) > 0:
      self.site_ids.default = devices[0][0]
