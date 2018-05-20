from wtforms import SelectField, Form, validators, SubmitField, DateTimeField, StringField, SelectField, PasswordField
from flask_wtf import FlaskForm
from enums import *
from models import Device
from datetime import datetime, timedelta
import random
from optgroup_select_field import OptgroupSelectWidget, OptgroupSelectField, OptgroupSelectMultipleField
from login_utils import current_user

class LoginForm(FlaskForm):
  email = StringField('Email', validators = [validators.DataRequired()])
  password = PasswordField('Password', validators = [validators.DataRequired()])

class EditProfileForm(FlaskForm):
  email = StringField('Email', render_kw={'readonly': True})
  password = PasswordField('New Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm_password', message='Passwords must match')
    ])
  confirm_password = PasswordField('Repeat Password')
  name = StringField('First Name')
  surname = StringField('Surname')
  role = StringField('Role', render_kw={'readonly': True})

  def __init__(self, *args, **kwargs):
    super(EditProfileForm, self).__init__(*args, **kwargs)
    prf_user = None
    if current_user.is_authenticated:
        prf_user = current_user.get_id() # return username in get_id()
        print(prf_user)
        self.email.default = prf_user.email
        self.name.default = prf_user.name
        self.surname.default = prf_user.surname
        self.role.default = prf_user.role.name

class PredictionForm(Form):
  #site_ids = OptgroupSelectMultipleField("Site Ids", [validators.DataRequired()], coerce = int)
  site_id = SelectField("Site Id", [validators.DataRequired()], coerce = int)
  predict_for_timerange = SelectField("Predict values for", choices =  [(WEEK, "This week"), (DAY, "Today")], coerce = int, default = DAY)
  aggregate_unit = SelectField("Aggregate Period", [validators.Optional()], coerce = int, choices = [(FIFTEEN_MINUTES, "15 Minutes"), (HOURLY, "Hourly")], default = HOURLY)
 
  def __init__(self, *args, **kwargs):
    super(PredictionForm, self).__init__(*args, **kwargs)
    device_records = Device.query.all()
    devices = [(d.id, d.device_id) for d in device_records]
    self.site_id.choices = devices
    self.site_id.default = 1

DEFAULT_DATESTR = "{} - {}".format((datetime.today() - timedelta(days = 7)).strftime("%Y-%m-%d"), datetime.today().strftime("%Y-%m-%d"))
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

