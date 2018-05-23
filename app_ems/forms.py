from wtforms import SelectField, Form, validators, SubmitField, DateTimeField, StringField, SelectField, PasswordField
from flask_wtf import FlaskForm
from enums import *
from models import Device, Emsuser
from datetime import datetime, timedelta
import random
from optgroup_select_field import OptgroupSelectWidget, OptgroupSelectField, OptgroupSelectMultipleField
from login_utils import current_user

class LoginForm(FlaskForm):
  email = StringField('Email', validators = [validators.DataRequired()], render_kw = {'placeholder' : "Email"})
  password = PasswordField('Password', validators = [validators.DataRequired()], render_kw = {'placeholder' : "Password"})

class EditProfileForm(Form):
  email = StringField('Email', render_kw={'readonly': True})
  password = PasswordField('New Password', [
        validators.EqualTo('confirm_password', message='Passwords must match')
    ])
  confirm_password = PasswordField('Repeat Password')
  name = StringField('First Name')
  surname = StringField('Surname')
  role = StringField('Role', render_kw={'readonly': True})

  def __init__(self, *args, **kwargs):
    super(EditProfileForm, self).__init__(*args, **kwargs)
    prf_user = None
    print(current_user.is_authenticated(), current_user.get_id())
    if current_user.is_authenticated():
        email = current_user.get_id() # return username in get_id()
        prf_user = Emsuser.query.filter(Emsuser.email == email).first()
        print(prf_user)
        self.email.data = prf_user.email
        self.name.data = prf_user.name
        self.surname.data = prf_user.surname
        self.role.data = prf_user.role.name
    print(self.email.data, self.email.default)

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
  site_ids = OptgroupSelectMultipleField("Site Ids", [validators.Optional()], coerce = int, default = [1, 2, 3, 4, 5])
  tag1 = SelectField("First Tag", [validators.Optional()], coerce = str)
  tag2 = SelectField("Second Tag", [validators.Optional()], coerce = str)
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
    tags = list(cdict.keys())
    tags = [(x if x else "None", x if x else "Untagged") for x in tags]
    self.tag1.choices = tags
    self.tag2.choices = tags

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

