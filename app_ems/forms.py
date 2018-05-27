from wtforms import SelectField, DecimalField, SelectMultipleField, Form, validators, SubmitField, DateTimeField, StringField, SelectField, PasswordField, HiddenField, TextField
from flask_wtf.file import FileField, FileRequired
from flask_wtf import FlaskForm
from enums import *
from models import Device, Emsuser
from datetime import datetime, timedelta
import random, re
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
  aggregate_unit = SelectField("Aggregate Period", [validators.Optional()], coerce = int, choices = [(HOURLY, "Hourly")], default = HOURLY)
 
  def __init__(self, *args, **kwargs):
    super(PredictionForm, self).__init__(*args, **kwargs)
    device_records = Device.query.all()
    devices = [(d.id, d.device_id) for d in device_records]
    self.site_id.choices = devices
    self.site_id.default = 1

DEFAULT_DATESTR = "{} - {}".format((datetime.today() - timedelta(days = 7)).strftime("%Y-%m-%d"), datetime.today().strftime("%Y-%m-%d"))
class FormulaCheckerForm(Form):
  metric_formula = TextField("Formula", default = "KwH", render_kw = {'data-toggle': 'tooltip', 'title' : "Check Help -> Formulae to get more info on formulae and verify them."})
  kwh =  DecimalField("Kwh value", default = 3.0)
  temperature = DecimalField("Temperature value", default = 3)
  sqm =  DecimalField("Sqm value", default = 3)
  customvar = DecimalField("Custom variable's value", default = 3, render_kw = {'data-toggle': 'tooltip', 'title' : "Enter ONE value of the custom variable here. "})
  computed_value = DecimalField('Computed Value', render_kw={'disabled': True})

class CustomSitesForm(Form):
  metric_formula = TextField("Metric Formula", default = "KwH", render_kw = {'data-toggle': 'tooltip', 'title' : "Check Help -> Formulae to get more info on formulae and verify them."})
  param_file = FileField("Upload Custom File ")
  chart_type = SelectField("Chart Type", [validators.DataRequired()],  choices = [(BAR , "Bar"), (LINE , "Line"), (COLUMN , "Column") ], default = COLUMN)
  site_ids = SelectMultipleField("Site Ids", [validators.Optional()], coerce = int, default = [1, 2, 3, 4, 5])
  tag1 = SelectField("First Tag", [validators.Optional()], coerce = str)
  tag2 = SelectField("Second Tag", [validators.Optional()], coerce = str)
  date_range = StringField("Date Range", [validators.DataRequired()], default = DEFAULT_DATESTR)
  date_range2 = StringField("Second Date Range", [validators.Optional()], default = "")
  aggregate_unit = SelectField("Aggregate Period", [validators.Optional()], coerce = int, choices = [(HOURLY, "Hourly"), (DAILY, "Daily"), (WEEKLY, "Weekly"), (MONTHLY, "Monthly"), (YEARLY, "Yearly")], default = DAILY)

  def __init__(self, *args, **kwargs):
    super(CustomSitesForm, self).__init__(*args, **kwargs)
    device_records = Device.query.all()
    devices = [(d.id, d.device_id) for d in device_records]
    self.site_ids.choices = devices
    tags = list(set([d.tag_size for d in device_records])) + list(set([d.tag_site_type for d in device_records]))
    tags = [(x if x else "None", x if x else "Untagged") for x in tags]
    self.tag1.choices = tags
    self.tag2.choices = tags

  def validate(self):
    rv = Form.validate(self)
    if not rv:
      return False

    pattern = re.compile("[a-zA-Z]+")
    reqd_vars = set(['sqrt', 'power', 'kwh', 'temperature', 'sqm', 'customvar'])
    got_vars = set(pattern.findall(self.metric_formula.data.lower()))
    if not got_vars.issubset(reqd_vars):
      self.metric_formula.errors.append("Allowed variables are kwh, sqm, temperature and customvar. Check Help page for more details")
      return False

    if "customvar" in self.metric_formula.data.lower() and not self.param_file.data:
      self.param_file.errors.append("If you have a custom variable in the formula, upload a CSV file with selected device ids mapped to the custom parameter")
      return False
    return True

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

