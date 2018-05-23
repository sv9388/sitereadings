from flask import request, render_template, Flask, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from datetime import datetime, timedelta
import random, json

app = Flask(__name__)
app.config.from_object('settings.DevelopmentConfig')
db = SQLAlchemy(app)
csrf = CSRFProtect(app)

from login_utils import *
from rest_utils import * 
from charts import * 
from energy_consumption import *
from heatmaps import *
from readings_pred import *
from enums import *
from forms import *

error_chart = {"chart":{"renderTo":CHART_DIV_ID, "ignoreHiddenSeries" : False}, "title" : {"text" : "Fix errors to populate chart"}, "yAxis":{"labels":{}},"series":[{"data":[]}]}

@app.route("/manage/devices")
@login_required
def manage_devices():
  return render_template("manage_devices.html", user = load_user(current_user.get_id()))

@app.route("/manage/users")
@login_required
def manage_users():
  return render_template("manage_users.html", user = load_user(current_user.get_id()))


@app.route('/profile', methods = ["GET", "POST"])
@login_required
def my_account():
  form = EditProfileForm()
  if request.method == "GET":
    return render_template('profile.html', form = form, user = load_user(current_user.get_id()))

  if not form.validate():
    return render_template('profile.html', form = form, user = load_user(current_user.get_id()), notif = ["error", "Fix the errors below to proceed"])

  user = load_user(current_user.get_id())
  if not user:
    return render_template('http_error.html', error_code = 401, error_detail = 'User not found. Please retry with correct details')

  print(user.email, form.email.data)
  if user.email != form.email.data:
    return render_template('http_error.html', error_code = 401, error_detail = 'You can only update your details here') 

  user.name = form.name.data
  user.surname = form.name.data
  if request.form['password'] != '':
    if request.form['password'] != request.form['confirm_password']:
      return render_template('profile.html', form = form, user = load_user(current_user.get_id()), notif = ['error', 'Passwords didn\'t match'])
    user.hash_password(request.form['password'])

  print("Are we here?", user) 
  db.session.add(user)
  db.session.commit()
  return render_template('profile.html', form = form, user = load_user(current_user.get_id()), notif = ['success', "Details updated successfully"])

@app.route("/powersines/compare_sites", methods = ["GET", "POST"])
@login_required
def compare_sites():
  form = CustomSitesForm(request.form)
  metric, sites, date_range, aggperiod, ctype, date_range2 = form.metric.default, form.site_ids.default, form.date_range.default, form.aggregate_unit.default, form.chart_type.default, form.date_range2.default
  start_date, end_date = [datetime.strptime(x, "%Y-%m-%d") for x in date_range.split(" - ")]
  qtype = QUERY_MTIMERANGES if date_range2 and date_range2.strip() != "" else QUERY_MSITES
  data_type = SITE if (form.tag1.data == "None" and form.tag2.data == "None") else TAG
  if qtype == QUERY_MTIMERANGES and not (form.tag1.data == "" and form.tag2.data == ""):
    return render_template("compare_sites.html", user = load_user(current_user.get_id()), form = form, chart = chart, notif = ["error" , "Daterange comparison and tag comparison cannot be chosen together"])

  start_date2, end_date2 = [datetime.strptime(x, "%Y-%m-%d") for x in date_range2.split(" - ")] if qtype == QUERY_MTIMERANGES else [None, None]
  tag1 = form.tag1.default
  tag2 = form.tag2.default
  if request.method == "GET":
    chart, errors = get_chart_data(sites, ctype, start_date, end_date, aggperiod, start_date2, end_date2, tag1, tag2, metric, qtype)
    notif = ["warning", errors] if errors and len(errors)>0 else ["success", "Fetched Data"]
    return render_template("compare_sites.html", user = load_user(current_user.get_id()), form = form, chart = chart, notif = notif)

  if not form.validate():
    print(form.errors)
    return render_template("compare_sites.html", user = load_user(current_user.get_id()), form = form, chart = error_chart, notif = ["warning", "There were some errors in your choices. Fix them to populate the chart"])

  print(form.site_ids.data, form.chart_type.data, form.date_range.data, form.date_range2.data, form.tag1.data, form.tag2.data, form.metric.data, form.aggregate_unit.data)
  sites = form.site_ids.data
  ctype = form.chart_type.data
  date_range = form.date_range.data
  date_range2 = form.date_range2.data
  metric = form.metric.data
  aggperiod  = form.aggregate_unit.data
  start_date, end_date = [datetime.strptime(x, "%Y-%m-%d") for x in date_range.split(" - ")]
  qtype = QUERY_MTIMERANGES if date_range2 and date_range2.strip() not in ["Invalid date - Invalid date", ""] else QUERY_MSITES
  start_date2, end_date2 = [datetime.strptime(x, "%Y-%m-%d") for x in date_range2.split(" - ")] if qtype == QUERY_MTIMERANGES else [None, None]
  tag1 = form.tag1.data
  tag2 = form.tag2.data
  print(sites, ctype, date_range, date_range2, tag1, tag2, metric, aggperiod)
  chart, errors = get_chart_data(sites, ctype, start_date, end_date, aggperiod, start_date2, end_date2, tag1, tag2, metric, qtype, data_type)
  notif = ["warning", errors] if errors and len(errors)>0 else ["success", "Fetched Data"]
  return render_template("compare_sites.html", user = load_user(current_user.get_id()), form = form, chart = chart, notif = notif)

@app.route("/powersines/deep_dive_site", methods = ["GET", "POST"])
@login_required
def deep_dive_site():
  form = DeepDiveForm()
  site_id = form.site_id.default
  if request.method == "POST":
    print(request.form)
    site_id = int(request.form["site_id"])
  print(site_id, type(site_id))
  chart_op, errors = get_dd_chart_data(site_id)
  notif = ["warning", errors] if errors and len(errors)>0 else ["success", "Fetched Data"]
  return render_template("heatmap.html", user = load_user(current_user.get_id()), form = form, chart_op = chart_op, notif = notif)

@app.route("/powersines/energy_consumption", methods = ["GET", "POST"])
@login_required
def energy_consumption():
  form = DeepDiveForm()
  site_id = form.site_id.default
  if request.method == "POST":
    print(request.form)
    site_id = int(request.form["site_id"])
  print(site_id, type(site_id))
  chart_op, errors = get_ideal_energy_consumption_curves_chart(site_id)
  notif = ["warning", errors] if errors and len(errors)>0 else ["success", "Fetched Data"]
  return render_template("ecmap.html", user = load_user(current_user.get_id()), form = form, chart_op = chart_op, notif = notif)

@app.route("/powersines/predict", methods = ["GET", "POST"])
@login_required
def predict_readings():
  form = PredictionForm(request.form)
  errors = ""
  site, predict_for, agg_period = form.site_id.default, form.predict_for_timerange.default, form.aggregate_unit.default

  if request.method == "GET":
    chart = get_predictions_chart(site, predict_for, agg_period)
    notif = ["warning", errors] if errors and len(errors)>0 else ["success", "Fetched Data"]
    return render_template("predict.html", user = load_user(current_user.get_id()), form = form, chart = chart, notif = ["warning", "Fix form errors before proceeding"])

  if not form.validate():
    print(form.errors)
    return render_template("predict.html", user = load_user(current_user.get_id()), form = form, chart = error_chart, notif = ["warning", "Fix form errors"])

  site, predict_for, agg_period = form.site_id.data, form.predict_for_timerange.data, form.aggregate_unit.data
  chart = get_predictions_chart(site, predict_for, agg_period)
  notif = ["warning", errors] if errors and len(errors)>0 else ["success", "Fetched Data"]
  return render_template("predict.html", user = load_user(current_user.get_id()), form = form, chart = chart, notif = notif)
