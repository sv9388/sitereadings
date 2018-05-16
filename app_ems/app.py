from flask import request, render_template, Flask, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_restless import APIManager
from datetime import datetime, timedelta
import random, json

app = Flask(__name__)
app.config.from_object('settings.DevelopmentConfig')
db = SQLAlchemy(app)

from charts import * 
from energy_consumption import *
from readings_pred import *
from enums import *
from forms import *

error_chart = {"chart":{"renderTo":CHART_DIV_ID, "ignoreHiddenSeries" : False}, "title" : {"text" : "Fix errors to populate chart"}, "yAxis":{"labels":{}},"series":[{"data":[]}]}

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/powersines/compare_sites", methods = ["GET", "POST"])
def compare_sites():
  form = CustomSitesForm(request.form)
  metric, sites, date_range, aggperiod, ctype, date_range2 = form.metric.default, form.site_ids.default, form.date_range.default, form.aggregate_unit.default, form.chart_type.default, form.date_range2.default
  start_date, end_date = [datetime.strptime(x, "%Y-%m-%d") for x in date_range.split(" - ")]
  qtype = QUERY_MTIMERANGES if date_range2 and date_range2.strip() != "" else QUERY_MSITES
  start_date2, end_date2 = [datetime.strptime(x, "%Y-%m-%d") for x in date_range2.split(" - ")] if qtype == QUERY_MTIMERANGES else [None, None]

  if request.method == "GET":
    chart, errors = get_chart_data(sites, ctype, start_date, end_date, aggperiod, start_date2, end_date2, metric, qtype)
    return render_template("compare_sites.html", form  = form, chart = chart, errors = errors)

  if not form.validate():
    print(form.errors)
    return render_template("compare_sites.html", form  = form, chart = error_chart, errors = "There were some errors in your choices. Fix them to populate the chart")

  print(form.site_ids.data, form.chart_type.data, form.date_range.data, form.date_range2.data, form.metric.data, form.aggregate_unit.data)
  sites = form.site_ids.data
  ctype = form.chart_type.data
  date_range = form.date_range.data
  date_range2 = form.date_range2.data
  metric = form.metric.data
  aggperiod  = form.aggregate_unit.data
  start_date, end_date = [datetime.strptime(x, "%Y-%m-%d") for x in date_range.split(" - ")]
  qtype = QUERY_MTIMERANGES if date_range2 and date_range2.strip() not in ["Invalid date - Invalid date", ""] else QUERY_MSITES
  start_date2, end_date2 = [datetime.strptime(x, "%Y-%m-%d") for x in date_range2.split(" - ")] if qtype == QUERY_MTIMERANGES else [None, None]
  print(sites, ctype, date_range, date_range2, metric, aggperiod)
  chart, errors = get_chart_data(sites, ctype, start_date, end_date, aggperiod, start_date2, end_date2, metric, qtype)
  print(chart)
  return render_template("compare_sites.html", form  = form, chart = chart, errors = errors)

@app.route("/powersines/deep_dive_site", methods = ["GET", "POST"])
def deep_dive_site():
  form = DeepDiveForm()
  site_id = form.site_id.default
  if request.method == "POST":
    print(request.form)
    site_id = int(request.form["site_id"])
  print(site_id, type(site_id))
  chart_op, errors = get_ideal_energy_consumption_curves_chart(site_id)
  return render_template("ecmap.html", form = form, chart_op = chart_op, errors = errors)

@app.route("/powersines/predict", methods = ["GET", "POST"])
def predict_readings():
  form = PredictionForm(request.form)
  errors = ""
  site, predict_for, agg_period = form.site_id.default, form.predict_for_timerange.default, form.aggregate_unit.default

  if request.method == "GET":
    chart = get_predictions_chart(site, predict_for, agg_period)
    return render_template("predict.html", form  = form, chart = chart, errors = errors)

  if not form.validate():
    print(form.errors)
    return render_template("predict.html", form  = form, chart = error_chart, errors = errors)

  site, predict_for, agg_period = form.site_id.data, form.predict_for_timerange.data, form.aggregate_unit.data
  chart = get_predictions_chart(site, predict_for, agg_period)
  return render_template("predict.html", form  = form, chart = chart, errors = errors)
