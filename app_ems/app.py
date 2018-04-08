from flask import request, render_template, Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_restless import APIManager
from datetime import datetime, timedelta
import random

app = Flask(__name__)
app.config.from_pyfile("./settings.py")
db = SQLAlchemy(app)

from charts import * 
from deep_dive import *
from enums import *
from forms import *

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/powersines/multiple_sites", methods = ["GET", "POST"])
def multiple_sites():
  form = SitesForm()
  sites, ctype, start_date, end_date, aggperiod, metric = form.site_ids.data, form.chart_type.default, datetime.today()-timedelta(days = 7), datetime.today(), form.aggregate_unit.default, form.metric.default
  print(type(sites[0]), aggperiod)

  if request.method == "POST":
    print(request.form)
    ctype = request.form['chart_type']
    daterange = request.form['date_range']
    metric = request.form['metric']
    aggperiod  = int(request.form['aggregate_unit'])
    sites = request.form.getlist('site_ids')
    start_date, end_date = [datetime.strptime(x, "%Y-%m-%d") for x in daterange.split(" - ")]

  sites = list(set([int(x) for x in sites]))
  print(start_date, end_date)
  chart, errors = get_chart_data(sites, ctype, start_date, end_date, aggperiod, metric = metric, qtype = QUERY_MSITES )
  return render_template("msites.html", form = form, chart = chart, thisurl = "multiple_sites", errors = errors)

@app.route("/powersines/two_timerange", methods = ["GET", "POST"])
def two_timeranges():
  form = TimerangesForm()
  sites, ctype, start_date, end_date, start_date2, end_date2, aggperiod, metric =  form.site_ids.data, form.chart_type.default, datetime.today() - timedelta(days = 7), datetime.today(), datetime.today() - timedelta(days = 14), datetime.today() - timedelta(days = 7), form.aggregate_unit.default, form.metric.default

  if request.method == "POST":
    print(request.form)
    sites = request.form.getlist('site_ids')
    daterange1 = request.form['date_range']
    daterange2 = request.form['date_range2']
    metric = request.form['metric']
    aggperiod  = int(request.form['aggregate_unit'])
    daterange = request.form['date_range']
    start_date, end_date = [datetime.strptime(x, "%Y-%m-%d") for x in daterange.split(" - ")]
    daterange2 = request.form['date_range2']
    start_date2, end_date2 = [datetime.strptime(x, "%Y-%m-%d") for x in daterange2.split(" - ")]

  chart, errors = get_chart_data(sites, ctype, start_date, end_date, aggperiod, start_date2, end_date2, metric, QUERY_MTIMERANGES)
  return render_template("mtimeranges.html", form  = form, chart = chart, thisurl = "two_timeranges", errors = errors)

@app.route("/powersines/deep_dive_site", methods = ["GET", "POST"])
def deep_dive_site():
  form = DeepDiveForm()
  site_id = form.site_id.default
  if request.method == "POST":
    print(request.form)
    site_id = int(request.form["site_id"])
  print(site_id, type(site_id))
  chart, errors = get_dd_chart_data(site_id)
  return render_template("deepdive.html", form = form, chart = chart, thisurl = "deep_dive_site", errors = errors)

