from flask import request, render_template, Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restless import APIManager

app = Flask(__name__)
app.config.from_pyfile("./settings.py")
db = SQLAlchemy(app)

from forms import ReadingForm
from charts import * #get_kwh_data
from enums import *

@app.route("/powersines/<string:attribute>", methods = ["GET", "POST"])
def ems_data(attribute = "kwh"):
  form = ReadingForm()
  sites, ctype, cperiod, aggperiod, metric = [1, 2, 3], BAR, LAST_7_DAYS, FIFTEEN_MINUTES, "kwh"
  if request.method == "POST":
    print(request.form)
    ctype = request.form['chart_type']
    cperiod = int(request.form['period'])
    aggperiod  = int(request.form['aggregate_unit'])
    sites = request.form.getlist('site_ids')
    sites = [int(x) for x in sites]
  print(sites)
  chart = get_chart_data(sites, ctype, cperiod, aggperiod, metric)
  return render_template("chart.html", form  = form, chart = chart, attribute = attribute)
