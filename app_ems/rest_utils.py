from collections import defaultdict
from werkzeug.utils import secure_filename
import os
from flask_restless import APIManager, ProcessingException
from app import app, db, csrf, current_user, request, jsonify
from models import *
import dateutil.parser

def hashp(*args, **kw):
  user = Besuser.query.filter_by(email = kw['result']['email']).first()
  if not user:
    raise ProcessingException(detail='Not Authorized', status=401)
  user.hash_password(kw['result']['password'])
  del kw['result']['password']
  db.session.add(user)
  db.session.commit()

api_manager = APIManager(app, flask_sqlalchemy_db = db)
bp = {}
bp['role'] = api_manager.create_api_blueprint(Role, methods = ["GET"])
bp['user'] = api_manager.create_api_blueprint(Emsuser, methods = ["GET", "POST", "PATCH", "DELETE"], postprocessors = {'POST' : [hashp]}, results_per_page = 0, max_results_per_page = 0)
bp['device'] = api_manager.create_api_blueprint(Device, methods = ["GET", "POST", "PATCH", "DELETE"], exclude_columns = ['weather', 'readings'], results_per_page = 0, max_results_per_page = 0)

for k, v in bp.items():
  csrf.exempt(v)
  app.register_blueprint(v)
