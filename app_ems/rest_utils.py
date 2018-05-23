from collections import defaultdict
from werkzeug.utils import secure_filename
import os
from flask_restless import APIManager, ProcessingException
from flask_restful import Resource, Api

from app import app, db, csrf, current_user, request, jsonify
from models import *
import dateutil.parser

def confirm_password(*args, **kw):
  print("PRE", request.method, args, kw)
  data = kw['data']
  if data['password'] != data['confirm_password']:
    raise ProcessingException(detail="Passwords don't match", status=400)
  del data['confirm_password']
  if request.method == "PATCH" and data['password'] == '':
    del data['password']
  data['role'] = { 'id' : int(data['role'])}
  print("PRE", request.method, data)
  kw['data'] = data

def hashp(*args, **kw):
  print("HASP", kw)
  if not 'password' in request.json.keys():
    return 

  print("HASHP", request.json, request.json.keys())
  user = Emsuser.query.filter_by(email = kw['result']['email']).first()
  if not user:
    raise ProcessingException(detail='Not Authorized', status=401)
  user.hash_password(kw['result']['password'])
  del kw['result']['password']
  db.session.add(user)
  db.session.commit()

def remove_password(*args, **kw):
  print(args, kw)
  if 'objects' in kw['result'].keys():
    for i in range(len(kw['result']['objects'])):
      del kw['result']['objects'][i]['password']
  else:
    del kw['result']['password']

api_manager = APIManager(app, flask_sqlalchemy_db = db)
bp = {}
bp['role'] = api_manager.create_api_blueprint(Role, methods = ["GET"])
bp['user'] = api_manager.create_api_blueprint(Emsuser, methods = ["GET", "POST", "PATCH", "DELETE"], preprocessors = {"POST" : [confirm_password], "PATCH_SINGLE" : [confirm_password]}, postprocessors = {'GET_SINGLE' : [remove_password], 'GET_MANY' : [remove_password], 'POST' : [hashp], 'PATCH_SINGLE' : [hashp]}, results_per_page = 0, max_results_per_page = 0) #, include_columns = ["id", "email", "name", "surname", "role", "devices.id", "devices.device_id", "devices.country", "devices.distributer_name", "devices.latitude", "devices.longitude", "devices.project", "devices.sqm", "devices.tag_site_type", "devices.tag_size", "devices.is_active", "password"])
bp['device'] = api_manager.create_api_blueprint(Device, methods = ["GET", "POST", "PATCH", "DELETE"], exclude_columns = ['weather', 'readings'], results_per_page = 0, max_results_per_page = 0)

for k, v in bp.items():
  csrf.exempt(v)
  app.register_blueprint(v)

api = Api(app, decorators=[csrf.exempt])
class DeviceTree(Resource):
  def get(self):
    devices = Device.query.all()
    hierarchy = {}
    for d in devices:
      system_name = d.system_name if d.system_name else "None"
      if d.distributer_name in hierarchy.keys():
        if d.project in hierarchy[d.distributer_name].keys():
          if system_name in hierarchy[d.distributer_name][d.project].keys():
            hierarchy[d.distributer_name][d.project][system_name].append(d.device_id)
          else:
            hierarchy[d.distributer_name][d.project][system_name] =  [d.device_id]
        else:
          hierarchy[d.distributer_name][d.project] = {system_name : [d.device_id]}
      else:
         hierarchy[d.distributer_name] = {d.project : {system_name : [d.device_id]}}

    tag_sizes = {}
    for d in devices:
      tag_size = d.tag_size if d.tag_size and len(d.tag_size)>0 else "Untagged"
      if tag_size in tag_sizes.keys():
        tag_sizes[tag_size].append(d.device_id)
      else:
        tag_sizes[tag_size] = [d.device_id]

    tag_site_types = {}
    for d in devices:
      tag_site_type = d.tag_site_type if d.tag_site_type and len(d.tag_site_type)>0 else "Untagged"
      if tag_site_type in tag_site_types.keys():
        tag_site_types[tag_site_type].append(d.device_id)
      else:
        tag_site_types[tag_site_type] = [d.device_id]

    return {"Structured" : hierarchy, "Unstructured" : {"Tag: Size" : tag_sizes, "Tag: Site Types" : tag_site_types}}
   
api.add_resource(DeviceTree, "/api/devicetree") 
