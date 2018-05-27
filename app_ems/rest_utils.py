from collections import defaultdict
from werkzeug.utils import secure_filename
import os, re, numpy as np
from flask_restful import Resource, Api

from app import app, db, csrf, current_user, load_user, request, jsonify
from models import * 
import dateutil.parser

def get_user_json(user, add_device = False):
  op = {"id" : user.id, "email" : user.email, "name" : user.name, "surname" : user.surname, "role_id" : user.role_id, "role" : {"id" : user.role.id, "name" : user.role.name}}
  if not add_device:
    return op
  devices = user.devices
  devices_op = [get_device_json(device, add_user = False) for device in devices]
  op['devices'] = devices_op
  return op

def get_device_json(device, add_user = False):
  op  = {"country" : device.country, "device_id" : device.device_id, "device_unique_name" : device.device_unique_name, "distributer_name" : device.distributer_name, "id" : device.id, "is_active" : device.is_active, "latitude" : device.latitude, "longitude" : device.longitude, "project" : device.project, "sqm" : device.sqm, "system_name" : device.system_name, "tag_site_type" : device.tag_site_type, "tag_size" : device.tag_size}
  if not add_user:
    return op

  users = device.users
  users_op = [get_user_json(user, add_device = False) for user in users]
  op['users'] = users_op
  return op

api = Api(app, decorators=[csrf.exempt])

class RoleApi(Resource):
  def get(self):
    roles = Role.query.all()
    op = [{'id' : x.id, 'name' : x.name} for x in roles]
    return jsonify(num_results = len(op), page = 1, total_pages = 1, objects = op)

class EmsuserApi(Resource):
  def get(self, device_id = None, emsuser_id = None):
    if emsuser_id:
      user = Emsuser.query.get(emsuser_id)
      return jsonify(get_user_json(user, add_device = True))

    users = Emsuser.query.all()
    print(device_id, type(device_id))
    add_device = True
    if device_id:
      add_device = False
      device = Device.query.get(device_id)
      users = Emsuser.query.with_parent(device).all()
      print(users[0].devices)
    op = [get_user_json(user, add_device = True) for user in users]
    return jsonify(num_results = len(op), page = 1, total_pages = 1, objects = op)

  def post(self):
    data = request.json
    print(data, request.form)
    pwd = data['password'] # plain
    if pwd == '' or data['name'] == "" or data['email'] == "" or not data['role']:
      return jsonify(errors = "All fields are required!")
    if pwd != data['confirm_password']:
      return jsonify(errors = "Passwords didn't match")
 
    role = Role.query.get(int(data['role']))
    data = { x : data[x] for x in data if 'password' not in x and 'role' not in x}
    user = Emsuser(**data)
    user.hash_password(pwd)
    user.role = role
    db.session.add(user)
    db.session.commit()
    return jsonify(get_user_json(user, add_device = True))

  def put(self, emsuser_id):
    if not emsuser_id:
      return jsonify(errors = "Bulk edits are not allowed")

    user = Emsuser.query.get(emsuser_id)
    data = request.json
    pwd = data['password'] # plain
    if pwd != data['confirm_password']:
      return jsonify(errors = "Passwords didn't match")

    role_id = int(data["role"]) if "role" in data.keys() else None
    data = { x : data[x] for x in data if 'password' not in x and 'role' not in x}
    user.name = data['name']
    user.surname = data['surname']
    if pwd != '':
      user.hash_password(pwd)
    if role_id:
      role = Role.query.get(role_id)
      user.role = role
    db.session.add(user)
    db.session.commit()
    return jsonify(get_user_json(user, add_device = True))

  def delete(self, emsuser_id):
    user = Emsuser.query.get(emsuser_id)
    db.session.delete(user)
    db.session.commit()
    return 


class DeviceApi(Resource):
  def get(self, device_id = None, emsuser_id = None):
    if device_id:
      device = Device.query.get(device_id)
      return jsonify(get_device_json(device, add_user = True))

    devices = Device.query.all()
    add_user = True
    if emsuser_id :
      emsuser = Emsuser.query.get(emsuser_id)
      devices = Device.query.with_parent(emsuser).all()
      print(devices[0].users)
    op = [get_device_json(device, add_user = True) for device in devices]
    return jsonify(num_results = len(op), page = 1, total_pages = 1, objects = op)

  def post(self):
    data = request.json
    new_users = [Emsuser.query.get(user_id["id"]) for user_id in data['users']] if 'emsusers' in data.keys() else None
    data = { x : data[x] for x in data if 'users' not in x }
    print(data)
    device = Device(**data)
    if new_users:
      for u in new_users:
        device.users.append(u)
    db.session.add(device)
    db.session.commit()
    return jsonify(get_device_json(device, add_user = True))

  def put(self, device_id):
    if not device_id:
      return jsonify(errors = "Bulk edits are not allowed")

    device = Device.query.get(device_id)
    data = request.json
    print(data)
    new_users = [Emsuser.query.get(user_id['id']) for user_id in data['users']] if 'users' in data.keys() else None
    print(new_users)
    device.country = data['country']
    device.device_unique_name = data['device_unique_name']
    device.distributer_name = data['distributer_name']
    device.latitude = data['latitude']
    device.longitude = data['longitude']
    device.project = data['project']
    device.sqm = data['sqm']
    device.system_name = data['system_name']
    device.tag_site_type = data['tag_site_type']
    device.tag_size = data['tag_size']
    if new_users:
      old_users = device.users
      for u in old_users:
        device.users.remove(u)
      for u in new_users:
        device.users.append(u)
    db.session.add(device)
    db.session.commit()
    return jsonify(get_device_json(device, add_user = True))

  def delete(self, device_id):
    device = Device.query.get(device_id)
    db.session.delete(device)
    db.session.commit()
    return


class DeviceTree(Resource):
  def get(self):
    q = Device.query
    user = load_user(current_user.get_id())
    if user.role.id != 1:
      q = q.filter_by(user_id = user.id)
    devices = q.all()
    hierarchy = {}
    for d in devices:
      system_name = d.system_name if d.system_name else "None"
      if d.distributer_name in hierarchy.keys():
        if d.project in hierarchy[d.distributer_name].keys():
          if system_name in hierarchy[d.distributer_name][d.project].keys():
            hierarchy[d.distributer_name][d.project][system_name].append({'id' : d.id, 'value' : d.device_unique_name})
          else:
            hierarchy[d.distributer_name][d.project][system_name] =  [{'id' : d.id, 'value' : d.device_unique_name}]
        else:
          hierarchy[d.distributer_name][d.project] = {system_name : [{'id' : d.id, 'value' : d.device_unique_name}]}
      else:
         hierarchy[d.distributer_name] = {d.project : {system_name : [{'id' : d.id, 'value' : d.device_unique_name}]}}

    tag_sizes = {}
    for d in devices:
      tag_size = d.tag_size if d.tag_size and len(d.tag_size)>0 else "Untagged"
      if tag_size in tag_sizes.keys():
        tag_sizes[tag_size].append({'id' : d.id, 'value' : d.device_unique_name})
      else:
        tag_sizes[tag_size] = [{'id' : d.id, 'value' : d.device_unique_name}]

    tag_site_types = {}
    for d in devices:
      tag_site_type = d.tag_site_type if d.tag_site_type and len(d.tag_site_type)>0 else "Untagged"
      if tag_site_type in tag_site_types.keys():
        tag_site_types[tag_site_type].append({'id' : d.id, 'value' : d.device_unique_name})
      else:
        tag_site_types[tag_site_type] = [{'id' : d.id, 'value' : d.device_unique_name}]

    return {"Structured" : hierarchy, "Unstructured" : {"Tag: Size" : tag_sizes, "Tag: Site Types" : tag_site_types}}

class FormulaConfirmer(Resource):   
  def post(self):
    data = request.json
    print(data, type(data))
    metric_formula = data['metric_formula'].lower()
    pattern = re.compile("[a-zA-Z]+")
    reqd_vars = set(['sqrt', 'power', 'kwh', 'temperature', 'sqm', 'customvar'])
    got_vars = set(pattern.findall(metric_formula))
    if not got_vars.issubset(reqd_vars):
      return {"errors" : "Allowed variables are kwh, sqm, temperature and customvar. Check Help page for more details"}
    kwh = data['kwh']
    temperature = data['temperature']
    sqm = data['sqm']
    customvar = data['customvar']
    op = None
    try:
      for x in list(got_vars):
        if x in ["power", "sqrt"]:
          metric_formula = metric_formula.replace(x, "np.{}".format(x)) 
      op = eval(metric_formula)
    except SyntaxError:
      return {"errors" : "Your formula had a syntax error. Check for closed brackets, missed signs, etc"}
    except ValueError:
      return {"errors" : "Your formula had a syntax error. Check for closed brackets, missed signs, etc"}
    except NameError as e:
      e = str(e)
      error = e[e.find("'"):e.rfind("'")]
      return {"errors" : "Operation {}' not supported".format(error)}
    return {"computed_value" : round(op, 2), "errors" : ""}

api.add_resource(RoleApi, "/api/role") 
api.add_resource(EmsuserApi, "/api/emsuser", "/api/device/<int:device_id>/emsusers", "/api/emsuser/<int:emsuser_id>")
api.add_resource(DeviceApi, "/api/device", "/api/emsuser/<int:emsuser_id>/devices", "/api/device/<int:device_id>")
api.add_resource(DeviceTree, "/api/devicetree") 
api.add_resource(FormulaConfirmer, "/api/formula/confirm")
