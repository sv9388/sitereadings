from app import app, db
from flask_restless import APIManager
from models import Device, Reading
manager = APIManager(app, flask_sqlalchemy_db = db)
manager.create_api(Device, methods = ["GET"])
manager.create_api(Reading, methods = ["GET"])

