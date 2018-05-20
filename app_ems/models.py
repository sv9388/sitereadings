from flask_security import RoleMixin, UserMixin
from sqlalchemy import extract, func
from sqlalchemy.schema import UniqueConstraint, CheckConstraint
from sqlalchemy.orm import backref
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from app import db

class Role(db.Model, RoleMixin):
  id = db.Column(db.Integer(), primary_key=True)
  name = db.Column(db.String(80), unique=True)
  def __str__(self):
    return self.name

  def __hash__(self):
    return hash(self.name)


class Emsuser(db.Model):
  def hash_password(self, password):
    self.password = pwd_context.encrypt(password)

  def verify_password(self, password):
    return pwd_context.verify(password, self.password)

  def generate_auth_token(self, expiration=3600):
    s = Serializer(app.config['SECRET_KEY'], expires_in=expiration)
    return s.dumps({'id': self.id})

  def is_active(self):
    return True

  def is_authenticated(self):
    return True

  def is_anonymous(self):
    return False

  def get_id(self):
    return str(self.email)

  @staticmethod
  def verify_auth_token(token):
    s = Serializer(app.config['SECRET_KEY'])
    data = None
    try:
      data = s.loads(token)
    except SignatureExpired:
      return None  # valid token, but expired
    except BadSignature:
      return None  # invalid token
    user = Besuser.query.get(data['id'])
    return user

  id = db.Column(db.Integer, primary_key=True)
  email = db.Column(db.String(255), unique=True, nullable = False)
  password = db.Column(db.String(130))
  name = db.Column(db.String(), nullable = False)
  surname = db.Column(db.String())
  role_id = db.Column(db.Integer, db.ForeignKey("role.id"), nullable = False)
  role = db.relationship('Role', backref = db.backref('emsuser', lazy = 'joined', cascade = "all, delete-orphan"))

class Device(db.Model):
  __tablename__ = "device"
  id = db.Column(db.Integer, primary_key = True)
  device_id = db.Column(db.String(10), nullable = False, unique = True) 
  distributer_name = db.Column(db.String()) #, nullable = False)
  project = db.Column(db.String()) #, nullable = False)
  system_name = db.Column(db.String()) #, nullable = False)
  is_active = db.Column(db.Boolean) #, nullable = False)
  country = db.Column(db.String())#, nullable = False)
  tag_site_type = db.Column(db.String()) 
  tag_size = db.Column(db.String())
  sqm = db.Column(db.Integer) #, nullable = False)
  latitude = db.Column(db.Float)
  longitude = db.Column(db.Float)
  user_id = db.Column(db.Integer, db.ForeignKey("emsuser.id"))
  user = db.relationship('Emsuser', backref = db.backref("devices", cascade="all,delete"), lazy = True)
  __table_args__ = (CheckConstraint('sqm > 0', name='Sq/m must be positive'),)

class Reading(db.Model):
  __tablename__ = "reading"
  id = db.Column(db.Integer, primary_key = True)
  rdate= db.Column(db.DateTime, nullable = False)
  total_power_kw = db.Column(db.Float, nullable = False)
  total_kwh = db.Column(db.Float, nullable = False)
  device_id = db.Column(db.Integer, db.ForeignKey("device.id"))
  device_assosciated = db.relationship("Device", backref = db.backref('readings')) #, lazy = 'joined', cascade = "all, delete-orphan"), uselist = False)
  __table_args__ = (UniqueConstraint('rdate', 'device_id', name='_date_device_uc'), )


class Weather(db.Model):
  __tablename__ = "weather"
  id = db.Column(db.Integer, primary_key = True)
  wdate= db.Column(db.DateTime, nullable = False)
  temp = db.Column(db.Float, nullable = False)
  device_id = db.Column(db.Integer, db.ForeignKey("device.id"))
  device_assosciated = db.relationship("Device", backref = db.backref('weather')) #, lazy = 'joined', cascade = "all, delete-orphan"), uselist = False)
  __table_args__ = (UniqueConstraint('wdate', 'device_id', name='_date_device_weather_uc'), )

class CrawlLimits(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  table_key = db.Column(db.String(10), nullable = False, unique = True)
  date_limit = db.Column(db.DateTime, nullable = False)
