from sqlalchemy import extract, func
from sqlalchemy.schema import UniqueConstraint, CheckConstraint
from sqlalchemy.orm import backref
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from app import db

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
