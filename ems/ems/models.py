from sqlalchemy import Integer, ForeignKey, String, Column, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.schema import UniqueConstraint
Base = declarative_base()

class Device(Base):
  __tablename__ = "device"
  id = Column(Integer, primary_key = True)
  device_id = Column(String(10), nullable = False, unique = True) 

class Reading(Base):
  __tablename__ = "reading"
  id = Column(Integer, primary_key = True)
  rdate= Column(DateTime, nullable = False)
  total_power_kw = Column(Float, nullable = False)
  total_kwh = Column(Float, nullable = False)
  device_id = Column(Integer, ForeignKey("device.id"))
  readings_for_device = relationship("Device", cascade = "all, delete-orphan", single_parent = True)
  __table_args__ = (UniqueConstraint('rdate', 'device_id', name='_date_device_uc'), )


class Weather(Base):
  __tablename__ = "weather"
  id = Column(Integer, primary_key = True)
  wdate = Column(DateTime, nullable = False)
  temp = Column(Float, nullable = False)
  device_id = Column(Integer, ForeignKey("device.id"))
  readings_for_device = relationship("Device", cascade = "all, delete-orphan", single_parent = True)
  __table_args__ = (UniqueConstraint('wdate', 'device_id', name='_date_device_uc'), )

