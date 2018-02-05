from settings import *
from sqlalchemy import Integer, ForeignKey, String, Column, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.schema import UniqueConstraint
Base = declarative_base()
class Device(Base):
  __tablename__ = "device"
  id = Column(Integer, primary_key = True)
  name = Column(String, nullable = False, unique = True)

class Reading(Base):
  __tablename__ = "reading"
  id = Column(Integer, primary_key = True)
  rdate= Column(DateTime, nullable = False)
  input_voltage_l1 = Column(Float, nullable = False)
  input_voltage_l2 = Column(Float, nullable = False)
  input_voltage_l3 = Column(Float, nullable = False)
  output_voltage = Column(Integer, nullable = False)
  current_l1 = Column(Float, nullable = False)
  current_l2 = Column(Float, nullable = False)
  current_l3 = Column(Float, nullable = False)
  pf_l1 = Column(Float, nullable = False)
  pf_l2 = Column(Float, nullable = False)
  pf_l3 = Column(Float, nullable = False)
  power_l1 = Column(Float, nullable = False)
  power_l2 = Column(Float, nullable = False)
  power_l3 = Column(Float, nullable = False)
  total_power_kw = Column(Float, nullable = False)
  total_kwh = Column(Integer, nullable = False)
  window2_consumption = Column(Integer, nullable = False)
  today_saved = Column(Float, nullable = False)
  today_consumption = Column(Float, nullable = False)
  yesterday_saved = Column(Float, nullable = False)
  yesterday_consumption = Column(Float, nullable = False)
  weekly_saved = Column(Float, nullable = False)
  weekly_consumption = Column(Float, nullable = False)
  last_week_saved = Column(Float, nullable = False)
  last_week_consumption = Column(Float, nullable = False)
  monthly_saved = Column(Float, nullable = False)
  monthly_consumption = Column(Float, nullable = False)
  last_month_saved = Column(Float, nullable = False)
  last_month_consumption = Column(Float, nullable = False)
  yearly_saved = Column(Float, nullable = False)
  yearly_consumption = Column(Float, nullable = False)
  last_year_saved = Column(Float, nullable = False)
  last_year_consumption = Column(Float, nullable = False)
  over_load_indication = Column(Integer, nullable = False)
  missing_phase_indication = Column(Integer, nullable = False)
  over_temperature_indication = Column(Integer, nullable = False)
  temperature_l1 = Column(Float, nullable = False)
  temperature_l2 = Column(Float, nullable = False)
  temperature_l3 = Column(Float, nullable = False)
  cpu_temperature = Column(Integer, nullable = False)
  device_mode = Column(Integer, nullable = False)
  reduction_step = Column(Integer, nullable = False)
  software_resets = Column(Integer, nullable = False)
  device_id = Column(Integer, ForeignKey("device.id"))
  readings_for_device = relationship("Device")
  __table_args__ = (UniqueConstraint('rdate', 'device_id', name='_date_device_uc'), )


def update_db():
  engine = create_engine(DB_URI)
  Base.metadata.create_all(engine)

if __name__ == "__main__":
  update_db()
