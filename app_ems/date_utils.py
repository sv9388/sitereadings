from enums import *
from datetime import datetime, timedelta

def delta_steps(start, end, agg_period):
  delta = None
  if agg_period == FIFTEEN_MINUTES:
    delta = timedelta(minutes = 15)
  elif agg_period == HOURLY:
    delta = timedelta(hours = 1)
  else:
    delta = timedelta(days = 1)
  curr = start
  op  = []
  while curr < end:
    op.append(curr)
    curr += delta
  return op

def get_date_range(cperiod):
  today = datetime.now()
  if cperiod == LAST_7_DAYS:
    return today + timedelta(days = -7), today
  if cperiod == LAST_30_DAYS:
    return today + timedelta(days = -30), today
  if cperiod == CURRENT_MONTH:
    return datetime(today.year, today.month, 1), today
  if cperiod == LAST_12_MONTHS:
    return today + timedelta(months = -12), today
  if cperiod == CURRENT_YEAR:
    return datetime(today.year, 1, 1), today
  return None, None

