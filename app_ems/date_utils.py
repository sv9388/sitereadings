from enums import *
from datetime import datetime, timedelta

def delta_steps(start, end, agg_period):
  print(start, end, agg_period)
  delta = None
  if not start and not end:
    return []
  if agg_period == HOURLY:
    delta = timedelta(hours = 1)
  else:
    delta = timedelta(days = 1)
  curr = start
  op  = []
  while curr < end:
    op.append(curr)
    curr += delta
  return op
