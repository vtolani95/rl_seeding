import tensorflow as tf
import logging
from baselines import logger

def add_value_to_summary(metric_summary, tag, val, log=True, tag_str=None):
  """Adds a scalar summary to the summary object. Optionally also logs to
  logging."""
  if metric_summary is not None:
    new_value = metric_summary.value.add();
    new_value.tag = tag
    new_value.simple_value = val
  if log:
    if tag_str is None:
      tag_str = tag + '{}'
    logger.info(tag_str, str(val))
