"""Assumed immediate pad translation feedback, not measured finger motion."""
import numpy as np


def follow_offset(error_local, rate_local, enabled=True):
    if not enabled:return np.zeros(3),np.zeros(3)
    u=.5*np.asarray(error_local)/.005;t=np.tanh(u)
    return .005*t,.5*(1-t*t)*np.asarray(rate_local)
