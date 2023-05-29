import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ConstantModel, GaussianModel, SineModel
from lmfit import Parameters, minimize, report_fit
import random
import itertools
import pandas as pd


# turns "None" values to 0s. Useful for dealing with IBMQ output
def get_partial_key_matches(dictionary, partialKey):
    return dict(filter(lambda item: partialKey in item[0], dictionary.items()))


def sum_dict(dictionary):
    return np.sum(list(dictionary.values()))


def deNone(value):
    return int(0 if not value == value or value is None else value)


def flatten(ndlist):
    return [item for sublist in ndlist for item in sublist]


# returns the index of the element nearest to x
def closest_index(values, x):
    values = np.asarray(values)
    return (np.abs(values - x)).argmin()