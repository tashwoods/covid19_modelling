import sys, os, math, shutil, time, argparse, matplotlib, csv, math, itertools
import random as stock_random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from pylab import *
from pandas.plotting import scatter_matrix
import seaborn as sns
from matplotlib.pyplot import cm
import multiprocessing
matplotlib.use("Agg") #disable python gui rocket in mac os dock
import matplotlib.pyplot as plt
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin #for attribute adder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import ConvergenceMonitor
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
import yfinance as yf
from countryinfo import CountryInfo
from census import Census
from us import states
from classes import *
from cycler import cycler
import random
import csv
import requests
import json
from urllib.request import urlopen
import plotly.express as px
import geopandas
#import imageio
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta, date #for time dependent gif plots
from scipy.optimize import curve_fit #to fit sigmoids of infection/death rates
from scipy.optimize import fsolve #to solve logistic curve for infection end date
import matplotlib.dates as mdates

#For LSTM based on https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
from keras.models import Sequential
#from keras.layers import Dense, L
#from keras.layers import LSTM
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import EarlyStopping
########
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers
#from keras.utils.vis_utils import model_to_dot
#import pydot as pyd
#from IPython.display import SVG
from keras.utils import plot_model


#Internal Package Modules
from organize_input_output import *
from plot import *
from lstm import *

