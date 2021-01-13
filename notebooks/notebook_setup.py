"""
Adds local dependencies to PYTHONPATH when imported as a module. Set up default
plotting parameters.
"""
import sys, os
import matplotlib.pyplot as plt



# Local packages
file_path = os.path.abspath(os.path.dirname(__file__))
pyStateSpace_path = os.path.join(file_path, '../../pyStateSpace')
pyTorchBridge_path = os.path.join(file_path, '../../pyTorchBridge')
try:
    import pystatespace
except ImportError:
    if pyStateSpace_path not in sys.path:
        sys.path.append(pyStateSpace_path)
try:
    import pytorchbridge
except ImportError:
    if pyTorchBridge_path not in sys.path:
        sys.path.append(pyTorchBridge_path)

local_path = os.path.join(file_path, '..')
sys.path.append(local_path)


# Plotting parameters
SMALL_SIZE = 15
MEDIUM_SIZE = 17
BIGGER_SIZE = 19

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth = 2.5)
