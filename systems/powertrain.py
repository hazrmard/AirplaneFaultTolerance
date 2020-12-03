from abc import ABC

from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import gym
np.random.seed(43)


############################################################################################
#                                                                        HELPER FUNCTIONS
############################################################################################

def get_poly(curve, pwr=5):
    """
        @brief: gets the coefficients for a polynomial approximation of given curve

        @input:
            curve: the curve as an np.array
            pwr: power of the curve to fit, default=5

        @output:
            an np.array of length(pwr) representing the coefficients of the curve
    """
    x_poly = np.arange(0, len(curve))
    y_poly = np.polyfit(x_poly, curve, pwr)
    return y_poly


def get_battery_curves(soc_ocv_file, R0_degradation_file, Q_degradation_file):
    """
        @brief: gets the degradation profile (predefined curves) for the battery

        @input:
            soc_ocv_file: a csv file containing the soc_ocv relationship as a column vector
            R0_degradation_file: a csv file containing the R0 degradation curve as a column vector
            Q_degradation_file: a csv file containing the Q degradation curve as a column vector

        @output:
            a dictionary mapping of degradation curve coefficients with ["z_coef", "r0_coef", "q_coef", "EOL"] keys
    """
    soc_ocv = []
    R0_degradation = []
    Q_degradation = []

    with open(soc_ocv_file, newline='') as f:
        soc_ocv = list(csv.reader(f))
    soc_ocv = np.asarray(soc_ocv).astype(np.float)

    with open(R0_degradation_file, newline='') as f:
        R0_degradation = list(csv.reader(f))
    R0_degradation = np.asarray(R0_degradation).astype(np.float)

    with open(Q_degradation_file, newline='') as f:
        Q_degradation = list(csv.reader(f))
    Q_degradation = np.asarray(Q_degradation).astype(np.float)

    z_coef = get_poly(soc_ocv)
    r0_coef = get_poly(R0_degradation)
    q_coef = get_poly(Q_degradation)
    eol = len(Q_degradation)

    return {"z_coef": z_coef, "r0_coef": r0_coef, "q_coef": q_coef, "eol": eol, "soc_ocv": soc_ocv}


def three_plot(title="title",
               figsize=(12, 4),
               plot1=np.array(([[1, 2, 3], [1, 1, 1]])),
               plot2=np.array(([[1, 2, 3], [2, 2, 2], [1, 2, 3], [2.3, 2.4, 2.5]])),
               plot3=np.array(([[1, 2, 3], [3, 3, 3], [4, 5, 6], [4, 4, 4], [1.4, 1.8, 2.2, 2.6, 3.0],
                                [3, 3.05, 3.15, 3.3, 3.55]])),
               label1=["label1"],
               label2=["label2a", "label2b"],
               label3=["label3-1", "label3-2", "label3-3"],
               title1="title1",
               title2="title2",
               title3="title3",
               axes1=["x-axis1", "y-axis1"],
               axes2=["x-axis2", "y-axis2"],
               axes3=["x-axis3", "y-axis3"],
               invert=[0,0,0],
               save=False,
               filename=""):
    """
        @brief: creates a 3 subplot figure.

        @input:
              title: super title
              figsize: figuresize
              plot1/2/3: an np.ndarray of X, Y pairs where X and Y are vectors. ex np.array(([X1, Y1, X2, Y2, ... Xn, Yn]))
              label1/2/3: a list of labels for each plot
              title1/2/3: subplot titles
              axes1/2/3: x and y axes labels for each subplot
              invert: a 3,1 list where 0 is no x axis invert and 1 is invert, ex [0,0,1] inverts the 3rd axis
              save: bool flag to save the file
              filename: filename for saving
    """
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(12, 4))

    for i in range(int(plot1.shape[0] / 2)):
        ax1.plot(plot1[i * 2], plot1[i * 2 + 1], label=label1[i])
    ax1.set(xlabel=axes1[0], ylabel=axes1[1])
    ax1.set_title(title1)
    if(invert[0]):
        ax1.invert_xaxis()
    ax1.legend()

    for i in range(int(plot2.shape[0] / 2)):
        ax2.plot(plot2[i * 2], plot2[i * 2 + 1], label=label2[i])
    ax2.set(xlabel=axes2[0], ylabel=axes2[1])
    ax2.set_title(title2)
    if(invert[1]):
        ax2.invert_xaxis()
    ax2.legend()

    for i in range(int(plot3.shape[0] / 2)):
        ax3.plot(plot3[i * 2], plot3[i * 2 + 1], label=label3[i])
    ax3.set(xlabel=axes3[0], ylabel=axes3[1])
    ax3.set_title(title3)
    if(invert[2]):
        ax3.invert_xaxis()
    ax3.legend()
    f.tight_layout(rect=[0, .09, 1, .9])
    f.suptitle(title)
    plt.show()
    if (save):
        assert len(filename) > 3, "please specify a filename"
        f.savefig(filename)




############################################################################################
#                                                                        BASE BATTERY CLASS
############################################################################################
class Battery:
    def __init__(self, *args, **kwargs):
        if (len(kwargs) == 0):
            self.z_coef = np.zeros((1, 1))
            self.r0_coef = np.zeros((1, 1))
            self.q_coef = np.zeros((1, 1))
            self.eod = 1
            self.eol = 1
        else:
            self.z_coef = kwargs["z_coef"]
            self.r0_coef = kwargs["r0_coef"]
            self.q_coef = kwargs["q_coef"]
            self.eol = kwargs["eol"]
        self.z = 1.0
        self.Ir = 0
        self.h = 0
        self.M0 = .0019
        self.M = .0092
        self.R0 = .0112
        self.R = 2.83e-4
        self.Q = 3.8695
        self.n = .9987
        self.G = 163.4413
        self.v0 = 4.2
        self.eod = 3.04
        self.RC = 3.6572
        self.ocv = self.v0

    def get_ocv(self):
        raise NotImplementedError

    def step(self, dt, current):
        raise NotImplementedError

    def reset(self):
        self.z = 1.0
        self.ocv = self.v0
        self.Ir = 0
        self.h = 0


class ContinuousBatteryCell(Battery, gym.Env):
    def __init__(self, *args, **kwargs):
        super(ContinuousBatteryCell, self).__init__(*args, **kwargs)

        # simple moving average filter
        self.filter_len = 25
        self.filter = lambda X: np.sum(X) / self.filter_len

        # accumulator and index for moving average filter
        self.accum = np.zeros((self.filter_len,))
        self.idx = 0

        # obs[0] = noisy observation of soc, obs[1] = filtered observation of soc
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,))
        self.state = [100.0, 100.0]

        # 0 for charging, 1 for discharging, not implemented yet
        self.action_space = gym.spaces.Discrete(2)  # charging or discharging

        # used for ode solver
        self.nsteps = lambda x: np.linspace(0, x, 100)
        self.period = .05

        # random seed to aid reproduction
        self._seed = 43
        self.seed(self._seed)

        # for degradation, max age is super.eol
        self.age = 0.0
        self.cycle_time = 0.0

        # for calculating the degradation rate, which is avg_load / cycle_time
        self.avg_load = 0.0
        self.count = 0

    def seed(self, seed):
        """sets the random seed"""
        self.observation_space.seed(self._seed)
        self.action_space.seed(self._seed)

    def get_v(self, z):
        """returns the ideal open circuit voltage for a given state of charge"""
        assert 0.0 <= z <= 100.0, "z in range [0.0, 100.0]"
        return np.polyval(self.z_coef, z)[0]

    def get_r0(self, age):
        """returns the value of r0 at a given battery age"""
        assert 0.0 <= age <= self.eol, "age in range [0.0, {}]".format(self.eol)
        return np.polyval(self.r0_coef, age)[0]

    def get_q(self, age):
        """returns the charge capacity at a given battery age"""
        assert 0.0 <= age <= self.eol, "age in range [0.0, {}]".format(self.eol)
        return np.polyval(self.q_coef, age)[0]

    def _cumulative_avg(self, current):
        """calculates the cumulative average current load"""
        self.avg_load = (current + self.count*self.avg_load)/(self.count + 1)
        self.count += 1

    def _dzdt(self, soc, t, current):
        """first order de for state of charge"""
        return -self.n * current / self.Q

    def _didt(self, Ir, t, current):
        """first order de for the current passing thru the first resistor"""
        return -1.0 / self.RC * Ir + 1.0 / self.RC * current

    def _dhdt(self, h, t, current):
        """first order de for the hysteresis value"""
        return -np.absolute(
            self.n * current * self.G / self.Q) * h + self.n * current * self.G / self.Q * self.M * np.sign(current)

    def step(self, dt, current):
        """steps the ode solver, noisy and filtered voltages"""
        _z = (1 - odeint(self._dzdt, 1.0, dt, args=(current,))[-1][0])
        _i = odeint(self._didt, self.Ir, dt, args=(current,))[-1][0]
        _h = odeint(self._dhdt, self.h, dt, args=(current,))[-1][0]

        # update the battery parameters
        self.Ir = _i
        self.z = self.z - _z
        self.h = _h
        self.ocv = np.polyval(self.z_coef, self.z * 100.0) - self.R * self.Ir - self.R0 * current + self.h + self.M0 * np.sign(current)

        # noisy observation
        noisy = np.clip(np.random.normal(self.z, .01), 0.0, 100.0)

        # update accumulator for the filter
        if (self.accum[-1] == 0):
            self.accum = np.ones((self.filter_len,)) * noisy
        else:
            self.accum[self.idx] = noisy
        self.idx += 1
        if (self.idx == self.filter_len):
            self.idx = 0

        # filtered observation
        filtered = np.clip(self.filter(self.accum), 0.0, 100.0)

        # update the state [noisy soc, filtered soc]
        self.state = [noisy, filtered]

        # living reward
        reward = -.01

        # never truly reach eod, always slightly above
        done = True if self.ocv <= self.eod + .1 else False

        # update current cycle time
        self.cycle_time += self.period

        # update cumulateive average
        self._cumulative_avg(current)

        if(done):
            self.age += self.avg_load/self.cycle_time * 850
            self.age = np.clip(self.age,  0.0, self.eol)

        return self.state, reward, done, locals()

    def reset(self):
        """resets the battery"""
        super().reset()
        self.state = [self.z, self.z]
        self.accum = np.zeros((self.filter_len,))
        self.idx = 0
        self.cycle_time = 0.0
        self.avg_load = 0.0
        self.count = 0
        self.Q = self.get_q(self.age)