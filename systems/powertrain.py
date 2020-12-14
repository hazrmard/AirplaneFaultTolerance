from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import gym
import json
import datetime
import os
np.random.seed(43)


############################################################################################
#                                                                        HELPER FUNCTIONS
############################################################################################


# --------------------------------------------------------------------------------- get_poly
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

# ------------------------------------------------------------------------ get_battery_curves
def get_battery_curves(soc_ocv_file, R0_degradation_file, Q_degradation_file):
    """
    @brief: gets the degradation profile (predefined curves) for the battery

    @input:
        soc_ocv_file: a csv file containing the soc_ocv relationship as a column vector
        R0_degradation_file: a csv file containing the R0 degradation curve as a column vector
        Q_degradation_file: a csv file containing the Q degradation curve as a column vector

    @output:
        a dictionary mapping of degradation curve coefficients with ["z_coef", "r0_coef", "q_coef", "EOL", "soc_ocv"] keys
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
    eol = max(len(Q_degradation), len(R0_degradation))

    return {"z_coef": z_coef[:,0].tolist(), "r0_coef": r0_coef[:,0].tolist(), "q_coef": q_coef[:,0].tolist(), "eol": eol}
    #return {"z_coef": z_coef, "r0_coef": r0_coef, "q_coef": q_coef, "eol": eol, "soc_ocv": soc_ocv}
    #return {"z_coef": list(z_coef), "r0_coef": list(r0_coef), "q_coef": list(q_coef), "eol": eol}


# ------------------------------------------------------------------------------- three_plot
def three_plot(title="title", figsize=(12, 4),
               plot1=np.array(([[1, 2, 3], [1, 1, 1]])),
               plot2=np.array(([[1, 2, 3], [2, 2, 2], [1, 2, 3], [2.3, 2.4, 2.5]])),
               plot3=np.array(([[1, 2, 3], [3, 3, 3], [4, 5, 6], [4, 4, 4], [1.4, 1.8, 2.2, 2.6, 3.0],
                                [3, 3.05, 3.15, 3.3, 3.55]])),
               label1=["label1"], label2=["label2a", "label2b"], label3=["label3-1", "label3-2", "label3-3"],
               title1="title1", title2="title2", title3="title3",
               axes1=["x-axis1", "y-axis1"], axes2=["x-axis2", "y-axis2"], axes3=["x-axis3", "y-axis3"],
               invert=[0,0,0], save=False, filename=""):
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

    @output: none
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


# ----------------------------------------------------------------------------------- cycle_test
def cycle_test(cell, random_load=True, verbose=0, show_plot=False,
               dt=1.0, save_plot=False, reset=True, file_name="",
               run_num=0, q_vals=[], t_vals=[], action=1):
    """
    @brief: cycles a battery once, accomadates chargning/discharging, random or steady loads

    @input:
        cell: a continuous battery cell
        random_load: boolean, whether to use a random load or constant current
        verbose: 0 = none, 1 = end of cycle information, 2 = even more info
        show_plot: boolean to show or not show plot
        dt: sample time, sets the cells period
        save_plot: boolean to save the plot
        reset: reset the battery or not, to be removed
        file_name: filename of plot
        run_num: run number of the cycle
        q_vals: list of q vals to save at the end of each cycle
        t_vals: list of cycle_time vals to save at the end of each cycle
        action: 0=charge, 1=discharge

    @output: none
    """
    cell.period = dt
    ct = 1/3600*cell.period
    dt = cell.nsteps(ct)
    # data holders for current, soc, noisy soc, filtered soc, voltage
    cs, cz, cn, cf, cv, vn, vf = [], [], [], [], [], [], []
    # random current profile for a given run
    if(random_load):
        x = np.linspace(0, 2 * np.pi, int(cell.z * 100) + 1)
        amp = np.random.uniform(1,2)
        pwr = np.random.uniform(.8, 1.4)
        fct = np.random.uniform(3,5)
        ofs = np.random.uniform(3,5)
        y = np.sin(amp*-x**pwr)/fct+ofs
    done = False
    i = 0
    while(not done):
        if random_load:
            c = np.random.normal(y[int(cell.z * 100)], .1)
        else:
            c = 3.8695
        if(action == 0):
            c = -c
        obs, reward, done, info = cell.step(dt, c)
        cs.append(c)
        cz.append(cell.z)
        cn.append(obs[0])
        cf.append(obs[1])
        vn.append(obs[2])
        vf.append(obs[3])
        cv.append(cell.ocv)
        if(verbose==2):
            if(i == 0):
                print("first 20 observations:")
            if(i < 20):
                print("noisy_z: {:.3f}\tfiltered_z: {:.3f}\tnoisy_v: {:.3f}\tfiltered_v: {:.3f}".format(obs[0], obs[1], obs[2], obs[3]))
        i += 1
    q_vals.append(cell.Q)
    t_vals.append(int(cell.cycle_time))
    if(verbose==2):
        print("elapsed seconds: {:.2f}".format(int(i * cell.period)))
        print("ending soc: {:.4f}".format(cell.z))
        print("cycle time: {:.2f}".format(cell.cycle_time))
    if(verbose > 0):
        print("run: {}\tQ: {:.3f}\tR0: {:.3f}\tavg_load: {:.3f}\tcycle_time: {}\tage: {:.3f}\teol: {}".format(run_num, cell.Q, cell.R0, cell.avg_load, int(cell.cycle_time), cell.age, int(cell.eol)))
    if(show_plot):
        factor = 1 / 60 * cell.period
        X = np.arange(0, i)*factor
        three_plot(title="Continuous Battery Cell Plots", figsize=(12,4),
                   plot1=np.array([X, np.array(cs)]),
                   plot2=np.array([X, vn, X, vf, X, cv]),
                   plot3=np.array([X[0:80], cn[0:80],X[0:80], cf[0:80],X[0:80], cz[0:80]], dtype=object),
                   label1=["current profile"],
                   label2=["observed voltage", "filtered voltage", "actual voltage"],
                   label3=["observed soc", "filtered soc", "actual soc"],
                   title1="Current Draw",
                   title2="Open Circuit Voltage",
                   title3="State of Charge",
                   axes1=['time (minutes)', 'current (A)'],
                   axes2=['time (minutes)', 'ocv (V)'],
                   axes3=['time (minutes)', 'soc (%)'],
                   save=save_plot,
                   filename=file_name)
    if(reset):
        cell.reset()



############################################################################################
#                                                                        BASE BATTERY CLASS
############################################################################################

class Battery:
    def __init__(self, *args, **kwargs):
        if (len(kwargs) == 0):
            self.name = "base_battery"
            self.z_coef = np.zeros((1, 1))
            self.r0_coef = np.zeros((1, 1))
            self.q_coef = np.zeros((1, 1))
            self.eol = 320
            self.z = 1.0
            self.Ir = 0
            self.h = 0
            self.M0 = .0019
            self.M = .0092
            self.R0 = .0112
            self.R0_init = self.R0
            self.R = 2.83e-4
            self.Q = 3.8695
            self.n = .9987
            self.G = 163.4413
            self.v0 = 4.2
            self.eod = 3.04
            self.RC = 3.6572
            self.ocv = self.v0
        else:
            print(f"Using [ {kwargs['name']} - {kwargs['modified']}] parameters")
            self.name = kwargs['name']
            self.z_coef = kwargs["z_coef"]
            self.r0_coef = kwargs["r0_coef"]
            self.q_coef = kwargs["q_coef"]
            self.eol = kwargs["eol"]
            self.z = kwargs["z"]
            self.Ir = kwargs["Ir"]
            self.h = kwargs["h"]
            self.M0 = kwargs["M0"]
            self.M = kwargs["M"]
            self.R0 = kwargs["R0"]
            self.R = kwargs["R"]
            self.Q = kwargs["Q"]
            self.n = kwargs["n"]
            self.G = kwargs["G"]
            self.v0 = kwargs["v0"]
            self.eod = kwargs["eod"]
            self.RC = kwargs["RC"]
            self.R0_init = self.R0
            self.ocv = self.v0

    def step(self, dt, current):
        raise NotImplementedError

    def reset(self):
        self.z = 1.0
        self.ocv = self.v0
        self.Ir = 0
        self.h = 0


############################################################################################
#                                                                 CONTINUOUS BATTERY CLASS
############################################################################################



class ContinuousBatteryCell(Battery, gym.Env):
    """
    @brief: implements continuous time 3 parameter battery cell with load-dependant degradation,
            noisy observations and moving average filters

    @params:
        observation_space: [noisy soc, filtered soc, noisy ocv, filtered ocv]
        action_space: [0/1] for charging / discharging

    @todo:
        1) implement RUL estimation
        2) implement charging / discharging
    """
    def __init__(self, *args, **kwargs):
        super(ContinuousBatteryCell, self).__init__(*args, **kwargs)

        # simple moving average filter
        self.filter = lambda X: np.sum(X) / self.filter_len

        # obs[0] = noisy observation of soc, obs[1] = filtered observation of soc
        # obs[2] = noisy observation of ocv, obs[3] = filtered observation of ocv
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,))

        # - for charging, + for discharging, not implemented yet
        self.action_space = gym.spaces.Box(low=-100.0, high = 100.0, shape=(1,))

        # used for ode solver
        self.nsteps = lambda x: np.linspace(0, x, 100)

        # assume the other keys are there too
        if "state" in kwargs.keys():
            #noisy z, filtered z, noisy v, filtered v
            self.state = kwargs['state']

            # simple moving average filter
            self.filter_len = kwargs["filter_len"]

            # accumulator and index for moving average filters
            self.idx = kwargs["idx"]
            self.accum_z = kwargs["accum_z"]
            self.accum_v = kwargs["accum_v"]

            # sample rate
            self.period = kwargs["period"]

            # random seed for reproduction
            self._seed = kwargs["_seed"]

            # for degradation and rul estimation
            self.age = kwargs["age"]
            self.cycle_time = kwargs["cycle_time"]

            # for calculating the degradation rate, which is avg_load / cycle_time
            self.avg_load = kwargs["avg_load"]
            self.count = kwargs["count"]
        else:
            self.state = [self.z, self.z, self.v0, self.v0]
            self.filter_len = 25
            self.accum_z = np.zeros((self.filter_len,)).tolist()
            self.accum_v = np.zeros((self.filter_len,)).tolist()
            self.idx = 0
            self.period = .05
            self._seed = 43
            self.age = 0.0
            self.cycle_time = 0.0
            self.avg_load = 0.0
            self.count = 0

        self.seed(self._seed)


    def save_state(self, filename=""):
        params = self.__dict__
        params.pop('filter', None)
        params.pop('observation_space', None)
        params.pop('action_space', None)
        params.pop('nsteps', None)
        params['accum_z'] = params['accum_z'].tolist()
        params['accum_v'] = params['accum_v'].tolist()
        params['modified'] = datetime.datetime.now().strftime("%d%b%y_%H-%M-%S")
        if (len(filename) < 1):
            if not os.path.exists(str(os.path.abspath(os.getcwd())+f"\\params\\{self.name}")):
                os.makedirs(str(os.path.abspath(os.getcwd())+f"\\params\\{self.name}"))
            filename = f"params/{self.name}/{params['modified']}.json"
        with open(filename, 'w') as f:
            json.dump(params, f)

    def load_state(self, filename):
        if(len(filename) < 1):
            print("please supply a filename")
            return
        else:
            with open(filename) as f:
                params = json.load(f)


    def seed(self, seed):
        """sets the random seed"""
        self.observation_space.seed(self._seed)
        self.action_space.seed(self._seed)

    def get_v(self, z):
        """returns the ideal open circuit voltage for a given state of charge"""
        assert 0.0 <= z <= 100.0, "z in range [0.0, 100.0]"
        return np.polyval(self.z_coef, z)

    def get_r0(self, age):
        """returns the value of r0 at a given battery age"""
        assert 0.0 <= age <= self.eol, "age in range [0.0, {}]".format(self.eol)
        return np.polyval(self.r0_coef, age)

    def get_q(self, age):
        """returns the charge capacity at a given battery age"""
        assert 0.0 <= age <= self.eol, "age in range [0.0, {}]".format(self.eol)
        return np.polyval(self.q_coef, age)

    def _dzdt(self, soc, t, current):
        """first order de for state of charge"""
        return -self.n * current / self.Q

    def _didt(self, Ir, t, current):
        """first order de for the current passing thru the first resistor"""
        return -1.0 / self.RC * Ir + 1.0 / self.RC * current

    def _dhdt(self, h, t, current):
        """first order de for the hysteresis value"""
        return -np.absolute(self.n * current * self.G / self.Q) * h + self.n * current * self.G / self.Q * self.M * np.sign(current)

    def step(self, dt, current):
        """steps the ode solver, noisy_z and filtered_z voltages"""
        _z = (1 - odeint(self._dzdt, 1.0, dt, args=(current,))[-1][0])
        _i = odeint(self._didt, self.Ir, dt, args=(current,))[-1][0]
        _h = odeint(self._dhdt, self.h, dt, args=(current,))[-1][0]

        # update the battery parameters
        self.Ir = _i
        self.z = self.z - _z
        self.h = _h
        self.ocv = np.polyval(self.z_coef, self.z * 100.0) - self.R * self.Ir - self.R0 * current + self.h + self.M0 * np.sign(current)

        # noisy observations
        noisy_z = np.clip(np.random.normal(self.z, .01), 0.0, 100.0)
        noisy_v = np.clip(np.random.normal(self.ocv, .025), -.5, 5.0)

        # update accumulator for filtered_z
        if (self.accum_z[-1] == 0):
            self.accum_z = np.ones((self.filter_len,)) * noisy_z
        else:
            self.accum_z[self.idx] = noisy_z

        # update accumulator for filtered_v
        if (self.accum_v[-1] == 0):
            self.accum_v = np.ones((self.filter_len,)) * noisy_v
        else:
            self.accum_v[self.idx] = noisy_v

        # update accumulator counter
        self.idx += 1
        if (self.idx == self.filter_len):
            self.idx = 0

        # filtered_z observation
        filtered_z = np.clip(self.filter(self.accum_z), 0.0, 100.00000)
        filtered_v = np.clip(self.filter(self.accum_v), -.5, 5.00000)

        # update the state [noisy_z soc, filtered_z soc]
        self.state = [noisy_z, filtered_z, noisy_v, filtered_v]

        # living reward
        reward = -.01

        # never truly reach eod, always slightly above when discharging
        if(current > 0):
            done = True if self.ocv <= self.eod + .1 else False
        else:  # charging
            done = True if self.z >= .995 else False

        # update current cycle time
        self.cycle_time += self.period

        # update cumulative average
        self.avg_load = (current + self.count * self.avg_load) / (self.count + 1)
        self.count += 1

        if(done):
            self.age += self.avg_load/self.cycle_time * 850
            self.age = np.clip(self.age,  0.0, self.eol)

        return self.state, reward, done, locals()

    def reset(self):
        """resets the battery"""
        super().reset()
        self.state = [self.z, self.z, self.v0, self.v0]
        self.accum_z = np.zeros((self.filter_len,))
        self.accum_v = np.zeros((self.filter_len,))
        self.idx = 0
        self.cycle_time = 0.0
        self.avg_load = 0.0
        self.count = 0
        self.Q = self.get_q(self.age)
        self.R0 = np.clip(self.get_r0(self.age), self.R0_init, .5)


############################################################################################
#                                      DISCRETE BATTERY CLASS (depreciated, historical only)
############################################################################################
class DiscreteBatteryCell(Battery):
    def __init__(self, *args, **kwargs):
        super(DiscreteBatteryCell, self).__init__(*args, **kwargs)
        self.soc_ocv = kwargs['soc_ocv']

    def get_ocv(self):
        if (self.z < 0.0):
            self.z = 0
        elif (self.z > 1.0):
            self.z = 1.0
        idx = int(np.ceil(self.z * 100))
        if (idx > 101):
            idx = 101
        elif (idx < 1):
            idx = 1
        return self.soc_ocv[idx]

    def step(self, dt, current):
        RC = np.exp(-dt / abs(self.RC))
        H = np.exp(-abs(self.n * current * self.G * dt / (3600 * self.Q)))
        self.Ir = RC * self.Ir + (1 - RC) * current
        self.h = H * self.h + (H - 1) * np.sign(current)
        self.z = self.z - self.n * current / 3600 / self.Q
        self.ocv = self.get_ocv() + self.M * self.h + self.M0 * np.sign(current) - self.R * self.Ir - self.R0 * current