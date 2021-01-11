# MIT License

# Copyright (c) 2020 Ibrahim Ahmed
# Copyright (c) 2020 Yves Sohege
# Copyright (c) 2017 Abhijit Majumdar

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import datetime
import signal
import threading
import math
from copy import deepcopy

import scipy.integrate
import scipy.stats as stats
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
from tqdm.auto import tqdm


lower, upper = 0, 1
mu = 0
sigma = 1
randDist = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu,
                                 scale=sigma)

QUADPARAMS = {
    'position': [0, 0, 0],      # (metres), optional, defalts to 0
    'orientation': [0, 0, 0],   # (degrees), optional, defaults to 0
    'ground_level': -np.inf,    # (metres), optional, location of ground plane
    'r': 0.1,                   # (metres), Radius of sphere representing centre of quadcopter
    'L': 0.3,                   # (metres), length of arm
    'prop_size': [10, 4.5],     # (inches), diameter & pitch of rotors
    'mass': 1.2                 # (kilograms)
    }

CONTROLLER_PARAMS = {
    'Motor_limits': [4000, 9000],
    'Tilt_limits': [-10, 10],            # degrees
    'Yaw_Control_Limits': [-900, 900],
    'Z_XY_offset': 500,
    'Linear_To_Angular_Scaler': [1, 1, 0],
    'Yaw_Rate_Scaler': 0.18,
    'Linear_PID': {
        'P':[300, 300, 7000],
        'I':[0.04, 0.04, 4.5],
        'D':[450, 450, 5000]},
    'Angular_PID':{
        'P':[22000, 22000, 1500],
        'I':[0, 0, 1.2],
        'D':[12000, 12000, 0]},
    }



class Motor:


    def __init__(self, prop_dia, prop_pitch, thrust_unit='N'):
        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.speed = 0 #RPM
        self.thrust = 0
        self.fault_mag = 0


    def set_speed(self, speed):
        self.speed = speed
        if self.fault_mag > 0:
            self.speed = self.speed * (1 - self.fault_mag)
        # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        self.thrust = 4.392e-8 * self.speed * math.pow(self.dia, 3.5) / (math.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)

        if self.thrust_unit == 'Kg':
            self.thrust = self.thrust * 0.101972


    def get_speed(self):
        return self.speed


    def set_fault(self, fault):
        self.fault_mag = fault



class Quadcopter:
    # State space representation: [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
    # From Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky
    def __init__(self, quad=QUADPARAMS, gravity=9.81, b=0.0245, turbulence=15, random=np.random.RandomState()):
        # a copy of `quad` so any QUADPARAMS are not changed
        self.quad = deepcopy(quad)
        self.random = random
        # Initialize variables
        self.reset()
        self.g = gravity
        self.b = b
        self.ground_level = self.quad.get('ground_level', -np.inf)
        self.wind = True

        # Wind disturbance
        self.nom_wind = np.zeros(3) # constant wind direction
        self.airspeed = turbulence          # used for turbulent wind generation
        self.rand_wind = self.generate_wind_turbulence(h=5) * (1 if turbulence > 0 else 0)

        # Motors
        self.quad['m1'] = Motor(self.quad['prop_size'][0], self.quad['prop_size'][1])
        self.quad['m2'] = Motor(self.quad['prop_size'][0], self.quad['prop_size'][1])
        self.quad['m3'] = Motor(self.quad['prop_size'][0], self.quad['prop_size'][1])
        self.quad['m4'] = Motor(self.quad['prop_size'][0], self.quad['prop_size'][1])
        # From Quadrotor Dynamics and Control by Randal Beard
        ixx = ((2*self.quad['mass']*self.quad['r']**2)/5)+(2*self.quad['mass']*self.quad['L']**2)
        iyy = ixx
        izz = ((2*self.quad['mass']*self.quad['r']**2)/5)+(4*self.quad['mass']*self.quad['L']**2)
        self.quad['I'] = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self.quad['invI'] = np.linalg.inv(self.quad['I'])

        # Scaffolding for running simulation
        self.ode = scipy.integrate.ode(self.state_dot).set_integrator('vode', nsteps=500, method='bdf')
        self.thread_object = None
        self.run = True


    def reset(self):
        self.stepNum = 0
        self.state = np.zeros(12, dtype=np.float32)
        self.state[0:3] = self.quad.get('position', 0.)
        self.state[6:9] = self.quad.get('orientation', 0.)
        self.time = datetime.datetime.now()


    def rotation_matrix(self, angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R


    def wrap_angle(self, val):
        return( (val + np.pi) % (2 * np.pi) - np.pi)


    def setWind(self, winds: np.ndarray):
        self.rand_wind = wind_vec


    def setNormalWind(self, winds: np.ndarray):
        self.nom_wind = winds


    def state_dot(self, time, state):
        state_dot = np.zeros(12)
        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self.state[3]
        state_dot[1] = self.state[4]
        state_dot[2] = self.state[5]
        # The acceleration
        height = self.state[2]
        F_d = np.zeros(3)
        air_density = 1.225 #kg/m^3
        C_d = 1
        cube_width = 0.1 # 10cm x 10cm cube as shape model of quadcopter
        A_yz = cube_width*cube_width
        A_xz = cube_width*cube_width
        A_xy = cube_width*cube_width

        A = [A_yz, A_xz, A_xy] # cross sectional area in each axis perpendicular to velocity axis

        #if wind is active the velocity in each axis is subject to wind
        nomX, nomY, nomZ = self.nom_wind

        if(self.stepNum > 19500):
            self.stepNum = 0
        randX = self.rand_wind[0, self.stepNum]
        randY = self.rand_wind[1, self.stepNum]
        randZ = self.rand_wind[2, self.stepNum]


        #wind_velocity_vector = self.rand_wind
        wind_velocity_vector = [nomX + randX, nomY + randY, nomZ + randZ] # wind velocity in each axis

        wind_vel_inertial_frame = np.dot(self.rotation_matrix(self.state[6:9]), wind_velocity_vector)
        V_b = [state[0], state[1], state[2]]
        V_a = wind_vel_inertial_frame  - V_b

        DragVector = [
            A[0] * (V_a[0] * abs(V_a[0])),
            A[1] * (V_a[1] * abs(V_a[1])),
            A[2] * (V_a[2] * abs(V_a[2]))
        ]

        F_d = [i * (0.5 * air_density * C_d) for i in DragVector]

        x_dotdot = np.array([0, 0, -self.quad['mass'] * self.g]) \
                   + np.dot(self.rotation_matrix(self.state[6:9]),
                            np.array([0, 0, (self.quad['m1'].thrust + self.quad['m2'].thrust + self.quad['m3'].thrust + self.quad['m4'].thrust)])) \
                            /self.quad['mass'] \
                   + F_d

        state_dot[3] = x_dotdot[0]
        state_dot[4] = x_dotdot[1]
        state_dot[5] = x_dotdot[2]
        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = self.state[9]
        state_dot[7] = self.state[10]
        state_dot[8] = self.state[11]
        # The angular accelerations
        omega = self.state[9:12]
        tau = np.array([self.quad['L'] * (self.quad['m1'].thrust-self.quad['m3'].thrust),
                        self.quad['L'] * (self.quad['m2'].thrust-self.quad['m4'].thrust),
                        self.b * (self.quad['m1'].thrust - self.quad['m2'].thrust + self.quad['m3'].thrust - self.quad['m4'].thrust)])
        omega_dot = np.dot(self.quad['invI'], (tau - np.cross(omega, np.dot(self.quad['I'], omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot


    def update(self, dt):
        self.stepNum += 1
        self.ode.set_initial_value(self.state, 0)
        self.state = self.ode.integrate(self.ode.t + dt)
        self.state[6:9] = self.wrap_angle(self.state[6:9])
        # Clipping altitude to grould level
        self.state[2] = max(self.ground_level, self.state[2])


    def generate_wind_turbulence(self, h):
        # dryden turbulence model
        height = float(h) * 3.28084                 # metres -> feet
        airspeed = float(self.airspeed) * 3.28084   # metres/s to feet/s
        mean = 0
        std = 1
        # create a sequence of 1000 equally spaced numeric values from 0 - 5
        t_p = np.linspace(0, 10, 20000)
        num_samples = 20000
        t_p = np.linspace(0, 10, 20000)

        # the random number seed used same as from SIMULINK blockset
        np.random.seed(23341)
        samples1 = 10 * self.random.normal(mean, std, size=num_samples)

        np.random.seed(23342)
        samples2 = 10 * self.random.normal(mean, std, size=num_samples)

        np.random.seed(23343)
        samples3 = 10 * self.random.normal(mean, std, size=num_samples)

        tf_u = u_transfer_function(height, airspeed)
        tf_v = v_transfer_function(height, airspeed)
        tf_w = w_transfer_function(height, airspeed)

        tout1, y1, x1 = signal.lsim(tf_u, samples1, t_p)
        # tout1, y1, x1 = signal.lsim(tf_u, n1, t_w)
        # covert obtained values to meters/second
        y1_f = [i * 0.305 for i in y1]
        tout2, y2, x2 = signal.lsim(tf_v, samples2, t_p)
        # tout2, y2, x2 = signal.lsim(tf_v, n2, t_w)
        y2_f = [i * 0.305 for i in y2]
        tout3, y3, x3 = signal.lsim(tf_w, samples3, t_p)
        # tout3, y3, x3 = signal.lsim(tf_w, n3, t_w)
        y3_f = [i * 0.305 for i in y3]

        return np.asarray([y1_f, y2_f, y3_f])


    def set_motor_speeds(self, speeds):
        self.quad['m1'].set_speed(speeds[0])
        self.quad['m2'].set_speed(speeds[1])
        self.quad['m3'].set_speed(speeds[2])
        self.quad['m4'].set_speed(speeds[3])


    def get_motor_speeds(self):
        return [self.quad['m1'].get_speed(), self.quad['m2'].get_speed(),
                self.quad['m3'].get_speed(), self.quad['m4'].get_speed()]


    def get_motor_speeds_rpm(self):
        return [self.quad['m1'].get_speed(), self.quad['m2'].get_speed(),self.quad['m3'].get_speed(),self.quad['m4'].get_speed()]


    def get_position(self):
        return self.state[0:3]


    def get_linear_rate(self):
        return self.state[3:6]


    def get_orientation(self):
        return self.state[6:9]


    def get_angular_rate(self):
        return self.state[9:12]


    def get_state(self):
        return self.state


    def set_position(self, position):
        self.state[0:3] = position


    def set_orientation(self, orientation):
        self.state[6:9] = orientation


    def get_time(self):
        return self.time


    def thread_run(self, dt, time_scaling):
        rate = time_scaling*dt
        last_update = self.time
        while(self.run==True):
            time.sleep(0)
            self.time = datetime.datetime.now()
            if (self.time-last_update).total_seconds() > rate:
                self.update(dt)
                last_update = self.time


    def stepQuad(self, dt=0.05):
        self.update(dt)
        return


    def set_motor_faults(self, faults):
        self.quad['m1'].set_fault(faults[0])
        self.quad['m2'].set_fault(faults[1])
        self.quad['m3'].set_fault(faults[2])
        self.quad['m4'].set_fault(faults[3])


    def start_thread(self, dt=0.05, time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run, args=(dt, time_scaling))
        self.thread_object.start()


    def stop_thread(self):
        self.run = False



# Low altitude Model
# transfer function for along-wind
def u_transfer_function(height, airspeed):
    # turbulence level defines value of wind speed in knots at 20 feet
    # turbulence_level = 15 * 0.514444 # convert speed from knots to meters per second
    turbulence_level = 15
    length_u = height / ((0.177 + 0.00823 * height) ** (0.2))
    # length_u = 1750
    sigma_w = 0.1 * turbulence_level
    sigma_u = sigma_w / ((0.177 + 0.000823 * height) ** (0.4))
    num_u = [sigma_u * (math.sqrt((2 * length_u) / (math.pi * airspeed))) * airspeed]
    den_u = [length_u, airspeed]
    H_u = signal.TransferFunction(num_u, den_u)
    return H_u


# transfer function for cross-wind
def v_transfer_function(height, airspeed):
    # turbulence level defines value of wind speed in knots at 20 feet
    # turbulence_level = 15 * 0.514444 # convert speed from knots to meters per second
    turbulence_level = 15
    length_v = height / ((0.177 + 0.00823 * height) ** (0.2))
    # length_v = 1750
    sigma_w = 0.1 * turbulence_level
    sigma_v = sigma_w / ((0.177 + 0.000823 * height) ** (0.4))
    b = sigma_v * (math.sqrt((length_v) / (math.pi * airspeed)))
    Lv_V = length_v / airspeed
    num_v = [(math.sqrt(3) * Lv_V * b), b]
    den_v = [(Lv_V ** 2), 2 * Lv_V, 1]
    H_v = signal.TransferFunction(num_v, den_v)
    return H_v


# transfer function for vertical-wind
def w_transfer_function(height, airspeed):
    # turbulence level defines value of wind speed in knots at 20 feet
    # turbulence_level = 15 * 0.514444 # convert speed from knots to meters per second
    turbulence_level = 15
    length_w = height
    # length_w = 1750
    sigma_w = 0.1 * turbulence_level
    c = sigma_w * (math.sqrt((length_w) / (math.pi * airspeed)))
    Lw_V = length_w / airspeed
    num_w = [(math.sqrt(3) * Lw_V * c), c]
    den_w = [(Lw_V ** 2), 2 * Lw_V, 1]
    H_v = signal.TransferFunction(num_w, den_w)
    return H_v



class Controller:


    def __init__(self, quadcopter: Quadcopter, params=CONTROLLER_PARAMS, ignore_yaw=True):
        self.quadcopter = quadcopter
        self.ignore_yaw = ignore_yaw # whether to control for a yaw target

        self.MOTOR_LIMITS = params['Motor_limits']
        self.TILT_LIMITS = [(params['Tilt_limits'][0] / 180.0) * 3.14, \
                            (params['Tilt_limits'][1] / 180.0) * 3.14]
        self.YAW_CONTROL_LIMITS = params['Yaw_Control_Limits']
        self.Z_LIMITS = [self.MOTOR_LIMITS[0] + params['Z_XY_offset'],
                         self.MOTOR_LIMITS[1] - params['Z_XY_offset']]
        
        self.LINEAR_P = params['Linear_PID']['P']
        self.LINEAR_I = params['Linear_PID']['I']
        self.LINEAR_D = params['Linear_PID']['D']
        self.ANGULAR_P = params['Angular_PID']['P']
        self.ANGULAR_I = params['Angular_PID']['I']
        self.ANGULAR_D = params['Angular_PID']['D']
        
        self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
        self.YAW_RATE_SCALER = params['Yaw_Rate_Scaler']
        self.reset()
        

    def reset(self):
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.target = np.zeros(3)
        self.yaw_target = 0.0
        self.thread_object = None
        self.run = True


    def wrap_angle(self, val):
        return((val + np.pi) % (2 * np.pi) - np.pi)


    def get_control(self) -> np.ndarray:
        [dest_x,    dest_y,     dest_z] = self.target
        [x,         y,          z,
         x_dot,     y_dot,      z_dot,
         theta,     phi,        gamma,       # pitch, roll, yaw
         theta_dot, phi_dot,    gamma_dot] = self.quadcopter.state

        x_error = dest_x - x
        y_error = dest_y - y
        z_error = dest_z - z

        self.xi_term += self.LINEAR_I[0] * x_error
        self.yi_term += self.LINEAR_I[1] * y_error
        self.zi_term += self.LINEAR_I[2] * z_error
        dest_x_dot = (self.LINEAR_P[0] * x_error) + (self.LINEAR_D[0] * -x_dot) + self.xi_term
        dest_y_dot = (self.LINEAR_P[1] * y_error) + (self.LINEAR_D[1] * -y_dot) + self.yi_term
        dest_z_dot = (self.LINEAR_P[2] * z_error) + (self.LINEAR_D[2] * -z_dot) + self.zi_term

        throttle = np.clip(dest_z_dot, self.Z_LIMITS[0], self.Z_LIMITS[1])
        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0] * \
                    (dest_x_dot * math.sin(gamma) - dest_y_dot * math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1] * \
                    (dest_x_dot * math.cos(gamma) + dest_y_dot * math.sin(gamma))
        dest_gamma = gamma if self.ignore_yaw else self.yaw_target
        dest_theta = np.clip(dest_theta, self.TILT_LIMITS[0], self.TILT_LIMITS[1])
        dest_phi = np.clip(dest_phi, self.TILT_LIMITS[0], self.TILT_LIMITS[1])
        theta_error = dest_theta - theta
        phi_error = dest_phi - phi
        gamma_dot_error = (self.YAW_RATE_SCALER * self.wrap_angle(dest_gamma-gamma)) - gamma_dot
        self.thetai_term += self.ANGULAR_I[0] * theta_error
        self.phii_term += self.ANGULAR_I[1] * phi_error
        self.gammai_term += self.ANGULAR_I[2] * gamma_dot_error
        x_val = self.ANGULAR_P[0] * (theta_error) + self.ANGULAR_D[0] * (-theta_dot) + self.thetai_term
        y_val = self.ANGULAR_P[1] * (phi_error) + self.ANGULAR_D[1] * (-phi_dot) + self.phii_term
        z_val = self.ANGULAR_P[2] * (gamma_dot_error) + self.gammai_term
        z_val = np.clip(z_val, self.YAW_CONTROL_LIMITS[0], self.YAW_CONTROL_LIMITS[1])
        m1 = throttle + x_val + z_val
        m2 = throttle + y_val - z_val
        m3 = throttle - x_val + z_val
        m4 = throttle - y_val - z_val
        m = np.asarray([m1, m2, m3, m4])
        m = np.clip(m, self.MOTOR_LIMITS[0], self.MOTOR_LIMITS[1])
        return m


    def update(self):
        m = self.get_control()
        self.quadcopter.set_motor_speeds(m)


    def update_target(self, target):
        self.target = target


    def update_yaw_target(self,target):
        self.yaw_target = self.wrap_angle(target)


    def thread_run(self,update_rate,time_scaling):
        update_rate = update_rate * time_scaling
        last_update = self.quadcopter.get_time()
        while(self.run==True):
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > update_rate:
                self.update()
                last_update = self.time


    def start_thread(self,update_rate=0.005,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(update_rate,time_scaling))
        self.thread_object.start()


    def stop_thread(self):
        self.run = False



class QuadcopterSupervisorEnv(gym.Env):

    # Bounds used to reset the environment
    bounds_position = (0, 10.)
    bounds_linear_rate = (-0.1, 0.1)
    bounds_orientation = (-0.25 * np.pi, 0.25 * np.pi)
    bounds_angular_rate = (-0.025 * np.pi, 0.025 * np.pi)
    # Range of inputs that the environment accepts. Can be more restrictive
    # than action_space. Inputs actions in step() are clipped to this range first:
    action_domain = (-1., 1.)
    # Action "range" is set on a quadcopter basis because each quadcopter may
    # have different motor speed limits.


    def __init__(self, ctrl: Controller, dt: float=1e-2, seed=None,
            deterministic_reset=False, simulate_takeoff=True):
        super().__init__()
        self.ctrl = ctrl
        self.dt = dt    # simulation interval
        self.max_n = 1000
        self.simulate_takeoff = simulate_takeoff
        # Error in position afforded by simulation time:
        # = max free fall velocity in bounds * simulation time step
        # self.pos_margin = np.sqrt(2 * 9.81 * np.diff(self.bounds_position)[0]) * dt
        self.pos_margin = 0.5
        self.deterministic_reset = deterministic_reset
        self.random = np.random.RandomState()
        self._seed = seed
        self.seed(self._seed)
        self.radius_core = self.ctrl.quadcopter.quad['r']  # center radius
        self.radius_arms = self.ctrl.quadcopter.quad['L']  # arm length

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32)
        )
        # These parameters are changed on reset() or at each step()
        self.n = None           # number of steps so far in episode
        self.direction = None   # unit vector pointing from start to end
        self._start_pos = None  # populated when reset() is called
        self._total_length = None # distance from start to end
        self._reset_params = None # parameter values when deterministically resetting
        self._accumulated_reward = 0.   # total reward in episode so far
        self._action_domain_span = None # domain of actions action space
        self._action_domain_min = None  # smallest action in domain
        self._action_range_span = None  # the range to which actions are mapped for PID controller
        self._action_range_min = None   # smallest action output added to PID controller


    def takeoff(self):
        target = self.ctrl.target
        self.ctrl.target = self.start
        m = self.ctrl.get_control()
        self.ctrl.quadcopter.set_motor_speeds(m)
        self.ctrl.quadcopter.update(dt=self.dt)
        self.ctrl.target = target


    def seed(self, seed):
        self.random.seed(seed)
        self.ctrl.quadcopter.random.seed(seed)


    def reset(self, position=None, linear_rate=None, orientation=None,
              angular_rate=None, target=None):
        
        self.ctrl.quadcopter.reset() # reset quadcopter time/state
        self.ctrl.reset() # reset controller PID parameters
        self.n = 0  # reset step count
        self._accumulated_reward = 0.

        if self.deterministic_reset and self._reset_params is not None:
            position, linear_rate, orientation, angular_rate, target = self._reset_params
        else:
            if position is None:
                position = self.random.rand(3) * np.diff(self.bounds_position) + np.min(self.bounds_position)
            if linear_rate is None:
                linear_rate = self.random.rand(3) * np.diff(self.bounds_linear_rate) + np.min(self.bounds_linear_rate)
            if orientation is None:
                orientation = self.random.rand(3) * np.diff(self.bounds_orientation) + np.min(self.bounds_orientation)
            if angular_rate is None:
                angular_rate = self.random.rand(3) * np.diff(self.bounds_angular_rate) + np.min(self.bounds_angular_rate)
            if target is None:
                target = self.random.rand(3) * np.diff(self.bounds_position) + np.min(self.bounds_position)

        if self.deterministic_reset:
            self.seed(self._seed)
            self._reset_params = (position, linear_rate, orientation, angular_rate, target)

        # assigning state
        self.ctrl.quadcopter.state[0:3] = position
        self.ctrl.quadcopter.state[3:6] = linear_rate
        self.ctrl.quadcopter.state[6:9] = orientation
        self.ctrl.quadcopter.state[9:12] = angular_rate
        self.ctrl.target = np.asarray(target, dtype=np.float32)
        # Reward calculation variables
        self._start_pos = self.ctrl.quadcopter.state[0:3]
        self.direction = (self.ctrl.target - self.ctrl.quadcopter.state[0:3])
        self._total_length = np.linalg.norm(self.direction) # start -> target shortest distance
        self.direction /= 1 if self._total_length==0 else self._total_length  # unit vector
        # supervisory control parameters for scaling input to units PID controller
        # can understand. Mapping is a linear function from domain to range.
        self._action_domain_span = self.action_domain[1] - self.action_domain[0]
        self._action_domain_min = self.action_domain[0]
        self._action_range_span = 2 * self.ctrl.MOTOR_LIMITS[0]
        self._action_range_min = -self.ctrl.MOTOR_LIMITS[0]
        self._action_gradient = self._action_range_span / self._action_domain_span

        if self.simulate_takeoff:
            self.takeoff()
        return self.state


    @property
    def state(self):
        state = np.zeros(12, dtype=np.float32)
        # State encodes relative position of target with respect to quadcopter
        state[0:3] = self.ctrl.target - self.ctrl.quadcopter.state[0:3] # position
        # linear rate [3:6], oriantation [6:9], angular rate [9:12]
        state[3:12] = self.ctrl.quadcopter.state[3:12]
        return state

    @property
    def start(self):
        return self._start_pos
    
    @property
    def end(self):
        return self.ctrl.target

    @property
    def relative_position(self):
        return self.ctrl.target - self.ctrl.quadcopter.state[0:3]
            
    @property
    def position(self):
        return self.ctrl.quadcopter.state[0:3]

    @property
    def velocity(self):
        return self.ctrl.quadcopter.state[3:6]

    @property
    def quadcopter(self):
        return self.ctrl.quadcopter


    def step(self, action: np.ndarray=0.):
        self.n += 1
        # Get controller action
        m = self.ctrl.get_control()  # units of motor speed
        # Apply additive supervisory correction
        # Action is the output of tanh, in range [-1, 1]. Converting to [0, self.ctrl.MOTOR_LIMITS[1]]
        # y - y1 = m * (x - x1) => y = m * (x - x1) + y1
        # Where (x1, y1) are minimum values of action domain and range respectively
        action_clipped = np.clip(action, *self.action_domain)
        action_in_motor_units = self._action_gradient \
                                    * (action_clipped - self._action_domain_min) \
                                + self._action_range_min
        m += action_in_motor_units
        # Update quadcopter
        self.ctrl.quadcopter.set_motor_speeds(m)
        self.ctrl.quadcopter.update(dt=self.dt)
        reward, done = self.reward(self.state)
        return self.state, reward, done, {}


    def reward(self, state):
        # Vector from quadcopter to target Q -> T
        qt = state[:3]
        # Distance to target: magnitude of direction vector of quad -> target
        target_dist = np.linalg.norm(qt)
        # Deviation is the shortest distance from the quadcopter to the line
        # connecting start and end (target) positions
        deviation = np.linalg.norm(np.cross(qt, self.direction))
        # Magnitude of velocity
        speed = np.linalg.norm(self.velocity)

        r = - target_dist / (1 if self._total_length==0 else self._total_length)
        end = False

        # Condition for keeping the episode continuing:
        if target_dist > self.pos_margin and self.n < self.max_n:
            if deviation > self.radius_arms:
                R, end = r * (1 + deviation), False
            elif deviation <= self.radius_arms:
                R, end = r, False
            self._accumulated_reward += R
        #Situations where episode must end:
        #- Close enough to the target
        elif target_dist <= self.pos_margin:
            R, end = 0, True
            # R, end = -self._accumulated_reward, True
            # R, end = self._total_length / self.n, True
            # return (np.abs(r) * self.n) / (speed + 1e-1), True
        # - Taking too long
        elif self.n >= self.max_n:
            R, end = 0, True
            # return r * (target_dist + speed), True
        return R, end



def plot_quadcopter(env: QuadcopterSupervisorEnv, *agents, labels=None, figsize=(8,8),
                    position=None, linear_rate=None, orientation=None,
                    angular_rate=None, target=None, resolution=10):
    # All args after *agents must be keyword arguments.
    predict_fns = []
    if len(agents) == 0:
        predict_fns.append(_make_agent(0))
    else:
        for agent in agents:
            if isinstance(agent, (int, float)):
                predict_fns.append(_make_agent(agent))
            elif agent is None:
                predict_fns.append(_make_agent(0))
            else:
                predict_fns.append(agent.predict)
    
    positions, actions, velocities, rewards = [], [], [], []
    for predict in tqdm(predict_fns, leave=False, desc='Agent #'):
        pos, act, vel, reward = [], [], [], 0
        env.seed(env._seed)
        state = env.reset(position=position, linear_rate=linear_rate,
                          orientation=orientation, angular_rate=angular_rate)
        pos.append(env.position)
        done = False
        while not done:
            action = predict(state)[0]
            state, reward, done, _ = env.step(action)
            pos.append(env.position)
            vel.append(env.velocity)
            act.append(action)
            reward += reward
        positions.append(np.asarray(pos))
        velocities.append(np.asarray(vel))
        actions.append(np.asarray(act))
        rewards.append(reward)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 1)
    ax = fig.add_subplot(gs[0:2, 0], projection='3d')
    labels = [str(i) for i in range(len(positions))] if labels is None else labels
    colors = []
    for label, pos, vel in zip(labels, positions, velocities):
        lines = ax.plot(pos[::resolution, 0], pos[::resolution, 1], pos[::resolution, 2],
                label=label, marker='.')
        colors.append(lines[0].get_color())
        ax.quiver(*pos[-1], *vel[-1], color='r')
    ax.text(*env.start, "start")
    ax.text(*env.end, "end")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # For equal data-aspect ratio along all dimensions
    ax_lims = np.asarray([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]) * 1.1
    ax.set_box_aspect(np.ptp(ax_lims, axis=1))
    ax.legend()

    ax = fig.add_subplot(gs[2:, 0])
    for i, (label, pos, act) in enumerate(zip(labels, positions, actions)):
        x = np.arange(len(act))[::resolution]
        ax.scatter(x, act[::resolution, 0], marker=4, s=15, c=colors[i], label='m0' if i==0 else '')
        ax.scatter(x, act[::resolution, 1], marker=5, s=15, c=colors[i], label='m1' if i==0 else '')
        ax.scatter(x, act[::resolution, 2], marker=6, s=15, c=colors[i], label='m2' if i==0 else '')
        ax.scatter(x, act[::resolution, 3], marker=7, s=15, c=colors[i], label='m3' if i==0 else '')

        # ax.plot(pos[::resolution, 0], label=f'{label}-x')
        # ax.plot(pos[::resolution, 1], label=f'{label}-y')
        # ax.plot(pos[::resolution, 2], label=f'{label}-z')
    ax.legend()

    plt.show()
    return positions, velocities, actions, rewards


def _make_agent(val):
    action = np.ones(4) * val
    def predict(state):
        return action, 1.
    return predict
