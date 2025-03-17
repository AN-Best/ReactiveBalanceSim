import derive
from derive import time_symbol, contact_force
import numpy as np
from scipy.integrate import odeint
from opty import Problem
from opty.utils import parse_free, f_minus_ma
from symmeplot.matplotlib import Scene3D
import matplotlib.pyplot as plt
import sympy as sm

#Derive the equations of motion
(mass_matrix, forcing_vector, kane, constants, coordinates, speeds, specified,
 visualization_frames, ground, origin, segments) = \
    derive.derive_equations_of_motion()

eom = f_minus_ma(mass_matrix,forcing_vector,coordinates+speeds)
print(sm.count_ops(eom))

#Pull in the Constants
par_map = derive.load_constants(constants,'example_constants.yml')

#Discretization characteristics
speed = 0.0 # m/s
num_nodes = 40
h = sm.symbols('h', real=True, positive=True)
duration = (num_nodes - 1)*h

#Need the time available in the free variable
delt = sm.Function('delt', real=True)(time_symbol)
eom = eom.col_join(sm.Matrix([delt.diff(time_symbol) - 1]))

states = coordinates + speeds + [delt]
num_states = len(states)

#Pull out the coordinates and speeds
#ax = lumbar position x
#ay = lumbar position y
#a = absolute torso angle
#b = lumbar angle
#c = right hip angle
#d = right knee angle
#e = right ankle angle
#f = left hip angle
#g = left knee angle
#h = left ankle angle
qax, qay, qa, qb, qc, qd, qe, qf, qg, qh = coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug, uh = speeds
Fax, Fay, Ta, Tb, Tc, Td, Te, Tf, Tg, Th = specified

#Set external torso force and torque to zero
traj_map = {Fax: np.zeros(num_nodes),
            Fay: np.zeros(num_nodes),
            Ta: np.zeros(num_nodes)}

#Add Bounds
bounds = {
    h: (0.001, 0.1),
    delt: (0.0, 10.0),
    qax: (0.0, 10.0),
    qay: (0.5, 2.0),
    qa: np.deg2rad((-60.0, 60.0)),
    uax: (0.0, 10.0),
    uay: (-10.0, 10.0),
}

#lumbar
bounds.update({k: (-np.deg2rad(30.0), np.deg2rad(30.0))
               for k in [qb]})
# hip
bounds.update({k: (-np.deg2rad(40.0), np.deg2rad(40.0))
               for k in [qc, qf]})
# knee
bounds.update({k: (-np.deg2rad(60.0), 0.0)
               for k in [qd, qg]})
# foot
bounds.update({k: (-np.deg2rad(30.0), np.deg2rad(30.0))
               for k in [qe, qh]})
# all rotational speeds
bounds.update({k: (-np.deg2rad(400.0), np.deg2rad(400.0))
               for k in [ua, ub, uc, ud, ue, uf, ug, uh]})
# all joint torques
bounds.update({k: (-100.0, 100.0)
               for k in [Tb, Tc, Td, Te, Tf, Tg, Th]})

#Instance Constraints
instance_constraints = (
    delt.func(0*h) - 0.0,
    qax.func(0*h) - 0.0,
    qax.func(duration) - speed*delt.func(duration),
    qay.func(0*h) - qay.func(duration),
    qa.func(0*h) - qa.func(duration),
    qb.func(0*h) - qb.func(duration),
    qc.func(0*h) - qf.func(duration),
    qd.func(0*h) - qg.func(duration),
    qe.func(0*h) - qh.func(duration),
    qf.func(0*h) - qc.func(duration),
    qg.func(0*h) - qd.func(duration),
    qh.func(0*h) - qe.func(duration),
    uax.func(0*h) - uax.func(duration),
    uay.func(0*h) - uay.func(duration),
    ua.func(0*h) - ua.func(duration),
    ub.func(0*h) - ub.func(duration),
    uc.func(0*h) - uf.func(duration),
    ud.func(0*h) - ug.func(duration),
    ue.func(0*h) - uh.func(duration),
    uf.func(0*h) - uc.func(duration),
    ug.func(0*h) - ud.func(duration),
    uh.func(0*h) - ue.func(duration),  
)

#Objective function and gradient
def obj(free):
    """Minimize the sum of the squares of the control torques."""
    T, h = free[num_states*num_nodes:-1], free[-1]
    return h*np.sum(T**2)


def obj_grad(free):
    T, h = free[num_states*num_nodes:-1], free[-1]
    grad = np.zeros_like(free)
    grad[num_states*num_nodes:-1] = 2.0*h*T
    grad[-1] = np.sum(T**2)
    return grad

#Create the problem
prob = Problem(
    obj,
    obj_grad,
    eom,
    states,
    num_nodes,
    h,
    known_parameter_map=par_map,
    known_trajectory_map=traj_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    integration_method='midpoint',
    time_symbol=time_symbol,
    parallel=True
)

#Set an initial guess
initial_guess = np.ones(prob.num_free)

#Optimize
solution, info = prob.solve(initial_guess)

#Pull out solution trajectory
xs, rs, _, h_val = prob.parse_free(solution)
times = np.linspace(0.0, (num_nodes - 1)*h_val, num=num_nodes)
if info['status'] in (0, 1):
    np.savetxt(f'human_gait_{num_nodes}_nodes_solution.csv', solution,
               fmt='%.2f')
