import derive
from derive import time_symbol, contact_force
import numpy as np
from opty import Problem
from opty.utils import f_minus_ma
from symmeplot.matplotlib import Scene3D
import matplotlib.pyplot as plt
import sympy as sm
import os
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
num_nodes = 50
h = sm.symbols('h', real=True, positive=True)
duration = (num_nodes - 1)*h

#Need the time available in the free variable
delt = sm.Function('delt', real=True)(time_symbol)
eom = eom.col_join(sm.Matrix([delt.diff(time_symbol) - 1]))

states = coordinates + speeds + [delt]
num_states = len(states)

#Pull out the coordinates and speeds
#ax = lumbar x
#ay = lumbar y
#a = absolute torso rotation
#b = lumbar 
#c = right hip 
#d = right knee 
#e = right ankle 
#f = left hip 
#g = left knee 
#h = left ankle 
qax, qay, qa, qb, qc, qd, qe, qf, qg, qh = coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug, uh = speeds
Fax, Fay, Ta, Tb, Tc, Td, Te, Tf, Tg, Th = specified

#Set external torso force and torque to zero
traj_map = {Fax: np.zeros(num_nodes),
            Fay: np.zeros(num_nodes),
            Ta: np.zeros(num_nodes)}

#Add Bounds
bounds = {
    h: (0.1, 0.1),
    delt: (0.0, 10.0),
    qax: (0.0, 0.0),
    qay: (0.5, 2.0),
    qa: np.deg2rad((0.0, 0.0)),
    uax: (-0.1, 0.1),
    uay: (-0.1, 0.1),
}

#lumbar
bounds.update({k: (-np.deg2rad(1.0), np.deg2rad(1.0))
               for k in [qb]})
# hip
bounds.update({k: (-np.deg2rad(1.0), np.deg2rad(1.0))
               for k in [qc, qf]})
# knee
bounds.update({k: (-np.deg2rad(1.0), 0.0)
               for k in [qd, qg]})
# ankle
bounds.update({k: (-np.deg2rad(1.0), np.deg2rad(1.0))
               for k in [qe, qh]})
# all rotational speeds
bounds.update({k: (-np.deg2rad(1.0), np.deg2rad(1.0))
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
    integration_method='backward euler',
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
    np.savetxt(f'stand_{num_nodes}_nodes_solution.csv', solution,
               fmt='%.2f')

def animate(fname='animation.gif'):
    trunk, pelvis, rthigh, rshank, rfoot, lthigh, lshank, lfoot = segments

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    hip_proj = origin.locatenew('m', qax*ground.x)
    scene = Scene3D(ground, hip_proj, ax=ax)

    # creates the stick person
    scene.add_line([
        rshank.joint,
        rfoot.toe,
        rfoot.heel,
        rshank.joint,
        rthigh.joint,
        pelvis.joint,
        trunk.joint,
        trunk.mass_center,
        trunk.joint,
        pelvis.joint,
        lthigh.joint,
        lshank.joint,
        lfoot.heel,
        lfoot.toe,
        lshank.joint,
    ], color="k")

    # creates a moving ground (many points to deal with matplotlib limitation)
    scene.add_line([origin.locatenew('gl', s*ground.x)
                    for s in np.linspace(-2.0, 2.0)],
                   linestyle='--', color='tab:green', axlim_clip=True)

    # adds CoM and unit vectors for each body segment
    for seg in segments:
        scene.add_body(seg.rigid_body)

    # show ground reaction force vectors at the heels and toes, scaled to
    # visually reasonable length
    scene.add_vector(contact_force(rfoot.toe, ground, origin)/600.0,
                     rfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(rfoot.heel, ground, origin)/600.0,
                     rfoot.heel, color="tab:blue")
    scene.add_vector(contact_force(lfoot.toe, ground, origin)/600.0,
                     lfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(lfoot.heel, ground, origin)/600.0,
                     lfoot.heel, color="tab:blue")

    scene.lambdify_system(states + specified + constants)
    gait_cycle = np.vstack((
        xs,  # q, u shape(2n, N)
        np.zeros((3, len(times))),  # Fax, Fay, Ta (hand of god), shape(3, N)
        rs,  # r, shape(q, N)
        np.repeat(np.atleast_2d(np.array(list(par_map.values()))).T,
                  len(times), axis=1),  # p, shape(r, N)
    ))
    scene.evaluate_system(*gait_cycle[:, 0])

    scene.axes.set_proj_type("ortho")
    scene.axes.view_init(90, -90, 0)
    scene.plot()

    ax.set_xlim((-0.8, 0.8))
    ax.set_ylim((-0.2, 1.4))
    ax.set_aspect('equal')

    ani = scene.animate(lambda i: gait_cycle[:, i], frames=len(times),
                        interval=h_val*1000)
    ani.save(fname, fps=int(1/h_val))

animation = animate('quiet_stance.gif')