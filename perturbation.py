import derive
from derive import time_symbol, contact_force
import numpy as np
from opty import Problem
from opty.utils import f_minus_ma, parse_free, create_objective_function
from symmeplot.matplotlib import Scene3D
import matplotlib.pyplot as plt
import sympy as sm
import os

#Derive the equations of motion
(mass_matrix, forcing_vector, kane, constants, coordinates, speeds, specified,
 visualization_frames, ground, origin, segments) = \
    derive.derive_equations_of_motion()

eom = f_minus_ma(mass_matrix,forcing_vector,coordinates+speeds)

#Pull in the Constants
par_map = derive.load_constants(constants,'example_constants.yml')

states = coordinates + speeds 
num_states = len(states)

#Discretization characteristics
duration = 2.0
h = 0.005
num_nodes = int(duration/h) + 1

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
Fax, Fay,v_sled , Ta, Tb, Tc, Td, Te, Tf, Tg, Th = specified

#Create sled velocity
sled_velocity = np.zeros(num_nodes)
sled_velocity = np.linspace(0,-0.5,num_nodes)

#Set external torso force and torque to zero and add sled velocity
traj_map = {Fax: np.zeros(num_nodes),
            Fay: np.zeros(num_nodes),
            v_sled: sled_velocity,
            Ta: np.zeros(num_nodes)
            }

#Add Bounds
bounds = {
    qax: (-10.0, 10.0),
    qay: (0.5, 1.5),
    qa: np.deg2rad((-60.0, 60.0)),
    uax: (-10.0, 10.0),
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
# ankle
bounds.update({k: (-np.deg2rad(30.0), np.deg2rad(30.0))
               for k in [qe, qh]})
# all rotational speeds
bounds.update({k: (-np.deg2rad(400.0), np.deg2rad(400.0))
               for k in [ua, ub, uc, ud, ue, uf, ug, uh]})
# all joint torques
bounds.update({k: (-100.0, 100.0)
               for k in [Tb, Tc, Td, Te, Tf, Tg, Th]})

#Set initial condition constraints
instance_constraints = (
    qax.func(0*h) - 0.0,
    qay.func(0*h) - 1.24,
    qa.func(0*h) - 0.0,
    qb.func(0*h) - 0.0,
    qc.func(0*h) - 0.0,
    qd.func(0*h) - 0.0,
    qe.func(0*h) - 0.0,
    qf.func(0*h) - 0.0,
    qg.func(0*h) - 0.0,
    qh.func(0*h) - 0.0,
    uax.func(0*h) - 0.0,
    uay.func(0*h) - 0.0,
    ua.func(0*h) - 0.0,
    ub.func(0*h) - 0.0,
    uc.func(0*h) - 0.0,
    ud.func(0*h) - 0.0,
    ue.func(0*h) - 0.0,
    uf.func(0*h) - 0.0,
    ug.func(0*h) - 0.0,
    uh.func(0*h) - 0.0
)


objective = sm.Integral(qax**2 + (qay-1.24)**2 + qa**2,time_symbol)

state_symbols = (qax, qay, qa, qb, qc, qd, qe, qf, qg, qh,
                 uax, uay, ua, ub, uc, ud, ue, uf, ug, uh)
specified_symbols = (Tb, Tc, Td, Te, Tf, Tg, Th)

sm.pprint(objective)

obj, obj_grad = create_objective_function(objective=objective,
                                          state_symbols=state_symbols,
                                          unknown_input_trajectories=specified_symbols, 
                                          unknown_parameters=tuple(),
                                          num_collocation_nodes=num_nodes,
                                          node_time_interval=h,
                                          time_symbol=time_symbol,
                                          integration_method='midpoint')


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

prob.add_option('max_iter',10000)
    
initial_guess = np.ones(prob.num_free)

#Optimize
solution, info = prob.solve(initial_guess)

#Pull out solution trajectory
xs, rs, _ = prob.parse_free(solution)

times = np.linspace(0.0, (num_nodes - 1)*h, num=num_nodes)
if info['status'] in (0, 1):
    np.savetxt(f'perturb_{num_nodes}_nodes_solution.csv', solution,
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
    scene.add_vector(contact_force(rfoot.toe, ground, origin,v_sled)/600.0,
                     rfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(rfoot.heel, ground, origin,v_sled)/600.0,
                     rfoot.heel, color="tab:blue")
    scene.add_vector(contact_force(lfoot.toe, ground, origin,v_sled)/600.0,
                     lfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(lfoot.heel, ground, origin,v_sled)/600.0,
                     lfoot.heel, color="tab:blue")

    scene.lambdify_system(states + specified + constants)
    gait_cycle = np.vstack((
        xs,  # q, u shape(2n, N)
        np.zeros((4, len(times))),  # Fax, Fay, Ta (hand of god), v_sled shape(4, N)
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
                        interval=h*1000)
    ani.save(fname, fps=int(1/h))

animation = animate('perturb.gif')