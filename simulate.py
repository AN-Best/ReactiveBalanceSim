import derive
import numpy as np
from scipy.integrate import odeint
from pydy.codegen.ode_function_generators import generate_ode_function
from pydy.viz import Scene


#Derive the equations of motion

(mass_matrix, forcing_vector, kane, constants, coordinates, speeds, specified,
 visualization_frames, ground, origin, segments) = \
    derive.derive_equations_of_motion()

constant_values = derive.load_constants(constants,'example_constants.yml')

rhs = generate_ode_function(
    forcing_vector,
    coordinates,
    speeds,
    constants=list(constant_values.keys()),
    mass_matrix=mass_matrix,
    specifieds=specified,
    generator='cython',
    constants_arg_type='array',
    specifieds_arg_type='array',
)

args = (np.zeros(len(specified)), np.array(list(constant_values.values())))

time_vector = np.linspace(0.0, 10.0, num=1000)
initial_conditions = np.zeros(len(coordinates + speeds))
initial_conditions[1] = 2.0  # set hip above ground
trajectories = odeint(rhs, initial_conditions, time_vector, args=args)

scene = Scene(ground, origin, *visualization_frames)
scene.states_symbols = coordinates + speeds
scene.constants = constant_values
scene.states_trajectories = trajectories
scene.times = time_vector

scene.display()