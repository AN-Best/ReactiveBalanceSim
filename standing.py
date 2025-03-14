import derive
import numpy as np
from scipy.integrate import odeint
from opty import Problem
from opty.utils import parse_free, f_minus_ma
from pydy.codegen.ode_function_generators import generate_ode_function
from pydy.viz import Scene
from collections import OrderedDict
import yaml

#Derive the equations of motion
(mass_matrix, forcing_vector, kane, constants, coordinates, speeds, specified,
 visualization_frames, ground, origin, segments) = \
    derive.derive_equations_of_motion()

eom = f_minus_ma(mass_matrix,forcing_vector,coordinates+speeds)

#Pull in the Constants
constant_values = derive.load_constants(constants,'example_constants.yml')

#Objective Function