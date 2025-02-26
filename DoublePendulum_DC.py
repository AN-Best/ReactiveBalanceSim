import opensim
import cyipopt
import numpy as np

#Load the Model
Model = opensim.Model("Models/DoublePendulum/DoublePendulum.osim")
State = Model.initSystem()

#Extract model details
Nstates = Model.getNumStateVariables()
Ncontrols = Model.getNumControls()
Ncoords = Model.getNumCoordinates()
model_act = Model.getActuators()
Nactuators = model_act.getSize()

#Setup times for collocation points
N =  100
duration = 5.0
h = duration/N
dc_time = h*np.arange(0,N-1,1)

#Create dictionary to pass to the solver
class structype():
    pass

auxdata = structype()
auxdata.model = Model
auxdata.time = dc_time
auxdata.h = h
auxdata.Nstates = Nstates
auxdata.Ncontrols = Ncontrols
auxdata.Ncoords = Ncoords
auxdata.Nactuators = Nactuators

