import opensim
from cyipopt import minimize_ipopt
import numpy as np

#Load the Model
Model = opensim.Model("Models/DoublePendulum/DoublePendulum.osim")
State = Model.initSystem()

#Extract model details
Nstates = Model.getNumStateVariables()
Ncontrols = Model.getNumControls()
Ncoords = Model.getNumCoordinates()
model_act = Model.getActuators()
print(model_act)
Nactuators = model_act.getSize()

#Setup times for collocation points
N =  25
duration = 5.0
h = duration/N
dc_time = h*np.arange(0,N-1,1)

#Create dictionary to pass to the solver
class structype():
    pass

auxdata = structype()
auxdata.Model = Model
auxdata.State = State
auxdata.time = dc_time
auxdata.N = N
auxdata.h = h
auxdata.Nstates = Nstates
auxdata.Ncontrols = Ncontrols
auxdata.Ncoords = Ncoords
auxdata.Nactuators = Nactuators

#Formulate initail guess
X0 = np.zeros(((Nstates+Ncontrols)*N,1))
for i in range(N):
    X0[Nstates*i:Nstates*(i+1),0] = 0.0001*np.random.rand(Nstates)
    X0[Nstates*N + Ncontrols*i:Nstates*N + Ncontrols*(i+1),0] = 0.0001*np.random.rand(Ncontrols)


#Define the objective function
def objective(X,auxdata):
    Nstates = auxdata.Nstates
    Ncontrols = auxdata.Ncontrols
    N = auxdata.N

    #Extract states and controls
    states = np.zeros((N,Nstates))
    controls = np.zeros((N,Ncontrols))
    for i in range(Nstates):
        states[i,:] = X[i*Nstates:(i+1)*Nstates,0]
    for i in range(Ncontrols):
        controls[i,:] = X[N*Nstates + i*Ncontrols:N*Nstates + (i+1)*Ncontrols,0]

    #Calculate the objective function
    Goal = np.pi*np.ones((N,Nstates))
    J = np.sum(np.sum(Goal-states)**2)

    return J


#Define the constraints
def constraints(X,auxdata):
    Nstates = auxdata.Nstates
    Ncontrols = auxdata.Ncontrols
    N = auxdata.N
    Model = auxdata.Model
    State = Model.initSystem()
    h = auxdata.h

    #Extract states and controls
    states = np.zeros((N,Nstates))
    controls = np.zeros((N,Ncontrols))
    for i in range(Nstates):
        states[i,:] = X[i*Nstates:(i+1)*Nstates,0]
    for i in range(Ncontrols):
        controls[i,:] = X[N*Nstates + i*Ncontrols:N*Nstates + (i+1)*Ncontrols,0]

    #Query OpenSim for the state derivatives
    x_dot = np.zeros((N-1,Nstates))
    for i in range(N-1):
        Model.setStateValues
        

    #Use backward euler to find derivative of collocation states
    states_dot = np.zeros((N-1,Nstates))
    for i in range(N-1):
        states_dot[i,:] = (states[i+1,:]-states[i,:])/h

    #Calculate the constraint violations
    C_Matrix = states_dot - x_dot
    C = np.reshape(C_Matrix,(N-1)*Nstates)

    return C

#Test Objective and Constraint Functions
J_test = objective(X0,auxdata)
print(J_test)
C_test = constraints(X0,auxdata)
    




    


