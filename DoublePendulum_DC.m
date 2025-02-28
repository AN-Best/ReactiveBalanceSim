%% Direct Collocation Example for Double Pendulum
clc;
import org.opensim.modeling.*

%% Load model and extract details

osModel = Model("Models/DoublePendulum/DoublePendulum.osim");
osState = osModel.initSystem();

Nstates = osModel.getNumStateVariables();
Ncontrols = osModel.getNumControls();
Ncoords = osModel.getNumCoordinates();
model_act = osModel.getActuators();
Nactuators = model_act.getSize();
model_states = osModel.getStateVariableNames();

%% Setup Collocation Parameters
N = 25;                 % # of nodal points
duration = 5.0;         % time in sec to complete task
h = duration/(N-1);     % time interval between nodes
dc_time = h*(0:N-1)';  % list of time points (temporal grid)
qGoal = pi/2;            % Goal position for the pendulum   
vGoal = 0;              % Goal velocity for the pendulum
%% Auxiliary data to be passed to the optimizer

auxdata.model = osModel;
auxdata.state = osState;
auxdata.time = dc_time; 
auxdata.N = N;
auxdata.h = h;
auxdata.Nstates = Nstates;
auxdata.Ncontrols = Ncontrols;
auxdata.Nactuators = Nactuators;
auxdata.Ncoords = Ncoords;
auxdata.qGoal = qGoal;
auxdata.vGoal = vGoal;
auxdata.model_act = model_act;
auxdata.model_states = model_states;

%% Formulate an initial guess

X0 = 0.01*rand((Nstates+Ncontrols)*N,1);

%% Test the Objective Function and the Constraints

Jtest = ObjectiveFunction(X0,auxdata);
[ctest,ceqtest] = Constraints(X0,auxdata);

%% Optimize the problem


OPTopts = optimoptions('fmincon','Algorithm','interior-point',...
                            'MaxFunctionEvaluations',5e5,...
                            'MaxIterations',1000,...
                            'OptimalityTolerance',1e-6,...
                            'StepTolerance',1e-15,...
                            'Display','iter-detailed');     
                       
OPTopts.PlotFcn = {@optimplotfval,@optimplotconstrviolation};

[X_opt,fval,exitflag,output] = fmincon(@(X) ObjectiveFunction(X,auxdata),...
                                X0,[],[],[],[],[],[],...
                                @(X) Constraints(X,auxdata),OPTopts);

%% Objective Function

function J = ObjectiveFunction(X,auxdata)

    Nstates = auxdata.Nstates;
    Ncontrols = auxdata.Ncontrols;
    N = auxdata.N;
    qGoal = auxdata.qGoal;
    vGoal = auxdata.vGoal;

    % Extract states and controls from X
    states = zeros(N,Nstates);
    controls = zeros(N,Ncontrols);
    for i = 1:N
        states(i,:) = X((i-1)*Nstates+1:i*Nstates);
        controls(i,:) = X(N*Nstates+(i-1)*Ncontrols+1:N*Nstates+i*Ncontrols);
    end

    % Compute the objective function
    Goal_Matrix = [qGoal*ones(N,Nstates/2), vGoal*ones(N,Nstates/2)];
    J = sum(sum((states-Goal_Matrix).^2));

end

%% Constraints Function

function [c,ceq] = Constraints(X,auxdata)

    import org.opensim.modeling.*
    
    Nstates = auxdata.Nstates;
    Ncontrols = auxdata.Ncontrols;
    N = auxdata.N;
    osModel = auxdata.model;
    osState = auxdata.state;
    model_states = auxdata.model_states;
    h = auxdata.h;

    %No inequality constraints
    c = [];

    % Extract states and controls from X
    states = zeros(N,Nstates);
    controls = zeros(N,Ncontrols);
    for i = 1:N
        states(i,:) = X((i-1)*Nstates+1:i*Nstates);
        controls(i,:) = X(N*Nstates+(i-1)*Ncontrols+1:N*Nstates+i*Ncontrols);
    end

    actuatorControls = Vector(1, 0.0);
    modelControls = osModel.updControls(osState);
    %Query Opensim for the state derivatives
    xdot = zeros(N-1,Nstates);
    for i = 1:N-1
        %Update the state
        for j = 0:1:Nstates-1
            osModel.setStateVariableValue(osState,model_states.get(j),states(i,j+1));
        end
        %Update the control
        for j = 0:1:Ncontrols-1
            actuatorControls.set(0, controls(i,j+1));
            osModel.updActuators().get(j).addInControls(actuatorControls,modelControls);
        end
        %Calculate the state derivatives
        for j = 0:1:Nstates-1
            xdot(i,j+1) = osModel.getStateVariableDerivativeValue(osState,model_states.get(j));
        end
    end

    %Use backward Euler to compute the constraint violation
    states_dot = zeros(N-1,Nstates);
    for i = 1:N-1
        states_dot(i,:) = (states(i+1,:)-states(i,:))/h;
    end

    %Equality constraints
    ceq = reshape(states_dot - xdot,[],1);

end
