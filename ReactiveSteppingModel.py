import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty import Problem
from opty.utils import parse_free
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches

# Create Reference Frames
W, P, HAT, RT, RS, RF, LT, LS,LF = sm.symbols('W P HAT RT RS RF LT LS LF',cls=me.ReferenceFrame)
#Time variable
t = me.dynamicsymbols._t
# Create Points for CoM of each segment
O, comP, comHAT, comRT, comRS, comRF, comLT, comLS, comLF = sm.symbols('O comP comHAT comRT comRS comRF comLT comLS comLF', cls=me.Point)
# Create Point for contacts
RightToe, RightHeel, LeftHeel, LeftToe = sm.symbols('RightToe RightHeel LeftHeel LeftToe',cls = me.Point)
# Set the velocity of the origin
O.set_vel(W,0)
# Create the generalized coordinates
qPx, qPy, qPz, qL, qRH, qRK, qRA, qLH, qLK, qLA = me.dynamicsymbols('qPx qPy qPz qL qRH qRK qRA qLH qLK qLA')
# Create the generalized speeds
vPx, vPy, vPz, vL, vRH, vRK, vRA, vLH, vLK, vLA = me.dynamicsymbols('vPx vPy vPz vL vRH vRK vRA vLH vLK vLA')
# Create generalized forces
uL, uRH, uRK, uRA, uLH, uLK, uLA = me.dynamicsymbols('uL uRH uRK uRA uLH uLK uLA')
# Create Constants
mFoot,mShank,mThigh,mPelvis,mHAT = sm.symbols('mFoot mShank mThigh mPelvis mHAT')
Ifoot,Ishank,Ithigh,Ipelvis,IHAT = sm.symbols('Ifoot Ishank Ithigh Ipelvis Ihat')
hfoot,hshank,hthigh,hpelvis,hHAT = sm.symbols('hfoot hshank hthigh hpelvis hHAT')
cfootx,cfooty,cshank,cthigh,cpelvis,cHAT = sm.symbols('cfootx cfooty cshank cthigh cpelvis cHAT')
Lfoot = sm.symbols('Lfoot')   
g = sm.symbols('g')

system = me.System(W)

# Define the Rigid Bodies
WorldGround = me.RigidBody('RB_WorldGround',
                           frame = W)

RB_Pelvis = me.RigidBody('RB_Pelvis',
                      masscenter=comP,
                      frame = P,
                      mass = mPelvis,
                      inertia = (me.inertia(P,ixx = 0,
                                            iyy = 0,
                                            izz = Ipelvis,
                                            ixy = 0,
                                            iyz = 0,
                                            izx = 0),comP))

RB_HAT = me.RigidBody('RB_HAT',
                   masscenter=comHAT,
                   frame = HAT,
                   mass = mHAT,
                   inertia = (me.inertia(HAT,ixx = 0,
                                         iyy = 0,
                                         izz = IHAT,
                                         ixy = 0,
                                         iyz = 0,
                                         izx = 0),comHAT))

RB_RightThigh = me.RigidBody('RB_RightThigh',
                         masscenter=comRT,
                         frame = RT,
                         mass = mThigh,
                         inertia = (me.inertia(RT,ixx = 0,
                                               iyy = 0,
                                               izz = Ithigh,
                                               ixy = 0,
                                               iyz = 0,
                                               izx = 0),comRT))

RB_RightShank = me.RigidBody('RB_RightShank',
                            masscenter=comRS,
                            frame = RS,
                            mass = mShank,
                            inertia = (me.inertia(RS,ixx = 0,
                                                iyy = 0,
                                                izz = Ishank,
                                                ixy = 0,
                                                iyz = 0,
                                                izx = 0),comRS))

RB_RightFoot = me.RigidBody('RB_RightFoot',
                            masscenter = comRF,
                            frame = RF,
                            mass = mFoot,
                            inertia = (me.inertia(RF, ixx = 0,
                                                iyy = 0,
                                                izz = Ifoot,
                                                ixy = 0,
                                                iyz = 0,
                                                izx = 0),comRF))

RB_LeftThigh = me.RigidBody('RB_LeftThigh',
                         masscenter=comLT,
                         frame = LT,
                         mass = mThigh,
                         inertia = (me.inertia(LT,ixx = 0,
                                               iyy = 0,
                                               izz = Ithigh,
                                               ixy = 0,
                                               iyz = 0,
                                               izx = 0),comLT))

RB_LeftShank = me.RigidBody('RB_LefttShank',
                            masscenter=comLS,
                            frame = LS,
                            mass = mShank,
                            inertia = (me.inertia(LS,ixx = 0,
                                                iyy = 0,
                                                izz = Ishank,
                                                ixy = 0,
                                                iyz = 0,
                                                izx = 0),comLS))

RB_LeftFoot = me.RigidBody('RB_LeftFoot',
                            masscenter = comLF,
                            frame = LF,
                            mass = mFoot,
                            inertia = (me.inertia(LF, ixx = 0,
                                                iyy = 0,
                                                izz = Ifoot,
                                                ixy = 0,
                                                iyz = 0,
                                                izx = 0),comLF))


system.add_bodies(WorldGround,RB_Pelvis,RB_HAT,
                  RB_RightThigh,RB_RightShank,RB_RightFoot,
                  RB_LeftThigh,RB_LeftShank,RB_LeftFoot)


#Define Joints
slider_Px = me.PrismaticJoint('slider_Px',
                              parent = WorldGround,
                              child = RB_Pelvis,
                              coordinates=qPx,
                              speeds = vPx,
                              joint_axis=W.x)

slider_Py = me.PrismaticJoint('slider_Py',
                              parent = WorldGround,
                              child = RB_Pelvis,
                              coordinates=qPy,
                              speeds = vPy,
                              joint_axis=W.y)

revolute_Pz = me.PinJoint('revolute_Pz',
                          parent = WorldGround,
                          child = RB_Pelvis,
                          coordinates=qPz,
                          speeds = vPz,
                          joint_axis=W.z)

lumbar = me.PinJoint('lumbar',
                     parent = RB_Pelvis,
                     child = RB_HAT,
                     coordinates = qL,
                     speeds = vL,
                     parent_point = (1-cpelvis)*hpelvis*P.y,
                     child_point = -1*cHAT*hHAT*HAT.y,
                     parent_interframe= P,
                     child_interframe = HAT)

right_hip = me.PinJoint('right_hip',
                        parent = RB_Pelvis,
                        child = RB_RightThigh,
                        coordinates = qRH,
                        speeds = vRH,
                        parent_point = -1*cpelvis*hpelvis*P.y,
                        child_point = (1-cthigh)*hthigh*RT.y,
                        parent_interframe = P,
                        child_interframe = RT)

right_knee = me.PinJoint('right_knee',
                         parent = RB_RightThigh,
                         child = RB_RightShank,
                         coordinates = qRK,
                         speeds = vRK,
                         parent_point = -1*cthigh*hthigh*RT.y,
                         child_point = (1-cshank)*hshank*RS.y,
                         parent_interframe = RT,
                         child_interframe = RS)

right_ankle = me.PinJoint('right_ankle',
                          parent = RB_RightShank,
                          child = RB_RightFoot,
                          coordinates=qRA,
                          speeds = vRA,
                          parent_point = -cshank*hshank*RS.y,
                          child_point = (1-cfootx)*Lfoot*RF.x + cfooty*hfoot*RF.y,
                          parent_interframe = RS,
                          child_interframe=RF)

left_hip = me.PinJoint('left_hip',
                        parent = RB_Pelvis,
                        child = RB_LeftThigh,
                        coordinates = qLH,
                        speeds = vLH,
                        parent_point = -1*cpelvis*hpelvis*P.y,
                        child_point = (1-cthigh)*hthigh*LT.y,
                        parent_interframe = P,
                        child_interframe = LT)

left_knee = me.PinJoint('left_knee',
                         parent = RB_LeftThigh,
                         child = RB_LeftShank,
                         coordinates = qLK,
                         speeds = vLK,
                         parent_point = -1*cthigh*hthigh*LT.y,
                         child_point = (1-cshank)*hshank*LS.y,
                         parent_interframe = LT,
                         child_interframe = LS)

left_ankle = me.PinJoint('left_ankle',
                          parent = RB_LeftShank,
                          child = RB_LeftFoot,
                          coordinates=qLA,
                          speeds = vLA,
                          parent_point = -cshank*hshank*LS.y,
                          child_point = (1-cfootx)*Lfoot*LF.x + cfooty*hfoot*LF.y,
                          parent_interframe = LS,
                          child_interframe=LF)


system.add_joints(slider_Px,slider_Py,revolute_Pz,
                  lumbar,
                  right_hip,right_knee,right_ankle,
                  left_hip, left_knee, left_ankle)


#Apply Loads
#Gravity
system.apply_uniform_gravity(-g*W.y)
#Joint Torque
system.add_actuators(me.TorqueActuator(uL,W.z,HAT,P),
                     me.TorqueActuator(uRH,W.z,P,RT),
                     me.TorqueActuator(uRK,W.z,RT,RS),
                     me.TorqueActuator(uRA,W.z,RS,RF),
                     me.TorqueActuator(uLH,W.z,P,LT),
                     me.TorqueActuator(uLK,W.z,LT,LS),
                     me.TorqueActuator(uLA,W.z,LS,LF))

system.form_eoms(explicit_kinematics = True)
mass_matrix = system.mass_matrix_full
forcing_vector = system.forcing_full

#










