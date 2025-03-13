# external libraries
import sympy as sy
from sympy import symbols, exp, acos, pi, Function, Abs
import sympy.physics.mechanics as me
from pydy.viz import VisualizationFrame, Cylinder, Sphere

time_symbol = symbols('t',real=True)
me.dynamicsymbols._t = time_symbol

sym_kwargs = {'positive': True,
              'real': True}

def time_varying(sym_string):
    funcs = symbols(sym_string,cls=Function,real=True)

    try:
        return tuple([f(time_symbol) for f in funcs])
    except TypeError:
        return funcs(time_symbol)
    
#Generic Body Segment
class BodySegment(object):
    viz_sphere_radius = 0.07
    viz_cylinder_radius = 0.035

    def __init__(self, label, description,parent_reference_frame,
                 origin_joint, joint_description, inertial_frame):
        
        self.label = label
        self.description = description
        self.parent_reference_frame = parent_reference_frame
        self.origin_joint = origin_joint
        self.joint_description = joint_description
        self.inertial_frame = inertial_frame

        self._create_symbols()
        self._kinematic_differential_equations()
        self._orient()
        self._set_angular_velocity()
        self._locate_joint()
        self._locate_mass_center()
        self._set_linear_velocities()
        self._inertia_dyadic()
        self._create_rigid_body()
        self._joint_torque()
        self._gravity() 

    def _create_symbols(self):
        """Generates all of the SymPy symbols and functions of time
        associated with this segment."""

        subscript = self.label.lower()

        self.t = time_symbol

        # constants
        self.g = symbols('g', **sym_kwargs)
        self.mass_symbol = symbols('m{}'.format(subscript), **sym_kwargs)
        self.inertia_symbol = \
            symbols('i{}'.format(subscript), **sym_kwargs)
        self.length_symbol = \
            symbols('l{}'.format(subscript), **sym_kwargs)
        self.mass_center_x_symbol = symbols('x{}'.format(subscript), real=True)
        self.mass_center_y_symbol = symbols('y{}'.format(subscript), real=True)

        self.constants = [self.g,
                            self.mass_symbol,
                            self.inertia_symbol,
                            self.length_symbol,
                            self.mass_center_x_symbol,
                            self.mass_center_y_symbol]

        # functions of time
        self.generalized_coordinate_symbol = \
            time_varying('q{}'.format(subscript))
        self.generalized_coordinate_derivative_symbol = \
            self.generalized_coordinate_symbol.diff(self.t)
        self.generalized_speed_symbol = time_varying('u{}'.format(subscript))
        self.joint_torque_symbol = time_varying('T{}'.format(subscript))

    def _kinematic_differential_equations(self):
        """Creates a list of the kinematic differential equations. This is the
        simple definition:

        0 = \dot{q}_i - u_i

        """
        self.kinematic_equations = \
            [self.generalized_coordinate_derivative_symbol -
             self.generalized_speed_symbol]
    
    def _orient(self):
        """Generates and orients the segment's reference frame relative to
        the parent reference frame by body fixed simple rotation about the
        generalized coordinate."""
        self.reference_frame = \
            self.parent_reference_frame.orientnew(
                self.label, 'Axis', (self.generalized_coordinate_symbol,
                                     self.parent_reference_frame.z))


    def _set_angular_velocity(self):
        """Sets the angular velocity with the generalized speed."""
        self.reference_frame.set_ang_vel(self.parent_reference_frame,
                                         self.generalized_speed_symbol *
                                         self.parent_reference_frame.z)

    def _locate_joint(self):
        """Creates a point with respect to the origin joint for the next
        joint in the segment. This assumes that new joint is in the negative
        y direction with repect to the origin joint."""
        self.joint = self.origin_joint.locatenew(self.joint_description,
                                                 -self.length_symbol *
                                                 self.reference_frame.y)

    def _locate_mass_center(self):
        """Creates a point with respect the origin joint for the mass center
        of the segment."""
        self.mass_center = self.origin_joint.locatenew(
            '{} mass center'.format(self.description),
            self.mass_center_x_symbol * self.reference_frame.x +
            self.mass_center_y_symbol * self.reference_frame.y)

    def _set_linear_velocities(self):
        """Sets the linear velocities of the mass center and new joint."""
        self.mass_center.v2pt_theory(self.origin_joint, self.inertial_frame,
                                     self.reference_frame)
        self.joint.v2pt_theory(self.origin_joint, self.inertial_frame,
                               self.reference_frame)

    def _inertia_dyadic(self):
        """Creates an inertia dyadic for the segment."""
        self.inertia_dyadic = me.inertia(self.reference_frame, 0, 0,
                                         self.inertia_symbol)

    def _create_rigid_body(self):
        """Creates a rigid body for the segment."""
        self.rigid_body = me.RigidBody(self.description, self.mass_center,
                                       self.reference_frame,
                                       self.mass_symbol,
                                       (self.inertia_dyadic, self.mass_center))

    def _joint_torque(self):
        """Creates the joint torque vector acting on the segment."""
        self.torque = self.joint_torque_symbol * self.reference_frame.z
        # TODO : add in passive joint stiffness and damping

    def _gravity(self):
        """Creates the gravitational force vector acting on the segment."""
        self.gravity = -self.mass_symbol * self.g * self.inertial_frame.y

    def visualization_frames(self):
        """Returns visualization frames for the animation of the system.
        The requires numerical values of the cylinders and spheres."""

        viz_frames = []

        cylinder = Cylinder(color='red',
                            name=self.label,
                            length=self.length_symbol,
                            radius=self.viz_cylinder_radius)

        center_point = self.origin_joint.locatenew('Cylinder Center',
                                                   -self.length_symbol / 2 *
                                                   self.reference_frame.y)

        viz_frames.append(VisualizationFrame('VizFrame',
                                             self.reference_frame,
                                             center_point, cylinder))

        viz_frames.append(VisualizationFrame('OriginJointFrame',
                                             self.reference_frame,
                                             self.origin_joint,
                                             Sphere(color='blue',
                                                    name=self.label + '_joint',
                                                    radius=self.viz_sphere_radius)))

        return viz_frames


# Trunk Segment
class TrunkSegment(BodySegment):
    def __init__(self, *args):
        super(TrunkSegment, self).__init__(*args)
        self._trunk_extra_kinematic_equations()

    def _create_symbols(self):
        super(TrunkSegment, self)._create_symbols()
        # TODO : Format these with the subscript instead of a directly.
        self.qa = time_varying('qax, qay')
        self.ua = time_varying('uax, uay')
        self.constants.remove(self.length_symbol)
        del self.length_symbol

    def _trunk_extra_kinematic_equations(self):
        qaxd, qayd = [f.diff(self.t) for f in self.qa]
        self.kinematic_equations += [self.ua[0] - qaxd, self.ua[1] - qayd]

    def _locate_joint(self):
        """The trunk only has one joint, the hip, there is no other point."""
        # This locates the hip joint relative to the ground origin point.
        self.joint = self.origin_joint.locatenew(self.joint_description,
                                                 self.qa[0] *
                                                 self.inertial_frame.x +
                                                 self.qa[1] *
                                                 self.inertial_frame.y)

    def _locate_mass_center(self):
        """Creates a point with respect the hip joint for the mass center
        of the segment."""
        self.mass_center = self.joint.locatenew(
            '{} mass center'.format(self.description),
            self.mass_center_x_symbol * self.reference_frame.x +
            self.mass_center_y_symbol * self.reference_frame.y)

    def _set_linear_velocities(self):
        """Sets the linear velocities of the mass center and new joint."""
        # The joint is the hip. The origin joint is the ground's origin.
        self.joint.set_vel(self.inertial_frame, self.ua[0] *
                           self.inertial_frame.x + self.ua[1] *
                           self.inertial_frame.y)
        self.mass_center.v2pt_theory(self.joint, self.inertial_frame,
                                     self.reference_frame)

    def visualization_frames(self):
        """This should go from the hip to the mass center."""

        viz_frames = []

        hip_to_mc_vector = self.mass_center.pos_from(self.joint)

        cylinder = Cylinder(color='red', name=self.label,
                            length=hip_to_mc_vector.magnitude(),
                            radius=self.viz_cylinder_radius)

        center_point = \
            self.joint.locatenew('Cylinder Center',
                                 hip_to_mc_vector.magnitude() / 2 *
                                 hip_to_mc_vector.normalize())

        viz_frames.append(VisualizationFrame('VizFrame',
                                             self.reference_frame,
                                             center_point, cylinder))

        viz_frames.append(VisualizationFrame('MassCenterFrame',
                                             self.reference_frame,
                                             self.mass_center,
                                             Sphere(color='blue',
                                                    name=self.label + '_joint',
                                                    radius=self.viz_sphere_radius)))

        return viz_frames
    
class FootSegment(BodySegment):

    viz_sphere_radius = 0.03
    viz_cylinder_radius = 0.01

    def __init__(self, *args):
        super(FootSegment, self).__init__(*args)
        self._locate_foot_points()
        self._set_foot_linear_velocities()

    def _create_symbols(self):
        super(FootSegment, self)._create_symbols()
        self.heel_distance = symbols('hx{}'.format(self.label.lower()),
                                     real=True)
        self.toe_distance = symbols('tx{}'.format(self.label.lower()),
                                    real=True)
        self.foot_depth = symbols('fy{}'.format(self.label.lower()),
                                  real=True)

        self.constants.remove(self.length_symbol)
        del self.length_symbol

        self.constants += [self.heel_distance, self.toe_distance,
                           self.foot_depth]

    def _locate_joint(self):
        """The foot has no joint."""
        pass

    def _locate_foot_points(self):

        self.heel = self.origin_joint.locatenew(
            '{} heel'.format(self.description), self.heel_distance *
            self.reference_frame.x + self.foot_depth *
            self.reference_frame.y)

        self.toe = self.origin_joint.locatenew(
            '{} toe'.format(self.description), self.toe_distance *
            self.reference_frame.x + self.foot_depth *
            self.reference_frame.y)

    def _set_linear_velocities(self):
        """There is no joint so pass this."""
        pass

    def _set_foot_linear_velocities(self):
        """Sets the linear velocities of the mass center and new joint."""
        self.mass_center.v2pt_theory(self.origin_joint, self.inertial_frame,
                                     self.reference_frame)
        self.heel.v2pt_theory(self.origin_joint, self.inertial_frame,
                              self.reference_frame)
        self.toe.v2pt_theory(self.origin_joint, self.inertial_frame,
                             self.reference_frame)

    def visualization_frames(self):
        """Returns a list of visualization frames needed to visualize the
        foot."""

        viz_frames = []

        heel_to_toe_length = self.toe.pos_from(self.heel).magnitude()
        bottom_cylinder = Cylinder(color='red',
                                   name=self.label + 'bottom',
                                   length=heel_to_toe_length,
                                   radius=self.viz_cylinder_radius)
        bottom_center_point = self.heel.locatenew('BottomCenter',
                                                  heel_to_toe_length / 2 *
                                                  self.reference_frame.x)
        # Creates a reference frame with the Y axis pointing from the heel
        # to the toe.
        bottom_rf = self.reference_frame.orientnew('Bottom', 'Axis',
                                                   (-pi / 2,
                                                    self.reference_frame.z))
        viz_frames.append(VisualizationFrame('BottomVizFrame',
                                             bottom_rf,
                                             bottom_center_point,
                                             bottom_cylinder))

        # top of foot
        ankle_to_toe_vector = self.toe.pos_from(self.origin_joint)
        top_cylinder = Cylinder(color='red',
                                name=self.label + 'top',
                                length=ankle_to_toe_vector.magnitude(),
                                radius=self.viz_cylinder_radius)
        angle = -acos(ankle_to_toe_vector.normalize().dot(bottom_rf.y))
        top_foot_rf = bottom_rf.orientnew('Top', 'Axis', (angle,
                                                          bottom_rf.z))
        top_foot_center_point = \
            self.origin_joint.locatenew('TopCenter',
                                        ankle_to_toe_vector.magnitude() / 2
                                        * top_foot_rf.y)
        viz_frames.append(VisualizationFrame('TopVizFrame', top_foot_rf,
                                             top_foot_center_point,
                                             top_cylinder))

        # back of foot
        heel_to_ankle_vector = self.origin_joint.pos_from(self.heel)
        back_cylinder = Cylinder(color='red',
                                 name=self.label + 'back',
                                 length=heel_to_ankle_vector.magnitude(),
                                 radius=self.viz_cylinder_radius)
        angle = acos(heel_to_ankle_vector.normalize().dot(bottom_rf.y))
        back_foot_rf = bottom_rf.orientnew('Back', 'Axis', (angle,
                                                            bottom_rf.z))
        back_foot_center_point = \
            self.heel.locatenew('BackCenter',
                                heel_to_ankle_vector.magnitude() / 2 *
                                back_foot_rf.y)
        viz_frames.append(VisualizationFrame('BackVizFrame', back_foot_rf,
                                             back_foot_center_point,
                                             back_cylinder))

        # spheres for the ankle, toe, and heel
        viz_frames.append(VisualizationFrame('AnkleVizFrame',
                                             self.reference_frame,
                                             self.origin_joint,
                                             Sphere(color='blue',
                                                    name=self.label + '_ankle',
                                                    radius=self.viz_sphere_radius)))
        viz_frames.append(VisualizationFrame('ToeVizFrame',
                                             self.reference_frame, self.toe,
                                             Sphere(color='blue',
                                                    name=self.label + '_toe',
                                                    radius=self.viz_sphere_radius)))
        viz_frames.append(VisualizationFrame('HeelVizFrame',
                                             self.reference_frame, self.heel,
                                             Sphere(color='blue',
                                                    name=self.label + '_heel',
                                                    radius=self.viz_sphere_radius)))

        return viz_frames


def contact_force(point, ground, origin):
    """Returns a contact force vector acting on the given point made of
    friction along the contact surface and elastic force in the vertical
    direction.

    Parameters
    ==========
    point : sympy.physics.mechanics.Point
        The point which the contact force should be computed for.
    ground : sympy.physics.mechanics.ReferenceFrame
        A reference frame which represents the inerital ground in 2D space.
        The x axis defines the ground line and positive y is up.
    origin : sympy.physics.mechanics.Point
        An origin point located on the ground line.

    Returns
    =======
    force : sympy.physics.mechanics.Vector
        The contact force between the point and the ground.

    """
    # This is the "height" of the point above the ground, where a negative
    # value means that the point is below the ground.
    y_location = point.pos_from(origin).dot(ground.y)

    # The penetration into the ground is mathematically defined as:
    #
    #               { 0 if y_location > 0
    # deformation = {
    #               { abs(y_location) if y_location < 0
    #

    penetration = (Abs(y_location) - y_location) / 2

    velocity = point.vel(ground)

    # The addition of "- y_location" here adds a small linear term to the
    # cubic stiffness and creates a light attractive force torwards the
    # ground. This is in place to ensure that gradients can be computed for
    # the optimization used in Ackermann and van den Bogert 2010.
    contact_stiffness, contact_damping = symbols('kc, cc', **sym_kwargs)
    contact_friction_coefficient, friction_scaling_factor = \
        symbols('mu, vs', **sym_kwargs)

    vertical_force = (contact_stiffness * penetration ** 3 - y_location) * \
        (1 - contact_damping * velocity.dot(ground.y))

    friction = -contact_friction_coefficient * vertical_force * \
        ((2 / (1 + exp(-velocity.dot(ground.x) /
                       friction_scaling_factor))) - 1)

    return friction * ground.x + vertical_force * ground.y

def derive_equations_of_motion():

    print('Forming positions, velocities, accelerations and forces.')
    segment_descriptions = {'A': (TrunkSegment, 'Trunk', 'Lumbar'),
                            'B': (BodySegment,'Pelvis','Hip'),
                            'C': (BodySegment, 'Right Thigh', 'Right Knee'),
                            'D': (BodySegment, 'Right Shank', 'Right Ankle'),
                            'E': (FootSegment, 'Right Foot', 'Right Heel'),
                            'F': (BodySegment, 'Left Thigh', 'Left Knee'),
                            'G': (BodySegment, 'Left Shank', 'Left Ankle'),
                            'H': (FootSegment, 'Left Foot', 'Left Heel')}
    
    ground = me.ReferenceFrame('N')
    origin = me.Point('O')
    origin.set_vel(ground, 0)

    segments = []
    constants = []
    coordinates = []
    speeds = []
    specified = []
    kinematic_equations = []
    external_forces_torques = []
    bodies = []
    visualization_frames = []

    for label in sorted(segment_descriptions.keys()):

        segment_class, desc, joint_desc = segment_descriptions[label]

        if label == 'A':  # trunk
            parent_reference_frame = ground
            origin_joint = origin
        elif label == 'F':  # left thigh
            # For the left thigh, set the trunk and hip as the
            # reference_frame and origin joint.
            parent_reference_frame = segments[1].reference_frame
            origin_joint = segments[1].joint
        else:  # thighs, shanks
            parent_reference_frame = segments[-1].reference_frame
            origin_joint = segments[-1].joint

        segment = segment_class(label, desc, parent_reference_frame,
                                origin_joint, joint_desc, ground)
        segments.append(segment)

        # constants, coordinates, speeds, kinematic differential equations
        if label == 'A':  # trunk
            coordinates += segment.qa
            speeds += segment.ua
            constants += segment.constants
        else:
            # skip g for all segments but the trunk
            constants += segment.constants[1:]

        coordinates.append(segment.generalized_coordinate_symbol)
        speeds.append(segment.generalized_speed_symbol)

        kinematic_equations += segment.kinematic_equations

        # gravity
        external_forces_torques.append((segment.mass_center,
                                        segment.gravity))

        # joint torques
        external_forces_torques.append((segment.reference_frame,
                                        segment.torque))
        external_forces_torques.append((segment.parent_reference_frame,
                                        -segment.torque))
        specified.append(segment.joint_torque_symbol)

        # contact force
        if label == 'E' or label == 'H':  # foot
            external_forces_torques.append((segment.heel,
                                            contact_force(segment.heel,
                                                          ground, origin)))
            external_forces_torques.append((segment.toe,
                                            contact_force(segment.toe,
                                                          ground, origin)))
        else:
            external_forces_torques.append((segment.joint,
                                            contact_force(segment.joint,
                                                          ground, origin)))

        # bodies
        bodies.append(segment.rigid_body)

        visualization_frames += segment.visualization_frames()

        # add contact force for trunk mass center.
    external_forces_torques.append((segments[0].mass_center,
                                    contact_force(segments[0].mass_center,
                                                  ground, origin)))
    # add hand of god
    # TODO : move this into segment.py
    trunk_force_x, trunk_force_y = time_varying('Fax, Fay')
    specified = [trunk_force_x, trunk_force_y] + specified
    external_forces_torques.append((segments[0].mass_center, trunk_force_x *
                                    ground.x + trunk_force_y * ground.y))

    # add contact model constants
    # TODO : these should be grabbed from the segments, not recreated.
    constants += list(sy.symbols('kc, cc, mu, vs', real=True, positive=True))

    # equations of motion
    print("Initializing Kane's Method.")
    kane = me.KanesMethod(ground, coordinates, speeds, kinematic_equations)
    print("Forming Kane's Equations.")
    kane.kanes_equations(bodies, loads=external_forces_torques)
    mass_matrix = kane.mass_matrix_full
    forcing_vector = kane.forcing_full

    return (mass_matrix, forcing_vector, kane, constants, coordinates, speeds,
            specified, visualization_frames, ground, origin, segments)




