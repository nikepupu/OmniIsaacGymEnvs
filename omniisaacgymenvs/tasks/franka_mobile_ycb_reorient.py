# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
import carb
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
import omni
# from omni.isaac.cloner import Cloner
# from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core import World
# from omni.debugdraw import get_debug_draw_interface
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
# from omniisaacgymenvs.robots.articulations.franka import Franka
# from omniisaacgymenvs.robots.articulations.franka_mobile import KinovaMobile
from omniisaacgymenvs.robots.articulations.franka_mobile import FrankaMobile
from omniisaacgymenvs.robots.articulations.views.franka_mobile_view import FrankaMobileView
from omniisaacgymenvs.robots.articulations.views.cabinet_view2 import CabinetView
# from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from pxr import Usd, UsdGeom
from pxr import Usd, UsdPhysics, UsdShade, UsdGeom, PhysxSchema
from typing import Optional, Sequence, Tuple, Union
from omni.isaac.core.utils.prims import get_all_matching_child_prims, get_prim_children, get_prim_at_path
from numpy.linalg import inv
from omni.isaac.core.utils.torch.rotations import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quats_to_rot_matrices,
    quat_diff_rad
)
# from pytorch3d.transforms import quaternion_to_matrix
from omni.physx.scripts import deformableUtils, physicsUtils
from omni.isaac.core.prims import RigidPrim,  RigidContactView, RigidPrimView
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.physx.scripts.physicsUtils import *
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Sdf, Gf, Tf, Vt, UsdPhysics, PhysxSchema
from omni.physx import get_physx_interface, get_physx_simulation_interface
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    mat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return mat.reshape(quaternions.shape[:-1] + (3, 3))

def quat_axis(q, axis_idx):
    """Extract a specific axis from a quaternion."""
    rotm = quaternion_to_matrix(q)
    axis = rotm[:, axis_idx]

    return axis

def transform_3d_bounding_box(corners, rotation_matrix, translation):
   

    transformed_corners = torch.stack([rotation_matrix @ corner + translation for corner in corners])
    

    return transformed_corners

def transform_3d_bounding_boxes(corners, rotation_matrices, translations):
    # Initialize a list to hold the transformed corners for each bounding box
    batch_transformed_corners = []

    # Iterate over each rotation matrix and translation vector
    for rotation_matrix, translation in zip(rotation_matrices, translations):
        # Transform corners for each bounding box
        transformed_corners = torch.stack([rotation_matrix @ corner + translation for corner in corners])
        batch_transformed_corners.append(transformed_corners)

    # Stack the results into a single tensor
    batch_transformed_corners = torch.stack(batch_transformed_corners)

    return batch_transformed_corners

def get_corners_from_min_max(min_point, max_point):
    # Generate all 8 corners from min and max points
    x_min, y_min, z_min = min_point
    x_max, y_max, z_max = max_point
    return [
        np.array([x_min, y_min, z_min]),
        np.array([x_min, y_min, z_max]),
        np.array([x_min, y_max, z_min]),
        np.array([x_min, y_max, z_max]),
        np.array([x_max, y_min, z_min]),
        np.array([x_max, y_min, z_max]),
        np.array([x_max, y_max, z_min]),
        np.array([x_max, y_max, z_max])
    ]

@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor = None, q12: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Combine transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{02} = T_{01} \times T_{12}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01 (torch.Tensor): Position of frame 1 w.r.t. frame 0.
        q01 (torch.Tensor): Quaternion orientation of frame 1 w.r.t. frame 0.
        t12 (torch.Tensor): Position of frame 2 w.r.t. frame 1.
        q12 (torch.Tensor): Quaternion orientation of frame 2 w.r.t. frame 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the position and orientation of
            frame 2 w.r.t. frame 0.
    """
    # compute orientation
    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01
    # compute translation
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01

    return t02, q02

def compute_angle_between(x1, x2):
    """Compute angle (radian) between two vectors."""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)
    return np.arccos(dot_prod).item()

def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)

class FrankaMobileYCBTaskReorient(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 37 + 1 #37 + 3 + 7
        self._num_actions = 12   # 10 + 1

        self.translations_orig = None

        self.num_episode = -1
        

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

    def set_up_scene(self, scene) -> None:

        self._usd_context = omni.usd.get_context()
        self.get_ycb()
        self.get_franka()
        
        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaMobileView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        
        self._props = RigidPrimView(
            prim_paths_expr="/World/envs/.*/ycb", name="prop_view", reset_xform_properties=False, track_contact_forces=True
        )
           
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._props)

        self.init_data()
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("franka_view"):
            scene.remove_object("franka_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("cabinet_view"):
            scene.remove_object("cabinet_view", registry_only=True)
        if scene.object_exists("drawers_view"):
            scene.remove_object("drawers_view", registry_only=True)
        if scene.object_exists("prop_view"):
            scene.remove_object("prop_view", registry_only=True)
        # self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._frankas = FrankaMobileView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._cabinets)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)
        
        

        self.init_data()

    def get_franka(self):

      
     

        position = [-0.7, 0.0, 0.0]
     
        franka = FrankaMobile(prim_path=self.default_zero_env_path + "/franka", name="franka",
                               translation=position, orientation=[1,0,0,0])

    def get_ycb (self):
        object_usd = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd'
        add_reference_to_stage(object_usd, self.default_zero_env_path + "/ycb")
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.default_zero_env_path + "/ycb")

        _physicsMaterialPath = prim.GetPath().AppendChild("physicsMaterial")
        material = PhysicsMaterial(
                prim_path=_physicsMaterialPath,
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        # -- enable patch-friction: yields better results!
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
        physx_material_api.CreateImprovePatchFrictionAttr().Set(True)

        physicsUtils.add_physics_material_to_prim(
                    stage,
                    prim,
                    _physicsMaterialPath,
                )
        prims = get_all_matching_child_prims(self.default_zero_env_path + "/ycb")
        for prim in prims:
            physicsUtils.add_physics_material_to_prim(
                    stage,
                    prim,
                    _physicsMaterialPath,
                )
            mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
            if not mass_api:
                mass_api = UsdPhysics.MassAPI.Apply(prim)
            
            mass_api.CreateMassAttr().Set(0.3)
        # 
        ycb_object = RigidPrim( prim_path = self.default_zero_env_path + "/ycb", scale=[0.8, 0.8, 0.8])



        prim_path = self.default_zero_env_path + "/ycb"
        bboxes = omni.usd.get_context().compute_path_world_bounding_box(prim_path)
      
        self.model_bbox_size = torch.tensor( np.array(bboxes[1]) - np.array(bboxes[0]))
        
        self.bboxes = get_corners_from_min_max(bboxes[0], bboxes[1])
        self.bboxes = torch.tensor(self.bboxes).float()

        orientation = [ 0.7071068, 0.7071068, 0, 0,  ]
        self.target_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        ycb_object = RigidPrim( prim_path = self.default_zero_env_path + "/ycb", orientation = orientation, scale=[0.8, 0.8, 0.8])
        bboxes_adjust = omni.usd.get_context().compute_path_world_bounding_box(prim_path)
        min_box = np.array(bboxes_adjust[0])
        zmin = min_box[2]

   
    
        position, orientation = ycb_object.get_world_pose()
        position[2] = -zmin 

        self.init_pos = position.clone()
        
        ycb_object.set_world_pose(position, orientation)

        matrix = quaternion_to_matrix(orientation.cpu().float())

        corner = transform_3d_bounding_box(self.bboxes, matrix, position.cpu()).numpy()


        self.vectors = self.batch_extract_direction_vectors(np.array([corner]))[0]
        self.handle_out = torch.tensor(self.vectors[0], device=self._device).to(torch.float32)
        self.handle_long = torch.tensor(self.vectors[1], device=self._device).to(torch.float32)
        self.handle_short = torch.tensor(self.vectors[2], device=self._device).to(torch.float32)
       
        stage = get_current_stage()
        prim = stage.GetPrimAtPath( "/World/envs/env_0/ycb/_03_cracker_box")
      
        UsdPhysics.CollisionAPI.Apply(prim)
        meshCollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
        meshCollisionAPI.CreateApproximationAttr().Set("convexHull")

        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(prim)


        

    def init_data(self) -> None:
        # enable ccd: /physicsScene
        physicsScenePath = '/physicsScene'
        stage = get_current_stage()
        scene = UsdPhysics.Scene.Get(stage, physicsScenePath)
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physxSceneAPI.CreateEnableCCDAttr().Set(True)
        self.franka_default_dof_pos = torch.tensor(
            [0, 0, 0, 1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device)

    def get_ee_pose(self):
        hand_position_w, hand_quat_w = self._frankas._hands.get_world_poses(clone=True)
       
        return hand_position_w, hand_quat_w
    
    def check_grasp(self, max_angle):
        def extract_envid(identifier):
            # extract the number follow env:
            return int(identifier.split('env_')[1].split('/')[0])


        contact_headers, contact_data = get_physx_simulation_interface().get_contact_report()
        
        grasped = torch.zeros((self._num_envs), device=self._device)
        rimpulses = torch.zeros((self._num_envs, 3), device=self._device)
        limpulses = torch.zeros((self._num_envs, 3), device=self._device)
        rangles = torch.zeros((self._num_envs), device=self._device)
        langles = torch.zeros((self._num_envs), device=self._device)

        for contact_header in contact_headers:
            collider0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1))

            env_id1 = -1
            env_id2 = -1
            if 'franka' in collider0 or 'ycb' in collider0:
                env_id1 = extract_envid(collider0) 
            if 'franka' in collider1 or 'ycb' in collider1:
                env_id2 = extract_envid(collider1)

            l = False
            r = False
            if env_id1 == env_id2 and env_id1 != -1:
                env_id = env_id1
            else:
                continue

            if ( (f'/World/envs/env_{env_id}/franka/panda_rightfinger' in collider0 )  and ( f'/World/envs/env_{env_id}/ycb' in collider1 ) ) or \
                ( (f'/World/envs/env_{env_id}/franka/panda_rightfinger' in collider1 ) and (f'/World/envs/env_{env_id}/ycb' in collider0) ):
                r = True

            if ( (f'/World/envs/env_{env_id}/franka/panda_leftfinger' in collider0 )  and ( f'/World/envs/env_{env_id}/ycb' in collider1 ) ) or \
                ( (f'/World/envs/env_{env_id}/franka/panda_leftfinger' in collider1 ) and (f'/World/envs/env_{env_id}/ycb' in collider0) ):
                l = True

            if l:
                contact_data_offset = contact_header.contact_data_offset
                num_contact_data = contact_header.num_contact_data
                limpulse = None
                for index in range(contact_data_offset, contact_data_offset + num_contact_data, 1):
                    impulse = contact_data[index].impulse
                    limpulse = torch.tensor(np.array(impulse)).cpu()
                    break
                ldirection = quaternion_to_matrix(self._frankas._lfingers.get_world_poses()[1])[env_id][:3,1].cpu().numpy()

                if limpulse is not None:
                    l_angle = np.rad2deg(compute_angle_between(ldirection, limpulse.numpy()))
                    langles[env_id] = l_angle
                    limpulses[env_id] = limpulse
                
            
            if r:
                contact_data_offset = contact_header.contact_data_offset
                num_contact_data = contact_header.num_contact_data
                rimpulse = None
                for index in range(contact_data_offset, contact_data_offset + num_contact_data, 1):
                    impulse = contact_data[index].impulse
                    rimpulse = torch.tensor(np.array(impulse)).cpu()
                    break

                
                rdirection = quaternion_to_matrix(self._frankas._rfingers.get_world_poses()[1])[env_id][:3,1].cpu().numpy()
                
                if rimpulse is not None:
                    r_angle =  np.rad2deg(compute_angle_between(rdirection, rimpulse.numpy()))
                    rangles[env_id] = r_angle
                    rimpulses[env_id] = rimpulse
     
        grasped =   (rangles < max_angle) & (langles < max_angle) & (rimpulses.norm(dim=1) > 0.01) & (limpulses.norm(dim=1) > 0.01)
        if torch.nonzero(grasped).shape[0] > 3:
            world = World()
            index = torch.nonzero(grasped)
            print('index: ', index)
            print('grasped: ', grasped[index])
            print('rangles: ', rangles[index])
            print('langles: ', langles[index])
            print('rimpulses: ', rimpulses[index])
            print('limpulses: ', limpulses[index])
            while True:
                world.render()
        # print(rimpulses.norm(dim=1))
        # print(rimpulses.norm(dim=1).shape)
        # exit()
        # print('grasped: ', grasped)
        # print('rangles: ', rangles)
        # print('langles: ', langles)
        # print('rimpulses: ', rimpulses)
        # print('limpulses: ', limpulses)

        return grasped


               
                    
                    # print("Contact: ", index - contact_data_offset)
                    # print("Contact position: " + str(contact_data[index].position))
                    # print("Contact normal: " + str(contact_data[index].normal))
                    # print("Contact impulse: " + str(contact_data[index].impulse))
                    # print("Contact separation: " + str(contact_data[index].separation))
                    # print("Contact faceIndex0: " + str(contact_data[index].face_index0))
                    # print("Contact faceIndex1: " + str(contact_data[index].face_index1))
                    # print("Contact material0: " + str(PhysicsSchemaTools.intToSdfPath(contact_data[index].material0)))
                    # print("Contact material1: " + str(PhysicsSchemaTools.intToSdfPath(contact_data[index].material1)))

        # if l :
        #     print('left finger in contact')
        
        # if r :
        #     print('right finger in contact')
            
        
            
            

        # for contact_header in contact_headers:
        #     print("Got contact header type: " + str(contact_header.type))
        #     print("Actor0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0)))
        #     print("Actor1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1)))
        #     print("Collider0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0)))
        #     print("Collider1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1)))
        #     print("StageId: " + str(contact_header.stage_id))
        #     print("Number of contacts: " + str(contact_header.num_contact_data))
            
        #     contact_data_offset = contact_header.contact_data_offset
        #     num_contact_data = contact_header.num_contact_data
            
        #     for index in range(contact_data_offset, contact_data_offset + num_contact_data, 1):
        #         print("Contact: ", index - contact_data_offset)
        #         print("Contact position: " + str(contact_data[index].position))
        #         print("Contact normal: " + str(contact_data[index].normal))
        #         print("Contact impulse: " + str(contact_data[index].impulse))
        #         print("Contact separation: " + str(contact_data[index].separation))
        #         print("Contact faceIndex0: " + str(contact_data[index].face_index0))
        #         print("Contact faceIndex1: " + str(contact_data[index].face_index1))
        #         print("Contact material0: " + str(PhysicsSchemaTools.intToSdfPath(contact_data[index].material0)))
        #         print("Contact material1: " + str(PhysicsSchemaTools.intToSdfPath(contact_data[index].material1)))



    def get_observations(self) -> dict:
        

        # drawer_pos, drawer_rot = self._cabinets._drawers.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
      
        # self.centers = (self.centers_orig +  self.forwardDir * self.cabinet_dof_pos[:, 3].unsqueeze(-1)).to(torch.float32).to(self._device)
       
        hand_pos, hand_rot = self.get_ee_pose()
        hand_pos = hand_pos - self._env_pos

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        
        centers = self._props.get_world_poses()[0] - self._env_pos

        rotation = self._props.get_world_poses()[1]

      
        tool_pos_diff = hand_pos  - centers

        # goal_position = self.init_pos.repeat( self._num_envs, 1) + torch.tensor([0.0, 0.0, 0.5], device=self._device).to(torch.float32)
        # print('hand_pos: ', hand_pos)
        # print('point_center: ', centers)
        # print('tool_pos_diff: ', tool_pos_diff)
        # exit()
        # grasped = self.check_grasp(30).unsqueeze(-1)
        # print('grasped: ', torch.nonzero(grasped))
        target_orientation = self.target_orientation.repeat(self._num_envs, 1).to(self._device)
        position, orientation =  self._props.get_world_poses()
       

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled, # 12
                franka_dof_vel * self.dof_vel_scale, # 12
                tool_pos_diff, # 3
                hand_pos, # 3
                hand_rot, # 4
                centers, # 3
                ((position[:,2]- self.init_pos[2]) > 0.2).to(torch.float32).unsqueeze(-1), # 1
                # rotation, # 4
                # target_orientation, # 4
                # goal_position# grasped
            ),
            dim=-1,
        )
        # print('obs: ',  self.obs_buf[0,:])
        # exit()
        observations = {self._frankas.name: {"obs_buf": self.obs_buf.to(torch.float32)}}
        
        # observations = {self._frankas.name: {"obs_buf": torch.zeros((self._num_envs, self._num_observations))}}
        # print('obs: ', observations)
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # self.actions = torch.zeros((self._num_envs, self._num_actions+5), device=self._device)
        # import pdb; pdb.set_trace()
        # self.actions[:, :10] = (actions.clone()[:,:10]).to(self._device)

        # self.actions = (actions.clone()).to(self._device)
        # targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        # map -1 to 1 to 0 to 1
        self.actions = actions.clone().to(self._device)
        # mode_prob = (self.actions[:, 10]  + 1.0 )/2
        gripper_mode_prob =  (self.actions[:, 11]  + 1.0 )/2
        # sample 0 or 1 based on mode_prob
        # mode = torch.bernoulli(mode_prob).to(torch.int32)

        # mode = (mode_prob > 0.5).long()
        # base_indices = torch.nonzero(mode).long()
        # arm_indices = torch.nonzero(1 - mode).long()
        
        self.actions[:, 0:-2] = (self.actions[:, 0:-2] + 1.0) / 2.0
        
        gripper_mode = (gripper_mode_prob > 0.5).long()
        gripper_indices = torch.nonzero(gripper_mode).long()
        gripper_inverse_indices = torch.nonzero(1 - gripper_mode).long()

        self.actions[gripper_indices,-1] = 1.0
        self.actions[gripper_indices,-2] = 1.0
        self.actions[gripper_inverse_indices,-1] = 0.0
        self.actions[gripper_inverse_indices,-2] = 0.0
      
        

        current_joint_positons = self._frankas.get_joint_positions(clone=False)
        base_positions = current_joint_positons[:, :3]
        arm_positions = current_joint_positons[:, 3:]

        # print(base_positions.shape)
        # print(arm_positions.shape)
        # print(self.franka_dof_targets.shape)
        # exit()


        targets = self.actions[:, 0:] *(self.franka_dof_upper_limits - self.franka_dof_lower_limits) + self.franka_dof_lower_limits

        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        

        # if len(base_indices) > 0:
        self.franka_dof_targets[:, :3 ] =  base_positions[:]
        # if len(arm_indices) > 0:
        #     self.franka_dof_targets[arm_indices, 3:] =  arm_positions[arm_indices]
        

        
        # self.franka_dof_targets[:,:3] = 0.0

        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # print(self.franka_dof_targets)
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def batch_extract_direction_vectors(self, batch_corners):
        
        direction_vectors = []

        for corners in batch_corners:
            print(corners)
            corners = np.array(corners)

            # Identify the extreme corner
            max_corner = np.max(corners, axis=0)
            min_corner = np.min(corners, axis=0)
          

            x_diff = max_corner[0] - min_corner[0]
            y_diff = max_corner[1] - min_corner[1]
            z_diff = max_corner[2] - min_corner[2]

            x_vector = np.array([x_diff, 0, 0])

            y_vector = np.array([0, y_diff, 0])

            z_vector = np.array([0, 0, z_diff])

            

            direction_vectors.append([x_vector, y_vector, z_vector])

        return torch.tensor(direction_vectors)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )

        pos[:, :3] = 0.0
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        self._frankas.set_joint_position_targets(self.franka_dof_targets)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        self._props.post_reset()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.num_episode += 1

    def post_reset(self):

        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        # if self.num_props > 0:
        #     self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
        #     self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
        #         self._num_envs, self.num_props
        #     )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def compute_upper_face_center(self, corners):
        # Convert to numpy array for easier handling
        corners = np.array(corners)

        # Sort corners by their z-values
        sorted_corners = corners[corners[:, 2].argsort()]

        # Select the top four corners
        upper_face_corners = sorted_corners[-4:]

        # Compute the average of the x and y coordinates
        center_x = np.mean(upper_face_corners[:, 0])
        center_y = np.mean(upper_face_corners[:, 1])
        center_z = np.mean(upper_face_corners[:, 2])

        return (center_x, center_y, center_z)
    
  


    def calculate_metrics(self) -> None:
        self.rew_buf[:] = 0.0
        hand_pos, hand_rot = self.get_ee_pose()

        position, orientation =  self._props.get_world_poses()

        matrix = quaternion_to_matrix(orientation.cpu().float())

        corners = transform_3d_bounding_boxes(self.bboxes, matrix, position.cpu()).numpy()
        
        upper_face_centers = torch.tensor([self.compute_upper_face_center(corners[i]) for i in range(corners.shape[0])]).to(self._device)
        # add 3 centimeter  to the z axis

        upper_face_centers = upper_face_centers + torch.tensor([0, 0, 0.02], device=self._device)
        upper_face_centers_exact = upper_face_centers + torch.tensor([0, 0, -0.045], device=self._device)

        tcp_to_obj_dist = torch.norm(upper_face_centers - hand_pos, dim=-1)
        delta = upper_face_centers_exact - hand_pos
        tcp_to_obj_dist_obj = torch.norm(upper_face_centers_exact - hand_pos, dim=-1)

        # reaching_reward = 1 - torch.tanh( 3.0 * tcp_to_obj_dist) 

        


        

        

        handle_out_length = torch.norm(self.handle_out, dim = -1).to(torch.float32).to(self._device)
        handle_long_length = torch.norm(self.handle_long, dim = -1).to(torch.float32).to(self._device)
        handle_short_length = torch.norm(self.handle_short, dim = -1).to(torch.float32).to(self._device)

        handle_out = self.handle_out / handle_out_length.unsqueeze(-1).to(torch.float32).to(self._device)
        handle_long = self.handle_long / handle_long_length.unsqueeze(-1).to(torch.float32).to(self._device)
        handle_short = self.handle_short / handle_short_length.unsqueeze(-1).to(torch.float32).to(self._device)


        self.franka_lfinger_pos = self._frankas._lfingers.get_world_poses(clone=False)[0] 
        self.franka_rfinger_pos = self._frankas._rfingers.get_world_poses(clone=False)[0] 
        

        
      

        # print(self.franka_lfinger_pos)
        # print(self.franka_rfinger_pos)
     
       
        short_ltip = ((self.franka_lfinger_pos - upper_face_centers_exact) * handle_long).sum(dim=-1) 
        short_rtip = ((self.franka_rfinger_pos - upper_face_centers_exact) * handle_long).sum(dim=-1)

        

        is_reached_long = (short_ltip * short_rtip) < 0

        is_reached_short = (delta * handle_short).sum(dim=-1).abs() < (handle_short_length / 2.0)
        is_reached_out = (delta * handle_out).sum(dim=-1).abs() < (handle_out_length / 2.0 )

        is_reached =  is_reached_out & is_reached_long & is_reached_short

        reaching_reward_obj = 1 - torch.tanh( 3.0 * tcp_to_obj_dist_obj) + 0.1 * (is_reached_short + is_reached_long + is_reached_out) 
        
        gripper_length = torch.norm(self.franka_lfinger_pos - self.franka_rfinger_pos, dim=-1)
       


        hand_grip_dir = quat_axis(hand_rot, 0).to(torch.float32).to(self._device)
       
        
        hand_sep_dir = quat_axis(hand_rot, 2).to(torch.float32).to(self._device)
      

        hand_down_dir = quat_axis(hand_rot, 1).to(torch.float32).to(self._device)
        
        dot1 = torch.max((hand_grip_dir * handle_out).sum(dim=-1), (-hand_grip_dir * handle_out).sum(dim=-1))
        dot2 = torch.max((hand_sep_dir * handle_short).sum(dim=-1), (-hand_sep_dir * handle_short).sum(dim=-1))
        dot3 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))

        rot_reward = dot1 + dot2 + dot3 - 3     

        # is_reached_pregrasped = (tcp_to_obj_dist < 0.03).to(self._device) & (rot_reward > -0.2).to(self._device)

        # is_reached_grasp = (tcp_to_obj_dist_obj < 0.02).to(self._device) & (rot_reward > -0.2).to(self._device)

        grasp_success = is_reached &  (tcp_to_obj_dist_obj < 0.025) & (gripper_length < handle_long_length + 0.01) & (rot_reward > -0.2)

        close_reward =   ( gripper_length -0.1) * (tcp_to_obj_dist_obj >= 0.01) * 0.1 + (0.1 - gripper_length ) * (tcp_to_obj_dist_obj < 0.02)

        num_envs = orientation.shape[0]


        
        target_orientation = self.target_orientation.repeat(num_envs, 1)
       
        object_rotation_diff = torch.abs(quat_diff_rad(orientation.to(self._device), target_orientation.to(self._device)))

        rot_diff_reward = 1 - torch.tanh( 5.0 * object_rotation_diff)

        # self.rew_buf +=   reaching_reward_obj  + 10 * close_reward + grasp_success *  10 *  (0.1 +  height_reward ) 
        # lift_reward =  grasp_success *  10 *  (0.1 +  100*( torch.clamp(position[:,2]- self.init_pos[2], max=0.2)) )
        # if < 0, then give a large penalty
        # lift_reward = 0.0 #torch.where(position[:,2]- self.init_pos[2] < -0.01, -1.0  , lift_reward)
        
        # add rotation diff reward only if the object is higher than 0.2 m
        target_rotation_reward =   rot_diff_reward 

        # standard_reward =  lift_reward + 2 * reaching_reward_obj + 5 * close_reward  + rot_reward * 0.5 
        

        # add rot_reward only if the object height is smaller than 0.2 m
       
        standard_reward = 2*reaching_reward_obj  + rot_reward * 0.5 + 5 * close_reward + grasp_success *  10 *  (0.1 +  50*(position[:,2]- self.init_pos[2])  ) 
        
        self.rew_buf =  ((position[:,2]- self.init_pos[2]) > 0.2 ) * (target_rotation_reward+ 1.0)  * 2000  + ((position[:,2]- self.init_pos[2]) <= 0.2 ) * standard_reward

        # if self.progress_buf[0] > 180:
        #     self.rew_buf =  ((position[:,2]- self.init_pos[2]) > 0.2 ) * target_rotation_reward  * 2000  
        #         # + ((position[:,2]- self.init_pos[2]) <= 0.2 ) * standard_reward
        # else:
        #     self.rew_buf =   5*reaching_reward_obj  + rot_reward * 0.5 + 5 * close_reward + grasp_success *  10 *  (0.1 +  10*(position[:,2]- self.init_pos[2]) * 10 ) 

        # writer.add_scalar("Reward/reaching_reward_obj", torch.mean(reaching_reward_obj), self.progress_buf[0] + self._max_episode_length * self.num_episode )
        # writer.add_scalar("Reward/rot_reward", torch.mean(((position[:,2]- self.init_pos[2])  < 0.1) * rot_reward * 0.5) , self.progress_buf[0] + self._max_episode_length * self.num_episode )
        # writer.add_scalar("Reward/close_reward", torch.mean(close_reward * 5), self.progress_buf[0] + self._max_episode_length * self.num_episode )
        # # writer.add_scalar("Reward/lift_reward", torch.mean(lift_reward)  , self.progress_buf[0] + self._max_episode_length * self.num_episode )
        # writer.add_scalar("Reward/target_rot_reward", torch.mean(target_rotation_reward * 100), self.progress_buf[0] + self._max_episode_length * self.num_episode)
      
        self.rew_buf[:] = self.rew_buf[:].to(torch.float32)
        

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        # self.reset_buf = torch.where(self.cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )

    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):

        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos