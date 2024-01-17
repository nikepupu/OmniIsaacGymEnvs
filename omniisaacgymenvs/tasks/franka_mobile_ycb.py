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
    quats_to_rot_matrices
)
# from pytorch3d.transforms import quaternion_to_matrix
from omni.physx.scripts import deformableUtils, physicsUtils
from omni.isaac.core.prims import RigidPrim,  RigidContactView, RigidPrimView
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.physx.scripts.physicsUtils import *
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Sdf, Gf, Tf, Vt, UsdPhysics, PhysxSchema
from omni.physx import get_physx_interface, get_physx_simulation_interface

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

class FrankaMobileYCBTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 37 #37 + 3 + 7
        self._num_actions = 12   # 10 + 1

        self.translations_orig = None
        

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

      
     

        position = [-1.5, 0.0, 0.0]
     
        franka = FrankaMobile(prim_path=self.default_zero_env_path + "/franka", name="franka",
                               translation=position, orientation=[1,0,0,0])

    def get_ycb (self):
        object_usd = 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd'
        add_reference_to_stage(object_usd, self.default_zero_env_path + "/ycb")
        orientation = [ 0.7071068, 0.7071068, 0, 0,  ]
        ycb_object = RigidPrim( prim_path = self.default_zero_env_path + "/ycb", orientation = orientation)

        prim_path = self.default_zero_env_path + "/ycb"
        bboxes = omni.usd.get_context().compute_path_world_bounding_box(prim_path)
        min_box = np.array(bboxes[0])
        zmin = min_box[2]


        position, orientation = ycb_object.get_world_pose()
        position[2] += -zmin 
       
        ycb_object.set_world_pose(position, orientation)

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
    
    def check_grasp(self):
        def extract_envid(identifier):
            # extract the number follow env:
            return int(identifier.split('env_')[1].split('/')[0])


        contact_headers, contact_data = get_physx_simulation_interface().get_contact_report()
        l = False
        r = False
        for contact_header in contact_headers:
            collider0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1))

            env_id1 = -1
            env_id2 = -1
            if 'franka' in collider0 or 'ycb' in collider0:
                env_id1 = extract_envid(collider0) 
            if 'franka' in collider1 and 'ycb' in collider1:
                env_id2 = extract_envid(collider1)

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
            print('contact part: ')
            print(collider0)
            print(collider1)

         
            
        if l :
            print('left finger in contact')
        
        if r :
            print('right finger in contact')
            
        
            
            

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

      
        tool_pos_diff = hand_pos  - centers
        # print('hand_pos: ', hand_pos)
        # print('point_center: ', centers)
        # print('tool_pos_diff: ', tool_pos_diff)
        # exit()
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled, # 12
                franka_dof_vel * self.dof_vel_scale, # 12
                tool_pos_diff, # 3
                hand_pos, # 3
                hand_rot, # 4
                centers, # 3
            ),
            dim=-1,
        )
        # print('obs: ',  self.obs_buf[0,:])
        # exit()
        observations = {self._frankas.name: {"obs_buf": self.obs_buf.to(torch.float32)}}
        self.check_grasp()
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
        # mode_prob = (self.actions[:, 0]  + 1.0 )/2
        # sample 0 or 1 based on mode_prob
        # mode = torch.bernoulli(mode_prob).to(torch.int32)

        # mode = (mode_prob > 0.5).long()
        # base_indices = torch.nonzero(mode).long()
        # arm_indices = torch.nonzero(1 - mode).long()



        # mode = self.actions[:, 0] <= 0
        # base_indices =  torch.nonzero(mode).long()

     
        # mode = self.actions[:, 0] > 0
        # arm_indices =  torch.nonzero(mode).long()

   
        
        self.actions[:, 0:] = (self.actions[:, 0:] + 1.0) / 2.0
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
        #     self.franka_dof_targets[base_indices, :3 ] =  base_positions[base_indices]
        # if len(arm_indices) > 0:
        #     self.franka_dof_targets[arm_indices, 3:] =  arm_positions[arm_indices]
        

        
        # self.franka_dof_targets[:,:3] = 0.0

        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # print(self.franka_dof_targets)
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

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

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = 0.0
        ycb_pose = self._props.get_world_poses()[0] - self._env_pos
        hand_pose = self.get_ee_pose()[0] - self._env_pos

        dist_nrom = torch.norm(ycb_pose - hand_pose, dim=-1)
        # print(dist_nrom)
        

        self.rew_buf += - dist_nrom 

        self.check_grasp()
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