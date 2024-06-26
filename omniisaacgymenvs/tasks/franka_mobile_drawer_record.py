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
import os
import json
from typing import List, Type
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
from omni.debugdraw import get_debug_draw_interface
from omniisaacgymenvs.tasks.base.rl_task_record import RLTaskRecord
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
    matrices_to_euler_angles,
    quat_from_euler_xyz
)
# from pytorch3d.transforms import quaternion_to_matrix
from omni.physx.scripts import deformableUtils, physicsUtils
from omni.isaac.core.utils.stage import add_reference_to_stage
import pxr
from omni.isaac.core.utils.prims import get_all_matching_child_prims, get_prim_children, get_prim_at_path, delete_prim

def load_annotation_file():
    folder = '/home/nikepupu/Desktop/gapartnet_new_subdivition/partnet_all_annotated_new/annotation'
    subfolders = sorted(os.listdir(folder))
    
    # filter out files starts with 4 and has 5 digits
    subfolders = [f for f in subfolders if f.startswith('4') and len(f) == 5]
    annotation_json = 'link_anno_gapartnet.json'
    annotation = {}
    for subfolder in subfolders:
        annotation_path = os.path.join(folder, subfolder, annotation_json)
        with open(annotation_path, 'r') as f:
            annotation[int(subfolder)] = json.load(f)
    return annotation

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

class FrankaMobileDrawerTaskRecord(RLTaskRecord):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 40 #37 + 3 + 7
        self._num_actions = 13 # 10 + 1

        self.annotations = load_annotation_file()

        self.translations_orig = None
        self.scene_index = 0
        self.cabinet_index = 0
        self.drawer_index = 0

        RLTaskRecord.__init__(self, name, env)
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
        import os
        import re
        import json

        def parse_folder(path='/home/nikepupu/Desktop/physcene'):
            infos = os.path.join(path, 'articulated_info')

            info_return = []
            for info in os.listdir(infos):
                numbers = re.findall(r'\d+', info)

                # numbers will be a list of all number sequences found, taking the first one
                number = numbers[0] if numbers else None
                # info_dict[number] = (info, infos)
                with  open(os.path.join(infos, info)) as f:
                    data = json.load(f)
                    nodes = data["nodes"]
                
                nodes_to_load = []
                for node in nodes:
                    if 'articulated' in node:
                        if node['articulated']:
                            GAPartNet_ID = node['GAPartNet_ID']
                            position = node['position']
                            orientation = node['orientation']
                            scale = node['scale']
                            # print(self.annotations[int(GAPartNet_ID)])
                            # exit()
                            node_dict = {
                                'GAPartNet_ID': GAPartNet_ID,
                                'position': position,
                                'orientation': orientation,
                                'scale': scale,
                                # 'link_name': node['link_name']
                            }
                            nodes_to_load.append(node_dict)
                scene_path = os.path.join(path, 'usd', 'physcene_'+ number, 'main.usd')
                info_return.append((scene_path, nodes_to_load))
            return info_return
                    
        # def find_next_drawer(info_return):
        #     while True:
        #         node_to_load = info_return[self.scene_index][1][self.cabinet_index]
        #         GAPartNet_ID = node_to_load["GAPartNet_ID"]
        #         file_to_read =  f'/home/nikepupu/Desktop/gapartnet_new_subdivition/partnet_all_annotated_new/annotation/{GAPartNet_ID}/link_anno_gapartnet.json'
        #         with open(file_to_read) as json_file:
        #             data = json.load(json_file)
                
        #         print(data)
        #         exit()

        info_return = parse_folder()
        scene_path = info_return[self.scene_index][0]
        self.info_return = info_return
        
        self.get_cabinet()
        self.get_scene()

        

        
        # self.get_franka()
        world = World()
        while True:
            world.render()

        
       

        # if self.num_props > 0:
        #     self.get_props()

        super().set_up_scene(scene, filter_collisions=False)



        self._frankas = FrankaMobileView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")
        # world = World()
        # while True:
        #     world.render()

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._cabinets)
        # scene.add(self._cabinets._drawers)

        # if self.num_props > 0:
        #     self._props = RigidPrimView(
        #         prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
        #     )
        #     scene.add(self._props)

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
        self._frankas = FrankaMobileView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        # self._frankas = KinovaMobileView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

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
    
    def get_scene(self):
        import json
        scene_path = self.info_return[self.scene_index][0]
        add_reference_to_stage(scene_path, '/World/envs/env_0/scene')
        # with open('/home/nikepupu/Desktop/physcene_33_rigid/contact_graph_cad.json') as f:
        #     self.scene_info = json.load(f)
        # print(self.scene_info["nodes"])
        # for node in self.scene_info["nodes"]:
            
        #     if  'cad_id' in node and node["cad_id"] == "storagefurniture_gpn_6":
        #         self.cabinet_orientation = node["orientation"]
        #         self.cabinet_position = node["position"]
        #         self.cabinet_scale = node["scale"]
       
        # self.cabinet_scale = torch.tensor([scale, scale, scale]).to(torch.float32)
        # self.cabinet_position = torch.tensor([0.0, 0.0, 0.0]).to(torch.float32)
        # self.cabinet_position[0] += 0.01
        # self.cabinet_position[1] -= 0.01

        # self.cabinet_position = [0, 0, 0]
        # self.cabinet_orientation = torch.tensor([ 0.7071068, 0, 0, 0.7071068]).to(torch.float32)
        # self.cabinet_orientation  =  torch.tensor([1, 0, 0, 0]).to(torch.float32)

        
        # self.cabinet_orientation[1] = 0.0
        # self.cabinet_orientation[2] = 0.0
        # print('cabinet orientation: ', self.cabinet_orientation)
        # exit()
        prims = get_all_matching_child_prims('/World/envs/env_0/scene', depth=1)
        for prim in prims:
            if 'ceiling' in prim.GetPath().pathString.lower():
                prim.SetActive(False)
   

    def get_franka(self):
        link_path =  f"/World/envs/env_0/cabinet/link_8"
        prim = get_prim_at_path(link_path)
        
        matrix = inv(np.array(omni.usd.get_world_transform_matrix(prim)))
        
        forwardDir = matrix[0:3, 2]
        forwardDir = forwardDir/np.linalg.norm(forwardDir)

        

        position = self.cabinet_position + forwardDir * 1.5
        position[2] = 0.0
        orientation = self.cabinet_orientation
        franka = FrankaMobile(prim_path=self.default_zero_env_path + "/franka", name="franka",
                               translation=position, orientation=orientation)
        
        print('franka_position: ', position)
        # exit()

        self.forwardDir = torch.tensor(self.forwardDir).to(self._device).repeat((self._num_envs,1))
        # stage = get_current_stage()
        # prim = stage.GetPrimAtPath(self.default_zero_env_path + "/franka")
        # _physicsMaterialPath = prim.GetPath().AppendChild("physicsMaterial")
        # prim = stage.GetPrimAtPath(self.default_zero_env_path + "/franka")
        # physicsUtils.add_physics_material_to_prim(
        #             stage,
        #             prim,
        #             _physicsMaterialPath,
        #         )
        
        # prim = stage.GetPrimAtPath(self.default_zero_env_path + "/franka/left_inner_finger_pad")
        # physicsUtils.add_physics_material_to_prim(
        #             stage,
        #             prim,
        #             _physicsMaterialPath,
        #         )
        # self._sim_config.apply_articulation_settings(
        #     "franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka")
        # )
    def rotate_points_around_z(self, points, center, angle_degrees):
        """
        Rotate a set of points around the z-axis by a given angle around a center point.

        :param points: np.array, array of points to rotate.
        :param center: np.array, the center point around which to rotate.
        :param angle_degrees: float, rotation angle in degrees.
        :return: np.array, array of rotated points.
        """
        # Convert the angle to radians
        angle_radians = np.radians(angle_degrees)

        # Create the rotation matrix for z-axis rotation
        cos_angle, sin_angle = np.cos(angle_radians), np.sin(angle_radians)
        rotation_matrix = np.array([[cos_angle, -sin_angle, 0],
                                    [sin_angle,  cos_angle, 0],
                                    [0,          0,         1]])

        # Translate points to origin (center point becomes the origin)
        center = center.copy()
        center[2] = 0
        translated_points = points - center

        # Apply the rotation
        rotated_points = np.dot(translated_points, rotation_matrix.T)

        # Translate points back to original center
        rotated_points += center

        return rotated_points
    
    def get_cabinet(self):
        
        while True:
            print(self.scene_index, self.cabinet_index, self.drawer_index, self.info_return[self.scene_index])
            node_to_load = self.info_return[self.scene_index][1][self.cabinet_index]
            
            position = node_to_load['position']
            orientation = node_to_load['orientation']
            # change orientation from xyzw to wxyz
            orientation = torch.tensor([orientation[3], orientation[0], orientation[1], orientation[2]]).to(torch.float32)


            print()

            scale = node_to_load['scale']
            
            scale_max = max(scale)
            scale_min = min(scale)
        
            GAPartNet_ID = int(node_to_load["GAPartNet_ID"])
            if GAPartNet_ID not in self.annotations.keys():
                self.cabinet_index += 1
                if self.cabinet_index >= len(self.info_return[self.scene_index][1]):
                    self.scene_index += 1
                    self.cabinet_index = 0
                    self.drawer_index = 0
               
                continue
            data = self.annotations[int(GAPartNet_ID)][self.drawer_index]
            if (not data['is_gapart']) or (data['category'] != 'slider_drawer'):
                self.drawer_index += 1
                if self.drawer_index >= len(self.annotations[int(GAPartNet_ID)]):
                    self.cabinet_index += 1
                    self.drawer_index = 0
                if self.cabinet_index >= len(self.info_return[self.scene_index][1]):
                    self.scene_index += 1
                    self.cabinet_index = 0
                    self.drawer_index = 0
                continue
            # print(data)
            # exit()
            


        
            
            cabinet = Cabinet(self.default_zero_env_path + "/cabinet", name="cabinet", 
                            usd_path=f"/home/nikepupu/Desktop/Orbit/NewUSD/{GAPartNet_ID}/mobility_relabel_gapartnet.usd", 
                            translation=position, orientation=orientation, scale=[scale_min, scale_min, scale_min])
            
            ############ cehck if ok
            env_path = f"{self.default_base_env_path}/env_{0}"
            bbox_link = None
            
            link_name = data['link_name']

            children = get_all_matching_child_prims(env_path + "/cabinet")
            # print('children: ', children)
            prims: List[Usd.Prim] = [x for x in children if x.IsA(UsdPhysics.Joint)] 
            # print(prims)
            
            for joint in prims:
                body1 = joint.GetRelationship("physics:body1").GetTargets()[0]
                if bbox_link is None and len(joint.GetRelationship("physics:body0").GetTargets()) > 0:
                    body0 = joint.GetRelationship("physics:body0").GetTargets()[0]
                    if (body0.pathString).endswith(link_name):
                        bbox_link = body1

            if not bbox_link:

                # delete
                delete_prim(env_path + "/cabinet")
                self.cabinet_index += 1
                if self.cabinet_index >= len(self.info_return[self.scene_index][1]):
                    self.scene_index += 1
                    self.cabinet_index = 0
                    self.drawer_index = 0
            else:
                break

        
        ##############
        

        # move cabinet to the ground
        prim_path = self.default_zero_env_path + "/cabinet"
        bboxes = omni.usd.get_context().compute_path_world_bounding_box(prim_path)
        min_box = np.array(bboxes[0])
        zmin = min_box[2]
        drawer = XFormPrim(prim_path=prim_path)
        position, orientation = drawer.get_world_pose()
        position[2] += -zmin
        self.cabinet_offset = -zmin 

        # print(self.cabinet_offset)
        # exit()
        drawer.set_world_pose(position, orientation)
        print('cabinet_position: ', position)
        

        # add physics material
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.default_zero_env_path + f"/cabinet/{link_name}/collisions")
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

        prims = get_all_matching_child_prims(self.default_zero_env_path + f"/cabinet/{link_name}/collisions")
        for prim in prims:
            physicsUtils.add_physics_material_to_prim(
                    stage,
                    prim,
                    _physicsMaterialPath,
                )

        # add collision approximation
        prim = stage.GetPrimAtPath( self.default_zero_env_path + "/cabinet")
        collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
        if not collision_api:
            collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        
        collision_api.CreateApproximationAttr().Set("convexDecomposition")

        
        
                          
        # self._sim_config.apply_articulation_settings(
        #     "cabinet", get_prim_at_path(cabinet.prim_path), self._sim_config.parse_actor_config("cabinet")
        # )

        device = self._device
        # hand_pos, hand_rot = self.get_ee_pose()

        data = self.annotations[int(GAPartNet_ID)]
       
        for d in data:
            if d['link_name'] == str(bbox_link).split('/')[-1]:
                corners = torch.tensor(d['bbox'])
        
        # print('before: ', corners)
        # self.corners = scale_max * corners 
        # print('after: ', self.corners)
        # exit()
        num_envs = self._num_envs
        self.bboxes = torch.zeros(( num_envs, 8, 3), device=device)
        

        link_path =  f"/World/envs/env_0/cabinet/{link_name}"
        prim = get_prim_at_path(link_path)
        
        matrix = inv(np.array(omni.usd.get_world_transform_matrix(prim)))
        
        self.forwardDir = matrix[0:3, 2]
        self.forwardDir = self.forwardDir/np.linalg.norm(self.forwardDir)
        self.forwardDir = torch.tensor(self.forwardDir).to(self._device).repeat((num_envs,1))

        corners = corners.to(self._device)
        self.handle_short = torch.zeros((num_envs, 3))
        self.handle_out = torch.zeros((num_envs, 3))
        self.handle_long = torch.zeros((num_envs, 3))

        for idx in range(num_envs):
            handle_out = corners[0] - corners[4]
            handle_long = corners[1] - corners[0]
            handle_short = corners[3] - corners[0]




            self.handle_short[idx] = handle_short
            self.handle_out[idx] = handle_out
            self.handle_long[idx] = handle_long
        
        self.handle_short = self.handle_short.to(self._device)
        self.handle_out = self.handle_out.to(self._device)
        self.handle_long = self.handle_long.to(self._device)
        
        # self.corners = corners.repeat((num_envs, 1,1)) * scale_max

        # self.corners_obj = corners.repeat((num_envs, 1,1)) * scale_max

        # self.centers_obj = ((corners[0] * scale_max +  corners[6] * scale_max )/2.0).repeat((num_envs, 1)).to(torch.float).to(self._device)
        

        self.corners = corners.repeat((num_envs, 1,1)) * scale_min

        self.corners_obj = corners.repeat((num_envs, 1,1)) * scale_min

        self.centers_obj = ((corners[0] * scale_min +  corners[6] * scale_min )/2.0).repeat((num_envs, 1)).to(torch.float).to(self._device)

        # world = World()
        t = position
        # print('self corners: ', self.corners)
        box = self.rotate_points_around_z(self.corners.cpu().numpy(), np.array([0,0,0]), 90)
        box = torch.tensor(box)
        box += torch.tensor(t)
        # print('box: ', box)
        self.corners = box

        self.centers_orig = ((corners[0] +  corners[6])/2.0).repeat((num_envs, 1)).to(torch.float).to(self._device) 
        self.centers = self.centers_orig.clone() 

        children = get_all_matching_child_prims(self.default_zero_env_path + "/cabinet")
            # print('children: ', children)
        prims = [x for x in children if x.IsA(UsdPhysics.Joint)] 
        
        for prim in prims:
            joint = pxr.UsdPhysics.PrismaticJoint.Get(stage, prim.GetPath())	
            if joint:
                upper_limit = joint.GetUpperLimitAttr().Get() #GetAttribute("xformOp:translate").Get()
                # print(prim.GetPath(), "upper_limit", upper_limit)
                mobility_prim = prim.GetParent().GetParent()
                mobility_xform = pxr.UsdGeom.Xformable.Get(stage, mobility_prim.GetPath())
                scale_factor = mobility_xform.GetOrderedXformOps()[2].Get()[0]
                # print("scale_factor", scale_factor)
                joint.CreateUpperLimitAttr(upper_limit * scale_factor)

        # world = World()
        # while True:
            
        #     color = 4283782485
        #     my_debugDraw = get_debug_draw_interface()
        #     corners = self.corners.clone()
        #     # for idx in range(self._num_envs):
        #         # corners[idx] = (self.corners[idx]).to(torch.float32).to(self._device)
            
        #     corners = corners.cpu().numpy()
        #     for corner in corners:
        #         my_debugDraw.draw_line(carb.Float3(corner[0]),color, carb.Float3(corner[4]), color)
        #         my_debugDraw.draw_line(carb.Float3(corner[1]),color, carb.Float3(corner[0]), color)
        #         my_debugDraw.draw_line(carb.Float3(corner[3]),color, carb.Float3(corner[0]), color)
            
        #     world.step(render=True)

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)
        # enable ccd: /physicsScene
        physicsScenePath = '/physicsScene'
        stage = get_current_stage()
        scene = UsdPhysics.Scene.Get(stage, physicsScenePath)
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physxSceneAPI.CreateEnableCCDAttr().Set(True)

        

        num_envs = self._num_envs

        self.actions = torch.zeros((num_envs, self._num_actions), device=self._device)
    
    def get_ee_pose_o(self):
        hand_position_w, hand_quat_w = self._frankas._hands.get_world_poses(clone=True)
       


        position_w = hand_position_w - self._cabinets.get_world_poses(clone=True)[0]
        rotation_matrix = quaternion_to_matrix( torch.tensor(self.cabinet_orientation).float() ).to(self._device)
        position_w =  torch.matmul(rotation_matrix.T, position_w.T).T


        rotation_matrix = quaternion_to_matrix( torch.tensor(self.cabinet_orientation).float() ).to(self._device)
        all_orientations = rotation_matrix.repeat((self._num_envs, 1,1)).to(torch.float).to(self._device)

        gripper_rot = quats_to_rot_matrices(hand_quat_w)
        hand_rot_w_object_space =  all_orientations.mT @ gripper_rot

        tmp  = matrices_to_euler_angles(hand_rot_w_object_space)
        

        roll, pitch, yaw = tmp[:, 0], tmp[:, 1], tmp[:, 2]
        # print(roll.shape)
        # print(pitch.shape)
        # print(yaw.shape)

        hand_rot_w_object_space =  quat_from_euler_xyz(roll, pitch, yaw)
        return position_w, hand_rot_w_object_space



    def get_ee_pose(self):
        hand_position_w, hand_quat_w = self._frankas._hands.get_world_poses(clone=True)
        diff = self._cabinets.get_world_poses(clone=True)[0]
        diff[:,2] = 0.0
        hand_position_w = hand_position_w - diff

        # rotation_matrix = quaternion_to_matrix( torch.tensor(self.cabinet_orientation).float() ).to(self._device)
        # hand_position_w =  torch.matmul(rotation_matrix.T, hand_position_w.T).T
       
        # print('hand_position_w: ', hand_position_w)
        # exit()
        ee_pos_offset = torch.tensor([0.0, 0.0, 0.105]).repeat((self._num_envs, 1)).to(hand_position_w.device)
        ee_rot_offset = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat((self._num_envs, 1)).to(hand_quat_w.device)
        # print(ee_pos_offset.shape)
        # print(ee_rot_offset.shape)
        position_w, quat_w = combine_frame_transforms(
            hand_position_w, hand_quat_w,  ee_pos_offset, ee_rot_offset
        )
        return position_w, quat_w

    def get_observations(self) -> dict:

        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)
        
        
        
        # drawer_pos, drawer_rot = self._cabinets._drawers.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        self.cabinet_dof_vel = self._cabinets.get_joint_velocities(clone=False)
      
        # self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        hand_pos, hand_rot = self.get_ee_pose_o()
        
        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )

        
        
        torch.set_printoptions(sci_mode=False)
        forwardDir = torch.tensor([1.0,0.0,0.0])
        centers = (self.centers_obj.to(self._device) +  forwardDir * self.cabinet_dof_pos[:, 3].unsqueeze(-1)).to(torch.float32).to(self._device)
        tool_pos_diff = hand_pos  - centers
        
        normalized_dof_pos = (self.cabinet_dof_pos[:, 3] - self.cabinet_dof_lower_limits) / (self.cabinet_dof_upper_limits - self.cabinet_dof_lower_limits)

        # print(self.cabinet_dof_pos[:,1].unsqueeze(-1).shape)
        # print(self.cabinet_dof_vel[:,1].unsqueeze(-1).shape)
        # print(normalized_dof_pos.unsqueeze(-1).shape)
        # exit()
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,  # 12
                franka_dof_vel * self.dof_vel_scale, #12
                tool_pos_diff, # 3
                hand_pos, # 3
                hand_rot, # 4
                centers, # 3
                # handle_out.reshape(-1, 3),
                # handle_long.reshape(-1, 3),
                # handle_short.reshape(-1, 3),
                self.cabinet_dof_pos[:, 3].unsqueeze(-1),
                normalized_dof_pos.unsqueeze(-1), # 1
                self.cabinet_dof_vel[:, 3].unsqueeze(-1), # 1 
            ),
            dim=-1,
        )
        # print(normalized_dof_pos)
        # print('==========')
        # print('obs: ',  self.obs_buf[0,:])
        # exit()
        observations = {self._frankas.name: {"obs_buf": self.obs_buf.to(torch.float32)}}
        # observations = {self._frankas.name: {"obs_buf": torch.zeros((self._num_envs, self._num_observations))}}
        if self.after_reset: 
            # print('obs: ', observations)
            self.after_reset = False
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
        mode_prob = (self.actions[:, 0]  + 1.0 )/2
        # sample 0 or 1 based on mode_prob
        # mode = torch.bernoulli(mode_prob).to(torch.int32)

        mode = (mode_prob > 0.5).long()
        base_indices = torch.nonzero(mode).long()
        arm_indices = torch.nonzero(1 - mode).long()


        # mode = self.actions[:, 0] <= 0
        # base_indices =  torch.nonzero(mode).long()

     
        # mode = self.actions[:, 0] > 0
        # arm_indices =  torch.nonzero(mode).long()

   
        
        self.actions[:, 1:] = (self.actions[:, 1:] + 1.0) / 2.0
        current_joint_positons = self._frankas.get_joint_positions(clone=False)
        base_positions = current_joint_positons[:, :3]
        arm_positions = current_joint_positons[:, 3:]

        # print(base_positions.shape)
        # print(arm_positions.shape)
        # print(self.franka_dof_targets.shape)
        # exit()

        targets = self.actions[:, 1:] *(self.franka_dof_upper_limits - self.franka_dof_lower_limits) + self.franka_dof_lower_limits

        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        if len(base_indices) > 0:
            self.franka_dof_targets[base_indices, :3 ] =  base_positions[base_indices]
        if len(arm_indices) > 0:
            self.franka_dof_targets[arm_indices, 3:] =  arm_positions[arm_indices]
        

        
        # self.franka_dof_targets[:,:3] = 0.0

        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # print(self.franka_dof_targets)
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        # timeline = omni.timeline.get_timeline_interface()
        # timeline.stop()
        # timeline.play()
        # self.initialize_views(self._scene)

        

        self.after_reset = True
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # if self.translations_orig is None:
        #     self.translations_orig = self._frankas.get_world_poses(indices=indices)[0]

    
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
   

        # # reset cabinet
        self._cabinets.set_joint_positions(
            torch.zeros_like(self._cabinets.get_joint_positions(clone=False)), indices=indices
        )
        self._cabinets.set_joint_velocities(
            torch.zeros_like(self._cabinets.get_joint_velocities(clone=False)), indices=indices
        )

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)

        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device)
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        

        

    def post_reset(self):
        num_envs = self._num_envs
        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        cabinet_dof_limits = self._cabinets.get_dof_limits()
        self.cabinet_dof_lower_limits = cabinet_dof_limits[0, 1, 0].to(device=self._device)
        self.cabinet_dof_upper_limits = cabinet_dof_limits[0, 1, 1].to(device=self._device)

        # if self.num_props > 0:
        #     self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
        #     self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
        #         self._num_envs, self.num_props
        #     )

        # randomize all envs
        indices = torch.arange(num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # import pdb; pdb.set_trace()
        self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        self.rew_buf[:] = 0.0
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