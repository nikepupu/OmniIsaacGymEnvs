# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from typing import Optional

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema


class KinovaMobile(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "kinova",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        self._usd_path = "/home/nikepupu/Desktop/mec_kinova_gripper.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths=[
            # base
            "virtual_base_x/base_y_base_x",
            "virtual_base_y/base_theta_base_y",
            "virtual_base_theta/base_link_base_theta",
            #arm
            "base_link/joint_1",
            "shoulder_link/joint_2",
            "half_arm_1_link/joint_3",
            "half_arm_2_link/joint_4",
            "forearm_link/joint_5",
            "spherical_wrist_1_link/joint_6",
            "spherical_wrist_2_link/joint_7",

            #hand
            "left_outer_finger/end_hand_prismatic_joint_left",
            "right_outer_finger/end_hand_prismatic_joint_right",

        ]


        drive_type = ['linear'] * 2  + ['angular'] +   ["angular"] * 7 + ["linear"] * 2
        default_dof_pos = [math.degrees(x) for x in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] + [0.00, 0.00]
        stiffness = [800]*3 + [400 * np.pi / 180] * 7 + [10000] * 2
        damping =  [200]*3 + [80 * np.pi / 180] * 7 + [100] * 2
        max_force = [300, 300, 300, 87, 87, 87, 87, 12, 12, 12, 200, 200]
        max_force = [x * 10  for x in max_force]
        max_velocity = [50 ] * 3 +  [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]] + [0.2, 0.2]
        # max_velocity = [x * 2  for x in max_velocity]

        for i, dof in enumerate(dof_paths):
            print(f"{self.prim_path}/{dof}")
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )
            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )
            print('Done')

    def set_kinova_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)

