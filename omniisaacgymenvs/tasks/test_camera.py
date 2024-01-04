# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from gym import spaces
import numpy as np
import torch

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from omniisaacgymenvs.tasks.base.rl_task import RLTask


class TestCamera(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 500

        self._num_observations = self.camera_width * self.camera_height * 3
        self._num_actions = 1

        # use multi-dimensional observation for camera RGB
        self.observation_space = spaces.Box(
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * -np.Inf, 
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * np.Inf)

        RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

        self.camera_type = self._task_cfg["env"].get("cameraType", 'rgb')
        self.camera_width = self._task_cfg["env"]["cameraWidth"]
        self.camera_height = self._task_cfg["env"]["cameraHeight"]
        
        self.camera_channels = 3
        self._export_images = self._task_cfg["env"]["exportImages"]

    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, self.camera_width, self.camera_height, 3), device=self.device, dtype=torch.float)
    
    def calculate_metrics(self):
        self.rew_buf[:] = 0.0
        self.rew_buf[:] = self.rew_buf[:].to(torch.float32)
    
    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        # self.reset_buf = torch.where(self.cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )
    
    def get_cube(self):
        from omni.isaac.core.objects import DynamicCuboid
        DynamicCuboid(
                name="red cube",
                position=np.array([0, 0, 0.5]),
                orientation=np.array([1, 0, 0, 0]),
                prim_path=self.default_zero_env_path + "/cube",
                scale=np.array([1, 1, 1]),
                size=1.0,
                color=np.array([255, 0, 0]),
            )

    def set_up_scene(self, scene) -> None:
        # self.get_cartpole()
        self.get_cube()

        RLTask.set_up_scene(self, scene)

        # start replicator to capture image data
        self.rep.orchestrator._orchestrator._is_started = True

        # set up cameras
        self.render_products = []
        env_pos = self._env_pos.cpu()
        for i in range(self._num_envs):
            camera = self.rep.create.camera(
                position=(-4.2 + env_pos[i][0], env_pos[i][1], 3.0), look_at=(env_pos[i][0], env_pos[i][1], 2.55))
            render_product = self.rep.create.render_product(camera, resolution=(self.camera_width, self.camera_height))
            self.render_products.append(render_product)

        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
        self.pytorch_writer.attach(self.render_products)

        print('set up scene: ', 'set_up_scene')

        return

    def get_observations(self) -> dict:
        # retrieve RGB data from all render products
        images = self.pytorch_listener.get_rgb_data()
        if images is not None:
            if self._export_images:
                from torchvision.utils import save_image, make_grid
                img = images/255
                save_image(make_grid(img, nrows = 2), 'test_camera_export.png')

            self.obs_buf = torch.swapaxes(images, 1, 3).clone().float()/255.0
        else:
            print("Image tensor is NONE!")

        return self.obs_buf
    
    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)
    
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    
    # ====================================================================================================

    def post_reset(self):
        pass

        
