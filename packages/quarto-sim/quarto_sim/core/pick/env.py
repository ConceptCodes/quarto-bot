import os
import numpy as np
import mujoco
import mujoco.viewer


class PickEnv:
    def __init__(self, xml_path=None):
        if xml_path is None:
            xml_path = os.path.join(
                os.path.dirname(__file__), "../scenes/pick_scene.xml"
            )
            xml_path = os.path.abspath(xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def reset(self):
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        joint_addrs = [int(self.model.joint(name).qposadr) for name in joint_names]
        init_pose = [0.0, -0.5, 1.0, 0.5, 0.0, 0.0]
        for addr, val in zip(joint_addrs, init_pose):
            self.data.qpos[addr] = val
        self._get_random_piece_position()
        self._step_count = 0
        # Store initial piece position for later reward calculation
        piece_addr = int(self.model.joint("quarto_joint").qposadr)
        self._initial_piece_pos = np.copy(self.data.qpos[piece_addr : piece_addr + 3])
        return self._get_obs()

    def _get_random_piece_position(self):
        joint_addr = int(self.model.joint("quarto_joint").qposadr)
        # Random position in front of gripper
        # x =  np.random.uniform(0.5, 0.75)
        x = 0.3
        y = 0
        z = 0.05
        self.data.qpos[joint_addr : joint_addr + 3] = [x, y, z]

    def step(self, action=None):
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        joint_addrs = [int(self.model.joint(name).qposadr) for name in joint_names]
        # Apply action as delta to each joint
        if action is not None:
            action = np.clip(action, -0.05, 0.05)  # limit step size for each joint
            for i, addr in enumerate(joint_addrs):
                self.data.qpos[addr] += action[i]
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        # Get gripper position from site
        gripper_site_id = self.model.site("gripperframe").id
        gripper_pos = np.array(self.data.site_xpos[gripper_site_id])

        # Get piece position
        piece_addr = int(self.model.joint("quarto_joint").qposadr)
        piece_pos = self.data.qpos[piece_addr : piece_addr + 3]
        dist = np.linalg.norm(gripper_pos[:2] - piece_pos[:2])
        close_enough = dist < 0.03

        gripper_addr = int(self.model.joint("gripper").qposadr)
        gripper_closed = self.data.qpos[gripper_addr] < -0.02
        piece_lifted = piece_pos[2] > 0.07

        reward = -2.0 * dist  # Strong penalty for distance

        # Penalize folding/self-collision (if you can detect it)
        # reward -= 5.0 if self._is_folded() else 0.0

        # Penalize large joint angles (optional)
        reward -= 0.1 * np.sum(
            np.abs(self.data.qpos)
        )  # or penalize deviation from neutral

        # Reward for being close and attempting grasp
        if close_enough:
            reward += 2.0
            if gripper_closed:
                reward += 2.0
        if piece_lifted:
            reward += 100.0  # Only big reward for actual success

        # Small step penalty
        reward -= 0.01
        done = self._step_count >= 100  # Limit episode length to 100 steps

        # Optional: penalty if piece moves further from gripper (encourage approach)
        if hasattr(self, '_prev_dist'):
            reward -= (dist - self._prev_dist) * 2  # penalize increasing distance
        self._prev_dist = dist

        # Increase episode length to 100 steps

        info = {
            "distance": dist,
            "gripper_pos": gripper_pos,
            "piece_pos": piece_pos,
            "gripper_closed": gripper_closed,
            "piece_lifted": piece_lifted,
        }
        self._step_count += 1
        return obs, reward, done, info

    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
        elif mode == "rgb_array":
            # Render to offscreen buffer and return as numpy array
            width, height = 480, 360  # fit default MuJoCo framebuffer
            with mujoco.Renderer(self.model, width, height) as renderer:
                renderer.update_scene(self.data)
                frame = renderer.render()
                return np.asarray(frame)
        else:
            raise NotImplementedError(f"Render mode {mode} not supported.")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        gripper_site_id = self.model.site("gripperframe").id
        gripper_pos = np.array(self.data.site_xpos[gripper_site_id])
        piece_addr = int(self.model.joint("quarto_joint").qposadr)
        piece_pos = self.data.qpos[piece_addr : piece_addr + 3]
        return np.concatenate([self.data.qpos, self.data.qvel, gripper_pos, piece_pos])
