import os
import numpy as np
import mujoco
import mujoco.viewer


class PickEnv:
    def __init__(self, xml_path=None):
        if xml_path is None:
            # Default to pick_scene.xml in the same directory
            xml_path = os.path.join(
                os.path.dirname(__file__), "../scenes/pick_scene.xml"
            )
            xml_path = os.path.abspath(xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def reset(self):
        # Reset state in-place so viewer always sees the latest state
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        # Set initial pose for robot arm joints (now 6 joints)
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        joint_addrs = [int(self.model.joint(name).qposadr) for name in joint_names]
        # Example: slightly bent arm pose (6 values)
        init_pose = [0.0, -0.5, 1.0, 0.5, 0.0, 0.0]
        for addr, val in zip(joint_addrs, init_pose):
            self.data.qpos[addr] = val
        # Find the freejoint address for 'quarto_piece'
        joint_addr = int(self.model.joint("quarto_joint").qposadr)
        # Set random position in front of gripper (example: y in [0.15, 0.25])
        x = 0.0
        y = np.random.uniform(0.15, 0.25)
        z = 0.05
        self.data.qpos[joint_addr : joint_addr + 3] = [x, y, z]
        self._step_count = 0
        return self._get_obs()

    def step(self, action=None):
        # Action: delta for robot arm joints (now 6 joints)
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
        reward = -dist
        if close_enough:
            reward += 1.0  # bonus for reaching
        done = close_enough or self._step_count > 49
        info = {"distance": dist, "gripper_pos": gripper_pos, "piece_pos": piece_pos}
        self._step_count += 1
        return obs, reward, done, info

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        # Return a dummy observation (e.g., qpos)
        return np.copy(self.data.qpos)
