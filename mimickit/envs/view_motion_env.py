import envs.base_env as base_env
import envs.char_env as char_env
import anim.motion as motion
import anim.motion_lib as motion_lib
import engines.engine as engine

import numpy as np
import torch
from util.logger import Logger

class ViewMotionEnv(char_env.CharEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize):
        self._time_scale = 1.0
        self._render_key_points_enabled = env_config.get("render_key_points", True)
        self._enable_motion_switch = env_config.get("enable_motion_switch", False)
        self._preview_motion_id = env_config.get("preview_motion_id", 0)
        engine_config["sim_freq"] = engine_config["control_freq"]

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize)
        return

    def _build_envs(self, env_config, num_envs):
        super()._build_envs(env_config, num_envs)

        motion_file = env_config["motion_file"]
        self._load_motions(motion_file)
        num_motions = self._motion_lib.get_num_motions()
        if (num_motions > 0):
            self._preview_motion_id %= num_motions
        return
    
    def _build_character(self, env_id, env_config, color=None):
        char_file = env_config["char_file"]
        char_id = self._engine.create_obj(env_id=env_id, 
                                          obj_type=engine.ObjType.articulated,
                                          asset_file=char_file, 
                                          name="character",
                                          start_pos=self._init_root_pos.cpu().numpy(),
                                          start_rot=self._init_root_rot.cpu().numpy(),
                                          enable_self_collisions=False,
                                          disable_motors=True,
                                          color=color)
        return char_id

    def _load_motions(self, motion_file):
        self._motion_lib = motion_lib.MotionLib(motion_file=motion_file, 
                                                kin_char_model=self._kin_char_model,
                                                device=self._device)
        return

    def _update_misc(self):
        super()._update_misc()
        self._sync_motion()
        return

    def _apply_action(self, actions):
        return

    def _sync_motion(self):
        motion_ids = self._get_env_motion_ids()
        motion_times = self._time_buf * self._time_scale
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        joint_dof = self._motion_lib.joint_rot_to_dof(joint_rot)
        
        char_id = self._get_char_id()
        
        self._engine.set_root_pos(None, char_id, root_pos)
        self._engine.set_root_rot(None, char_id, root_rot)
        self._engine.set_root_vel(None, char_id, 0.0)
        self._engine.set_root_ang_vel(None, char_id, 0.0)
        
        self._engine.set_dof_pos(None, char_id, joint_dof)
        self._engine.set_dof_vel(None, char_id, 0.0)
        
        body_pos, body_rot = self._kin_char_model.forward_kinematics(root_pos=root_pos,
                                                                     root_rot=root_rot,
                                                                     joint_rot=joint_rot)
        
        if (self._has_key_bodies()):
            self._ref_body_pos[:] = body_pos

        self._engine.set_body_pos(None, char_id, body_pos)
        self._engine.set_body_rot(None, char_id, body_rot)
        return

    def _render_scene(self):
        super()._render_scene()
        self._render_key_points()
        return
    
    def _setup_gui(self):
        super()._setup_gui()

        def toggle_key_points():
            self._render_key_points_enabled = not self._render_key_points_enabled
            state = "on" if self._render_key_points_enabled else "off"
            Logger.print("View motion key points: {}".format(state))
            return
        self._engine.register_keyboard_callback("K", toggle_key_points)

        num_motions = self._motion_lib.get_num_motions()
        if (self._enable_motion_switch and num_motions > 1):
            def prev_motion():
                self._change_motion(-1)
                return
            self._engine.register_keyboard_callback("LEFT", prev_motion)

            def next_motion():
                self._change_motion(1)
                return
            self._engine.register_keyboard_callback("RIGHT", next_motion)

        Logger.print("View motion controls: Enter play/pause, Space step, K toggle key points")
        if (self._enable_motion_switch and num_motions > 1):
            Logger.print("View motion controls: Left/Right switch clips")
            self._log_current_motion()
        return
    
    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        
        if (self._has_key_bodies()):
            char_id = self._get_char_id()
            body_pos = self._engine.get_body_pos(char_id)
            self._ref_body_pos = torch.zeros_like(body_pos)
        return

    def _get_env_motion_ids(self):
        num_motions = self._motion_lib.get_num_motions()
        if (self._enable_motion_switch):
            motion_ids = torch.full_like(self._env_ids, self._preview_motion_id)
        else:
            motion_ids = torch.remainder(self._env_ids, num_motions)
        return motion_ids

    def _update_done(self):
        motion_ids = self._get_env_motion_ids()
        motion_len = self._motion_lib.get_motion_length(motion_ids)
        motion_loop_mode = self._motion_lib.get_motion_loop_mode(motion_ids)
        self._done_buf[:] = compute_done(self._done_buf, self._time_buf, 
                                         motion_len, motion_loop_mode)
        return

    def _render_key_points(self):
        if (self._render_key_points_enabled and self._has_key_bodies()):
            line_width = 2.0
            num_key_bodies = self._key_body_ids.shape[0]
            cols = np.array(3 * num_key_bodies * [[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            
            num_envs = self.get_num_envs()
            for i in range(num_envs):
                key_body_pos = self._ref_body_pos[i][self._key_body_ids]
                key_body_pos = key_body_pos.cpu().numpy()

                start_verts = 0.2 * np.array([[-1.0, 0.0, 0.0],
                                        [0.0, -1.0, 0.0],
                                        [0.0, 0.0, -1.0]],
                                       dtype=np.float32)
                
                end_verts = 0.2 * np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]],
                                       dtype=np.float32)

                key_body_pos = np.expand_dims(key_body_pos, -2)
                start_verts = np.expand_dims(start_verts, 0)
                start_verts = key_body_pos + start_verts
                end_verts = np.expand_dims(end_verts, 0)
                end_verts = key_body_pos + end_verts

                start_verts = start_verts.reshape([-1, 3])
                end_verts = end_verts.reshape([-1, 3])
                
                self._engine.draw_lines(i, start_verts, end_verts, cols, line_width)

        return

    def _change_motion(self, delta):
        num_motions = self._motion_lib.get_num_motions()
        motion_id = (self._preview_motion_id + delta) % num_motions
        self._set_motion(motion_id)
        return
    
    def _set_motion(self, motion_id):
        motion_id = int(motion_id)
        if (motion_id == self._preview_motion_id):
            return

        self._preview_motion_id = motion_id
        self._reset_envs(self._env_ids)
        self._sync_motion()
        self._update_observations()
        self._update_info()
        self._log_current_motion()
        return
    
    def _log_current_motion(self):
        num_motions = self._motion_lib.get_num_motions()
        motion_file = self._motion_lib.get_motion_file(self._preview_motion_id)
        Logger.print("Viewing motion {:d}/{:d}: {}".format(self._preview_motion_id + 1, num_motions, motion_file))
        return
    
    def _get_char_color(self):
        engine_name = self._engine.get_name()
        if (engine_name == "isaac_lab"):
            col = np.array([0.25, 0.4, 0.1])
        elif (engine_name == "newton"):
            col = np.array([0.3, 0.5, 0.1])
        else:
            col = np.array([0.5, 0.9, 0.1])
        return col


@torch.jit.script
def compute_done(done_buf, time, motion_len, motion_loop_mode):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    num_loops = 5

    timeout = torch.zeros_like(done_buf)
    end_time = motion_len.clone()
    loop_ids = motion_loop_mode == motion.LoopMode.WRAP.value
    end_time[loop_ids] *= num_loops

    timeout = time >= end_time
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    done[timeout] = base_env.DoneFlags.TIME.value

    return done
