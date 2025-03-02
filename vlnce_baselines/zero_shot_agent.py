import os
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines
import cv2

import numpy as np
import math
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import networkx as nx

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import ZeroShotGraphMap

from .utils import get_camera_orientations12
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from llm.utils import llm_waypoint_predictor_single_view, vis_ghost_nodes
from vlnce_baselines.models.utils import angle_feature_torch
from llm.prompting.prompt_manager import PromptManager
from llm.prompting.GPT_api import gpt4v_infer


@baseline_registry.register_trainer(name="Zero-Shot-AO-Planner")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len)  # * 0.97 transfered gt path got 0.96 spl
        self.case_id = 0
        self.prompt_manager = PromptManager(config)

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],  # Back
                'Down': [-math.pi / 2, 0 + shift, 0],  # Down
                'Front': [0, 0 + shift, 0],  # Front
                'Right': [0, math.pi / 2 + shift, 0],  # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],  # Left
                'Up': [math.pi / 2, 0 + shift, 0],  # Up
            }
            sensor_uuids = []
            # H = 224
            H = 512  # check this
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _initialize_policy(
            self,
            config: Config,
            load_from_ckpt: bool,
            observation_space: Space,
            action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(
            torch.load(cwp_fn, map_location=torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS, 'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                                  output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.IL.lr)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")))
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                                                        device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                                                                device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                        self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                            split=self.split, role=role
                        ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every = self.config.IL.log_every
        writer = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter - idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)

    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.

            with autocast():
                self.rollout('train', ml_weight, sample_ratio)
            self.scaler.scale(self.loss).backward()  # self.loss.backward()
            self.scaler.step(self.optimizer)  # self.optimizer.step()
            self.scaler.update()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})

        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
            self,
            checkpoint_path: str,
            writer: TensorboardWriter,
            checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1

        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],  # Back
                'Down': [-math.pi / 2, 0 + shift, 0],  # Down
                'Front': [0, 0 + shift, 0],  # Front
                'Right': [0, math.pi / 2 + shift, 0],  # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],  # Left
                'Up': [math.pi / 2, 0 + shift, 0],  # Up
            }
            sensor_uuids = []
            # H = 224
            H = 512  # check this
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False,  # unseen: 11006
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
            aggregated_states = {}
            num_episodes = len(self.stat_eps)
            if num_episodes != 0:
                for stat_key in next(iter(self.stat_eps.values())).keys():
                    aggregated_states[stat_key] = (
                            sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
                    )

                print('num_episodes:', num_episodes)
                for k, v in aggregated_states.items():
                    print(f"Average episode {k}: {v:.6f}")

        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                    sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total, dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k, v in aggregated_states.items():
                v = torch.tensor(v * num_episodes).cuda()
                cat_v = gather_list_and_concat(v, self.world_size)
                v = (sum(cat_v) / total).item()
                aggregated_states[k] = v

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}  # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict

        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k, v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori'] * self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def align_with_waypoint_model(self, observations, llm_waypoint_path_cands_batch):
        batch_size = observations['rgb'].shape[0]
        ''' encoding rgb/depth at all directions ----------------------------- '''
        NUM_IMGS = 12
        depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
        rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

        # reverse the order of input images to clockwise
        a_count = 0
        for i, (k, v) in enumerate(observations.items()):
            if 'depth' in k:  # You might need to double check the keys order
                for bi in range(v.size(0)):
                    ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                    depth_batch[ra_count + bi * NUM_IMGS] = v[bi]
                    rgb_batch[ra_count + bi * NUM_IMGS] = observations[k.replace('depth', 'rgb')][bi]
                a_count += 1

        ''' waypoint prediction ----------------------------- '''

        # for cand
        cand_rgb = []
        cand_depth = []
        cand_angle_fts = []
        cand_img_idxes = []
        cand_angles = []
        cand_distances = []

        for j in range(batch_size):
            # for angle & distance
            waypoints = llm_waypoint_path_cands_batch[j]["waypoints"]
            angle_rad_cc = [waypoint["angle"] for waypoint in waypoints][
                           ::-1]  # need to inverse to align with waypoint model
            angle_rad_cc = torch.tensor(angle_rad_cc)
            angle_rad_c = 2 * math.pi - angle_rad_cc

            cand_angle_fts.append(angle_feature_torch(angle_rad_c))
            cand_angles.append(angle_rad_cc.tolist())

            distance = [waypoint["distance"] for waypoint in waypoints]
            cand_distances.append(distance)

            # for img idxes
            angle_idxes = angle_rad_c * 120 / (2 * math.pi)  # align with waypoint model
            img_idxes = 12 - (angle_idxes.numpy() + 5) // 10
            img_idxes = img_idxes.astype(int)
            img_idxes[img_idxes == 12] = 0
            cand_img_idxes.append(img_idxes)

        outputs = {
            'cand_rgb': cand_rgb,  # [K x 2048]
            'cand_depth': cand_depth,  # [K x 128]
            'cand_img_idxes': cand_img_idxes,  # [K]
            'cand_angles': cand_angles,  # [K]
            'cand_distances': cand_distances,  # [K]
        }
        return outputs

    def rollout(self, mode, ml_weight=None, sample_ratio=None):

        self.envs.resume_all()
        observations = self.envs.reset()

        ob = observations[0]
        instruction = ob['instruction']['text']
        instruction_data = {"instruction": instruction}

        instr_max_len = self.config.IL.max_text_len  # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.stat_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.path_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [ZeroShotGraphMap(have_real_pos,
                               self.config.IL.loc_noise,
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs

        self.path_graph = [nx.DiGraph() for _ in range(self.envs.num_envs)]

        self.prompt_manager.history = []
        self.prompt_manager.planning = ["Navigation has just started, with no planning yet."]

        self.case_id += 1
        waypoint_path_coord = []

        if not os.path.exists(self.config.LLM_DIR):
            os.makedirs(self.config.LLM_DIR)

        for stepk in range(self.max_len):

            total_actions += self.envs.num_envs

            ghost_node_to_img_dict = {}
            img_to_ghost_node_dict = {}

            print(f'############ {stepk} ############')
            scene_id = self.envs.current_episodes()[0].scene_id.split('/')[-1].split('.')[-2]

            metrics = self.envs.get_metrics()
            agent_positions = [item['position']['position'][-1] for item in metrics]  # not camera position
            camera_positions = [pos + np.array([0, 1.25, 0]) for pos in agent_positions]
            base_headings = [item['top_down_map_vlnce']['agent_angle'] for item in metrics]

            position, heading = camera_positions[0], base_headings[0]

            image_list = [('front', 'rgb'), ('left', 'rgb_90'), ('backward', 'rgb_180'), ('right', 'rgb_270')]
            rgb_images = []
            llm_results = []
            waypoints = []
            paths = []
            vis_ghost_cnt = self.gmaps[0].ghost_cnt
            for direction, rgb_name in image_list:
                img_file_name = os.path.join(self.config.LLM_DIR, f'case{self.case_id}_step{stepk}_{rgb_name}__{scene_id}_p{position[0]:.3f}_{position[1]:.3f}_{position[2]:.3f}_h{heading:.3f}.png')

                instr_file_name = img_file_name.replace('.png', '_instr.json')
                with open(instr_file_name, 'w') as file:
                    json.dump(instruction_data, file)

                try:
                    llm_result, llm_waypoint_path_cands = llm_waypoint_predictor_single_view(ob, img_file_name, rgb_name, position, heading)
                except:
                    llm_result, llm_waypoint_path_cands = None, None
                    print('################ Gemini Errors in waypoint and path prediction ################')

                if llm_result:
                    vis_path_name = img_file_name.replace('.png', "_path_results.jpg")
                    time.sleep(1)
                    rgb_images.append({direction: vis_path_name})
                    llm_results.append(llm_result)
                    waypoint_path_coord.extend(llm_waypoint_path_cands["paths"])
                    waypoints.extend(llm_waypoint_path_cands["waypoints"])
                    paths.extend(llm_waypoint_path_cands["paths"])

                    ghost_node_image, vis_ghost_cnt, img_to_ghost_node_dict, ghost_node_to_img_dict = vis_ghost_nodes(
                        img_file_name, llm_result, vis_ghost_cnt, img_to_ghost_node_dict, ghost_node_to_img_dict
                    )
                    ghost_node_image_name = img_file_name.replace('.png', "_ghost_results.jpg")
                    cv2.imwrite(ghost_node_image_name, ghost_node_image)



            llm_waypoint_path_cands_batch = [{"waypoints": waypoints, "paths": paths}]

            for i in range(self.envs.num_envs):
                llm_waypoint_path_cands = llm_waypoint_path_cands_batch[i]
                agent_position = agent_positions[i].tolist()

                # update path graph
                self.path_graph[i].add_node(tuple(agent_position))
                waypoints_positions = [waypoint["position"].tolist() for waypoint in llm_waypoint_path_cands["waypoints"]]
                waypoints_paths = llm_waypoint_path_cands["paths"]
                for waypoint_position, waypoints_path in zip(waypoints_positions, waypoints_paths):
                    self.path_graph[i].add_node(tuple(waypoint_position))
                    waypoints_path = [agent_position] + waypoints_path
                    self.path_graph[i].add_edge(tuple(agent_position), tuple(waypoint_position), path=waypoints_path)
                    self.path_graph[i].add_edge(tuple(waypoint_position), tuple(agent_position),
                                                path=waypoints_path[::-1])

            wp_outputs = self.align_with_waypoint_model(observations=batch, llm_waypoint_path_cands_batch=llm_waypoint_path_cands_batch)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                waypoints_positions = [waypoint["position"] for waypoint in llm_waypoint_path_cands_batch[i]["waypoints"]]
                cand_pos.append(waypoints_positions)

            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs


            for i in range(self.envs.num_envs):
                self.gmaps[i].update_graph(prev_vp[i], stepk + 1,
                                           cur_vp[i], cur_pos[i], #cur_embeds,
                                           cand_vp[i], cand_pos[i], # cand_pos[i] is waypoints_positions
                                           cand_real_pos[i])

            agent_json_output = os.path.join(self.config.LLM_DIR, f'case{self.case_id}_step{stepk}_{scene_id}_agent_output.json')
            if os.path.exists(agent_json_output):
                # load existing json data
                print(f"load file {agent_json_output}")
                with open(agent_json_output, 'r') as f:
                    json_output = json.load(f)
            else:
                messages = self.prompt_manager.make_graph_baseline_prompts(instruction, img_to_ghost_node_dict, self.gmaps[0].ghost_pos, stepk)
                # print('User:')
                # for content in messages[1]["content"]:
                #     print(content)

                if llm_results:
                    if self.config.GPT_VERSION == "gpt-4o-2024-05-13":
                        nav_output, tokens = gpt4v_infer(messages, model="gpt-4o-2024-05-13", response_format={"type": "json_object"})
                    else:
                        raise NotImplementedError

                    print('output\n', nav_output)
                    try:
                        if self.config.GPT_VERSION == "gpt-4o-2024-05-13":
                            json_output = json.loads(nav_output)
                        else:
                            raise NotImplementedError

                        try:
                            self.prompt_manager.planning.append(json_output["New Planning"])
                        except:
                            self.prompt_manager.planning.append("No planning in last step.")

                    except:
                        json_output = {"Thought": "format error", "Action": 'Stop'}
                        print('##############\nformat error\n##############')

                else:
                    print('llm_results', llm_results)
                    json_output = {"Thought": "no valid waypoints or path predictions in all directions",
                                   "Action": 'Stop'}

                with open(agent_json_output, 'w') as file:
                    json.dump(json_output, file)

            action = str(json_output["Action"])


            try:
                selected_path_id = self.prompt_manager.parse_num(action)
                selected_path_coord = waypoint_path_coord[selected_path_id]

            except:
                action = 'stop'
                print('###################\n GPT Errors in node prediction, stop \n#########################')

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):

                if action.lower() == 'stop':
                    env_actions.append(
                        {
                            'action': {
                                'act': "only_stop",
                            },
                            'vis_info': None,
                        }
                    )
                else:
                    if action in ghost_node_to_img_dict:
                        selected_img = ghost_node_to_img_dict[action]
                    else:
                        print("########### Errors in action prediction #################\n")
                        action = list(ghost_node_to_img_dict.keys())[0]
                        selected_img = ghost_node_to_img_dict[action]

                    direction = self.prompt_manager.convert_img_name_to_direction(selected_img)
                    self.prompt_manager.make_graph_history(action, stepk, direction=direction, selected_img=selected_img)

                    ghost_vp = f'g{action}'
                    if ghost_vp in gmap.ghost_aug_pos.keys():
                        ghost_pos = gmap.ghost_aug_pos[ghost_vp]  # only augment in training
                    else:
                        print("########### Errors in ghost ID prediction #################\n")
                        ghost_vp = list(gmap.ghost_aug_pos.keys())[0]
                        ghost_pos = gmap.ghost_aug_pos[ghost_vp]  # only augment in training

                    if self.config.VIDEO_OPTION:
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                        }
                    else:
                        vis_info = None


                    if stepk == self.max_len - 1:
                        # move and stop
                        env_actions.append(
                            {
                                'action': {
                                    'act': 0,
                                    'back_path': [],
                                    'tryout': use_tryout,
                                    'local_path_planning': selected_path_coord
                                },
                                'vis_info': vis_info,
                            }
                        )
                    else:
                        env_actions.append(
                            {
                                'action': {
                                    'act': 'local_path_planning',
                                    'tryout': use_tryout,
                                    'local_path_planning': selected_path_coord
                                },
                                'vis_info': vis_info,
                            }
                        )

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            ob = observations[0]
