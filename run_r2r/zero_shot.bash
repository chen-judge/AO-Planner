export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

split="val_unseen_sample100"
#split="val_unseen"
#data_path=data/logs/llm/0730_no_SAM
#data_path=data/logs/llm/release_test
data_path=data/logs/llm/06072_GPT4o_ZS_ThreeeWaypoints_graph_baseline_semantic_multi_start_traj5_sample100

flag2="--exp_name release_r2r
      --run-type eval
      --exp-config run_r2r/zero_shot_eval.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/0717_all_train_ckpt_RGBD/ckpt.iter12000.pth
      IL.back_algo control
      IL.max_traj_len 5
      LLM_DIR ${data_path}
      GPT_VERSION gpt-4o-2024-05-13
      TRAINER_NAME Zero-Shot-AO-Planner
      EVAL.SPLIT ${split}
      "

python run.py $flag2
