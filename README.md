# AO-Planner EMNLP submission

实验参考：[https://www.kdocs.cn/l/cmfhFU3GoHi2](https://www.kdocs.cn/l/cmfhFU3GoHi2)

## Usage
main.bash 
- 使用ss_trainer_ETP.py，进行train val test

train.bash
- 仅训练的脚本

evaluate.bash   
- val unseen上，测试已经训练好的ETPNav

LLM_eval.bash
- 我自己实现的：Gemini for local planning and directly use ETPNav for global planning without finetuning
- should use "cond activate vlnce". 
- eval mode (on val unseen set) 
- 代表性实验：05281_traj5_waypoint3
- 性能不太行，agent无法适应不同风格的candidates
- 需要训练一个新的waypoint model，然后finetuning第二阶段的vln agent

LLM_eval_parallel.bash
- 尝试在一个bash脚本中同时启动Gemini & ETPNav，
- 仍有bug。后来放弃，转用两个分别启动

zero_shot_eval.bash
- TRAINER_NAME Zero-Shot, 使用的是zero_shot_agent.py
- 双阶段纯zero-shot实验
- 代表性实验：06062及其之前的一些实验，大概SR在27%
- 有尝试过top-down map，ablation of LLMs，self-refine，单/多起点
- 但是喂给second-stage agent还是第一阶段包含所有points的path可视化，而不是只提供预测的waypoint and path candidates

zero_shot_graph_eval.bash
- TRAINER_NAME Zero-Shot-Graph
    - 基于zero_shot_agent_graph.py
    - 用到了graph打ID的顺序来可视化waypoints and paths candidates
    - 代表实验：06070_GPT4o_ZS_graph_planning_semantic_multi_start_traj5_sample100。
    - 发现action space太大，且GPT倾向于选择ID小的。更像是宽度优先搜索。agent跳来跳去的
    - 采用duet的方式，直接选择global action space，而不是像MapGPT那种一步一步的选择
    - 不完善，但感觉还是有改进空间的
- TRAINER_NAME Zero-Shot-Graph-Baseline
    - 基于zero_shot_agent_graph_baseline.py，是zero_shot_agent_graph.py把global action space移除掉的baseline
    - 同样用到了graph打ID的顺序来可视化waypoints and paths candidates。**相比于之前的zero_shot_agent.py，没有多余的撒在地板上的点**
    - 代表实验：06072_GPT4o_ZS_ThreeeWaypoints_graph_baseline_semantic_multi_start_traj5_sample100，以及**最后full val unseen都是这个测的**
    - 对比性能，似乎是有些优势的：[https://kdocs.cn/l/cmfhFU3GoHi2?linkname=HzaAdF43TM](https://kdocs.cn/l/cmfhFU3GoHi2?linkname=HzaAdF43TM)


这里记得要切换, 上面的要使用py37环境，下面的使用vlnce(python 3.6)环境
```
from vlnce_baselines import ss_trainer_ETP, dagger_trainer, local_llm_agent, zero_shot_agent, zero_shot_agent_graph, zero_shot_agent_graph_baseline
# from vlnce_baselines import ss_trainer_ETP, dagger_trainer