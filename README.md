# Scan-RL: Next-Best View Policy for 3D Reconstruction
This is the release code of Scan-RL presented in the paper Next-Best View Policy for 3D Reconstruction. The Houses3K dataset used in this paper can be found in this [link](https://github.com/darylperalta/Houses3K).

## Scan-RL
### Diagram
<img src='imgs/Diagram.png' width="500"/>

## Setting up the Environment
- Necessary Python packages can be found in cleaned_reqss.txt.
- To install the environments, you need to install this fork of [gym-unrealcv](https://github.com/darylperalta/gym-unrealcv). Additional instructions are included there.





- Unreal Environments can be found [here]( https://drive.google.com/drive/folders/12Mo7vrlws0mcU99q-U7CzECoLLO50iUh?usp=sharing). You can find the instructions in [gym-unrealcv](https://github.com/darylperalta/gym-unrealcv) on where to place these Unreal Environments.
- Comands to run the code can be found in commands.txt

### Single House Experiment

- [Weights](https://drive.google.com/drive/folders/1Rd7VJHZIQB3rn-XL35MR7E7UrK_3vlg_?usp=sharing)
- Circular baseline script can be found [here](https://github.com/darylperalta/gym-unrealcv/blob/v0.2/example/circular_agent/circular_agent_close_depth.py).
- Sample Usage

Training *(Discrete Action Space)*

```
python main_unreal.py --nb_episodes 500 --batch_size 32 --epsilon_decay 0.999 --epsilon 1.0 --save_interval 20 --consecutive_frames 3 --type DDQN --env DepthFusionBGray-v0
 ```

Training *(Continuous Action Space)*

```
python main_unreal.py --nb_episodes 500 --batch_size 32 --epsilon_decay 0.999 --epsilon 1.0 --save_interval 20 --consecutive_frames 6 --type DDPG --env DepthFusionBGrayContinuous-v0
```

Testing
```
python load_and_run_unreal.py --type DDQN --consecutive_frames 6 --model_path '/hdd/AIRSCAN/sfm_results/RL_VP/new_baselines/bat6/2dist_45az_3elev/models/DDQN_ENV_DepthFusionBGray-v0_NB_EP_1000_BS_32_LR_0.00025_ep_10000.h5' --epsilon 0.0
```

### Multiple Houses New Split
- [Weights](https://drive.google.com/drive/folders/1N5ixPHbSHh_SUTj3GMEFMRUHL4JQFKLu?usp=sharing)
- Circular baseline script can be found [here](https://github.com/darylperalta/gym-unrealcv/blob/v0.2/example/circular_agent/circular_agent_baseline.py).
- Sample Usage

Training

```
python main_unreal.py --nb_episodes 2500 --batch_size 10 --epsilon_decay 0.999 --epsilon 1.0 --save_interval 20 --consecutive_frames 6 --type DDQN --env DepthFusionBGrayMultHouseRand-v0
```

Testing

```
python main_unreal.py --nb_episodes 2500 --batch_size 10 --epsilon_decay 0.999 --epsilon 1.0 --save_interval 20 --consecutive_frames 6 --type DDQN --env DepthFusionBGrayMultHouseRand-v0

```

### Non-House Target Model (Stanford Bunny) Experiment
- [Weights](https://drive.google.com/drive/folders/1WnsleXGK0S0KcC0XMzYb6Ddw4LCBHPTb?usp=sharing)
- Sample Usage

```
python load_and_run_unreal.py --model_path '/home/daryl/gym-unrealcv/new_bunny_3pen_89cov/models/DDQN_ENV_Bunny-v0_NB_EP_1000_BS_32_LR_0.00025_ep_10000.h5' --consecutive_frames 6 --type DDQN --env Bunny-v0 --epsilon 0.0

```

## Citation




### Acknowledgements
The RL implementation was based on this [repo](https://github.com/germain-hug/Deep-RL-Keras).
Gym environments are based on UnrealCV and [gym-unrealcv](https://github.com/darylperalta/gym-unrealcv).
