# hilotel
`git clone https://github.com/AmanuelErgogo/hilotel.git`

`cd hilotel`

`python3 execution_time_result.py`

##### Inspect datatset

`python3 tools/inspect_demonstrations.py logs/robomimic/XArm7Env-v1/hilo_exp_pick_and_place.hdf5`

##### train

`python3 train.py --task XArm7Env-v1 --dataset logs/robomimic/XArm7Env-v1/hilo_exp_pick_and_place.hdf5 --algo bcq --name hilotel_exp_pick_and_place_bcq`

##### evaluate

`python3 play_offline_policy.py --task XArm7Env-v1 --checkpoint logs/robomimic/XArm7Env-v1/hilotel_exp_pick_and_place_bc_side_init_no_prev_act/20231212183154/models/model_epoch_350.pth`

`python3 play_hilotel.py --task XArm7Env-v1 --checkpoint logs/robomimic/XArm7Env-v1/hilotel_exp_pick_and_place_bc_side_init_no_prev_act/20231212183154/models/model_epoch_350.pth`
