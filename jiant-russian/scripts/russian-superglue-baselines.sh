#!/bin/bash
# Functions to run Russian-SuperGLUE BERT baselines.
# Usage: ./scripts/superglue-baselines.sh ${TASK} ${GPU_ID} ${SEED}
#   - TASK: one of {"danetqa", "rcb", "parus", "muserc", "rucos", "terra", "russe", "rwsd", "all"},

#   - GPU_ID: GPU to use, or -1 for CPU. Defaults to 0.
#   - SEED: random seed. Defaults to 111.

source user_config.sh
seed=${3:-111}
gpuid=${2:-0}

function danetqa() {
    python main.py --config jiant/config/superglue_bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = danetqa, pretrain_tasks = \"danetqa\", target_tasks = \"danetqa\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 1000"
}

function rcb() {
    python main.py --config jiant/config/superglue_bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = rcb, pretrain_tasks = \"rcb\", target_tasks = \"rcb\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 60"
}

function parus() {
    python main.py --config jiant/config/superglue_bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = parus, pretrain_tasks = \"parus\", target_tasks = \"parus\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 100"
}

function muserc() {
    python main.py --config jiant/config/superglue_bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = muserc, pretrain_tasks = \"muserc\", target_tasks = \"muserc\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 1000, val_data_limit = -1"
}

function rucos() {
    python main.py --config jiant/config/superglue_bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = rucos, pretrain_tasks = \"rucos\", target_tasks = \"rucos\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 8, val_interval = 10000, val_data_limit = -1"
}

function terra() {
    python main.py --config jiant/config/superglue_bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = terra, pretrain_tasks = \"terra\", target_tasks = \"terra,lidirus\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 625"
}

function russe() {
    python main.py --config jiant/config/superglue_bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = russe, pretrain_tasks = \"russe\", target_tasks = \"russe\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 1000"
}

function rwsd() {

    python main.py --config jiant/config/superglue_bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = rwsd, pretrain_tasks = \"rwsd\", target_tasks = \"rwsd\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 139, optimizer = adam"
}



if [ $1 == "danetqa" ]; then
    danetqa
elif [ $1 == "rcb" ]; then
    rcb
elif [ $1 == "parus" ]; then
    parus
elif [ $1 == "muserc" ]; then
    muserc
elif [ $1 == "rucos" ]; then
    rucos
elif [ $1 == "terra" ]; then
    terra
elif [ $1 == "russe" ]; then
    russe
elif [ $1 == "rwsd" ]; then
    rwsd
elif [ $1 == "all" ]; then
    rwsd
    russe
    terra
    rucos
    muserc
    parus
    rcb
    danetqa


fi
