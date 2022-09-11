# -*- coding: utf-8 -*-
# @Time : 2022/8/16 20:50
# @Author : LuoJiahuan
# @Email : luojiahuan001@gmail.com
# @File : pipeline.py
import time
import os
import logging

from multiprocessing import Queue, Process

import torch.cuda

os.makedirs("logs", exist_ok=True)
logging.basicConfig()
logger = logging.getLogger('pipeline')
fh = logging.FileHandler('pipeline.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)
#####
num_gpus = torch.cuda.device_count()


class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def get_current_time(self):
        return (time.time() - self.start_time) / 3600


class GPUManager(object):
    def __init__(self, num_gpus=4):
        self.gpu_queue = Queue()
        for device_id in range(num_gpus):
            self.gpu_queue.put(device_id)

    def require(self):
        try:
            return self.gpu_queue.get()
        except:
            return None

    def add_gpu(self, gpu_id):
        self.gpu_queue.put(gpu_id)


timer = Timer()
gpu_manager = GPUManager(num_gpus=num_gpus)


def run_gpu_model(cmd, log_file=None):
    if log_file:
        cmd = f"nohup {cmd} > {log_file}"
    while True:
        gpu_id = gpu_manager.require()
        if gpu_id is not None:
            try:
                run_cmd = f"export CUDA_VISIBLE_DEVICES={gpu_id} && {cmd}"
                logger.info(f"{run_cmd} 开始时间: {timer.get_current_time()}")
                os.system(run_cmd)
                logger.info(f"{run_cmd} 结束时间: {timer.get_current_time()}")
            except:
                logger.warning(f'{cmd} failed')
            gpu_manager.add_gpu(gpu_id)
            break


def train_model(cmd):
    # 预训练
    logger.info(f"训练 {cmd} 开始: {timer.get_current_time()}")
    run_gpu_model(f'{cmd}')
    logger.info(f"训练 {cmd} 结束: {timer.get_current_time()}")

def generate_task():
    task_list = []
    base_cmd = 'python federatedscope/main.py --cfg federatedscope/gfl/baseline/grid_search_gin_minibatch.yaml ' \
               '--client_cfg federatedscope/gfl/baseline/grid_search_per_client.yaml '
    for batch_size in [64, 128, 256, 512]:
        for layer in [2, 3, 4]:
            for hidden in [64, 128, 256, 512]:
                for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
                    expand_args = f"train.optimizer.lr {lr} model.layer {layer} " \
                                  f"model.hidden {hidden} data.batch_size {batch_size}"
                    cmd = base_cmd + expand_args
                    task_list.append(cmd)
    return task_list

def generate_task2():
    task_list = []
    base_cmd = 'python federatedscope/main.py --cfg federatedscope/gfl/ricky/a.yaml ' \
               '--client_cfg federatedscope/gfl/ricky/c.yaml '
    for augment in ['','NodeSam','MotifSwap','DropEdge', 'DropNode', 'ChangeAttr', 'AddEdge', 'NodeAug']:
        for use_aug_val_in_training_set in [False,True]:
            for jk_mode in ['cat','last']:
                expand_args = f"model.jk_mode {jk_mode} data.augment {augment}" \
                                f"data.use_aug_val_in_training_set {use_aug_val_in_training_set}"
                cmd = base_cmd + expand_args
                task_list.append(cmd)
    return task_list


def train():
    # 预训练
    model_processes = []
    tasks = generate_task2()
    for cmd in tasks:
        p = Process(target=train_model, args=(cmd,))
        p.start()
        time.sleep(60)
        logger.info(f'模型训练 {cmd}...')
        model_processes.append(p)
    for p in model_processes:
        p.join()
    logger.info(f'pretrain 结束: {timer.get_current_time()} 小时')


if __name__ == '__main__':
    train()
