# CVE-2021-1056
![LICENSE](https://img.shields.io/github/license/pokerfaceSad/CVE-2021-1056) 

[CVE-2021-1056](https://ubuntu.com/security/CVE-2021-1056) is a vulnerability I submitted to NVIDIA PSIRT. Personally, it may lead to high security risks in multi-tenant HPC clusters, especially in cloud machine-learning platforms.

This repository simply demonstrates the vulnerability on GPU containers created by [`nvidia-container-runtime`](https://github.com/NVIDIA/nvidia-container-runtime).



## How it works

By creating specific character device files an attacker in a GPU container(container created by `nvidia-container-runtime`) is able to get access to all GPU devices on the host. 

It also works on GPU pod created by `k8s-device-plugin` on kubernetes cluster.



## Prerequisite

* Docker 19.03
* `nvidia-container-toolkit`

* NVIDIA Driver 418.87.01 / 450.51.05
* NVIDIA GPU Tesla V100 / TITAN V / Tesla K80

NOTE: only a few test environments included, but refer to [NVIDIA Security Bulletin](https://nvidia.custhelp.com/app/answers/detail/a_id/5142),  this vulnerability works on all GeForce, NVIDIA RTX/Quadro, NVS and Tesla series GPU, and all version drivers.



## QuickStart

See [QuickStart.md](docs/QuickStart.md)



## How to prevent

Recommended

* Refer to the [NVIDIA Security Bulletin](https://nvidia.custhelp.com/app/answers/detail/a_id/5142) or  to update the NVIDIA GPU driver

Or

* Add arg `--cap-drop MKNOD` to the  `docker run` to forbid the `mknod` in containers
* Enable `security context`  in kubernetes clusters when creating a pod



## License

This project is licensed under the MIT License.



## Issues and Contributing

Feel free to submit [Issues](https://github.com/pokerfaceSad/CVE-2021-1056/issues/new) and [Pull Requests](https://github.com/pokerfaceSad/CVE-2021-1056/pulls) if you have any problems.

