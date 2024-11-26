@echo off

:: 设置环境变量，解决 OpenMP 冲突问题
set KMP_INIT_AT_FORK=FALSE
set KMP_WARNINGS=FALSE
set KMP_SETTINGS=FALSE
set KMP_DUPLICATE_LIB_OK=TRUE

:: 运行 Python 脚本
python visualizer_drag.py ^
    checkpoints/stylegan3-r-ffhqu-256x256.pkl
