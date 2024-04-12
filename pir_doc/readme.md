## 编译日志

<!-- https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile-by-make.html#jianchanindejisuanjihecaozuoxitongshifoufuhewomenzhichidebianyibiaozhun -->
<!-- 文档中 对于需要编译GPU 版本 PaddlePaddle的用户：(** CUDA11.0 - CUDA12.0 **), 这句话应该理解为需要编译多GPU,单机感觉没必要装nccl2-->


<!-- cmake比较智障，认不到指定nvcc -->
<!-- cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc .. -->
time cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc -DPY_VERSION=3.8 -DWITH_GPU=ON -DON_INFER=ON -DWITH_NVCC_LAZY=ON

<!-- jinja2如果报错的话需要更新一下,3.1.3这个版本过了,以及需要解决一些依赖问题(如果是新开的虚拟环境) -->
<!-- pip3 install --upgrade Jinja2 -->
<!-- sudo apt-get install -y patchelf -->

<!-- $(nproc)的输出是12，i512400+3060+32G内存的wsl的结果是 -j2才能过，不然err137资源不足....应该是爆内存了 -->
<!-- 看了一下好像是在编flashattn这种kernel的时候爆掉的，具体表现是VmmemWSL这个进程直接占用超过18G内存，后面开了j8暂时只有12G -->
time make -j$(nproc)


<!-- py包 -->
source activate paddle
pip install python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl

<!-- 测试 -->
<!--error adding symbols：DSO missing from command line https://blog.csdn.net/weixin_44251398/article/details/131970626 -->
<!--加上 -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries"  -->

<!--这个PR需要合入，遇到了里面的问题：https://github.com/PaddlePaddle/Paddle/issues/61311
https://github.com/PaddlePaddle/Paddle/pull/62497 -->
time cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc -DPY_VERSION=3.8 -DWITH_GPU=ON -DON_INFER=ON -DWITH_NVCC_LAZY=ON -DWITH_TESTING=ON  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries"
time make -j8


export FLAGS_PIR_OPTEST=true
export FLAGS_PIR_OPTEST_WHITE_LIST=true
export FLAGS_enable_pir_in_executor=true

export FLAGS_PIR_OPTEST=false
export FLAGS_PIR_OPTEST_WHITE_LIST=false
export FLAGS_enable_pir_in_executor=false
export GLOG_v=8
export GLOG_logtostderr=1

cd build
time cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc -DPY_VERSION=3.8 -DWITH_GPU=ON -DON_INFER=ON -DWITH_NVCC_LAZY=ON -DWITH_TESTING=ON  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries" && make -j 8

pip install -U --force-reinstall /home/cmcandy/code/PD/Paddle/build/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
cd .. 
nohup python test/legacy_test/test_partial_sum_op.py > ../pir_doc/1-partial_sum/logs2.log 2>&1 &


cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc -DPY_VERSION=3.8 -DWITH_GPU=ON -DON_INFER=ON -DWITH_NVCC_LAZY=ON -DWITH_TESTING=ON  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries"
make -j 8 > ../../pir_doc/build.log 2>&1 &



(cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc -DPY_VERSION=3.8 -DWITH_GPU=ON -DON_INFER=ON -DWITH_NVCC_LAZY=ON -DWITH_TESTING=ON  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries" && make -j 8 && pip install -U --force-reinstall  /home/cmcandy/code/PD/Paddle/build/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl) > ../../Develop_Diary/pir_doc/3-fake_quantize/cmake_log.log 2>&1 &

(cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc -DPY_VERSION=3.8 -DWITH_GPU=ON -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries" && make -j 8 && pip install -U --force-reinstall  /home/cmcandy/code/PD/Paddle/build/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl) > ../../Develop_Diary/pir_doc/3-fake_quantize/cmake_log.log 2>&1 &


(make -j 12 && pip install -U --force-reinstall /home/cmcandy/code/PD/Paddle/build/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl ) > ../../Develop_Diary/pir_doc/3-fake_quantize/cmake_log.log 2>&1 &