#!/bin/bash

set -x

HCC_PATH=/opt/rocm/hcc
HCC=$HCC_PATH/bin/hcc
HCC_RUNTIME_PATH=$HCC_PATH/lib/

HIP_PATH=/opt/rocm/hip
HIP=$HIP_PATH/bin/hipcc
HIP_RUNTIME_PATH=$HIP_PATH/lib/

rm early_finalize_1_hip.obj1.o  early_finalize_1_hip.obj2.o  early_finalize_1_hip.obj3.o  early_finalize_1_hip early_finalize_1_hip.non_ef

$HIP -D_OBJ_1 early_finalize_1_hip.cpp -c  -o early_finalize_1_hip.obj1.o -fno-gpu-rdc -fPIC
$HIP -D_OBJ_2 early_finalize_1_hip.cpp -c  -o early_finalize_1_hip.obj2.o -fno-gpu-rdc -fPIC
$HIP -D_OBJ_3 early_finalize_1_hip.cpp -c  -o early_finalize_1_hip.obj3.o -fno-gpu-rdc -fPIC

g++ early_finalize_1_hip.obj1.o  early_finalize_1_hip.obj2.o  early_finalize_1_hip.obj3.o -o early_finalize_1_hip -L$HCC_RUNTIME_PATH -Wl,-rpath=$HCC_RUNTIME_PATH -lmcwamp -L$HIP_RUNTIME_PATH -Wl,-rpath=$HIP_RUNTIME_PATH -lhip_hcc

#$HIP  -D_OBJ_1 -D_OBJ_2 -D_OBJ_3 early_finalize_1_hip.cpp  -o  early_finalize_1_hip.non_ef


