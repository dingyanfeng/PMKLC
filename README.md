[![DOI](https://zenodo.org/badge/912676874.svg)](https://doi.org/10.5281/zenodo.15526366)
# PMKLC

### About The PMKLC

**PMKLC** is a novel Parallel Multi-Knowledge Learning-based Compressor, which is used to compress  genomics data without loss. PMKLC has two compression modes **PMKLC-S** and **PMKLC-M**, where the former runs on a resource-constrained single GPU and the latter is multi-GPU accelerated.

### Requirements

0. 1xGPU for PMKLC-S or 4xGPU for PMKLC-M
1. Python 3.11.9
2. nvcc 12.1

### Usage

#### 1.PMKLC-S

##### 1.1 compress with PMKLC-S

```shell
nvcc SKMER_S.cu cnpy.cpp -O2 -o SKMER_S -lz
# The above command is optional; 
# users can compile SKMER themselves or use pre compiled executable file SKMER_S;
# if run the above command, make sure zlib be installed.
bash PMKLC_M_Compression.sh [FILE TO COMPRESS] [GPU ID] [BATCH-SIZE] [k] [s] SPuM
# [GPU ID]: code running on this GPU
# [k]: window size of GskE, default 3
# [s]: step size of GskE, default 3
```

##### 1.2 decompress with PMKLC-S

```shell
bash PMKLC_S_Decompression.sh [COMPRESSED FILE] [GPU ID] [k] [s] SPuM
# [COMPRESSED FILE]: The output file of compressor.
# [k]: window size of GskE, default 3
# [s]: step size of GskE, default 3
```

#### 2.PMKLC-M

##### 2.1 compress with PMKLC-M

```shell
nvcc SKMER_M.cu cnpy.cpp -O2 -o SKMER_M -lz
# The above command is optional; 
# users can compile SKMER themselves or use pre compiled executable file SKMER_S;
# if run the above command, make sure zlib be installed.
bash PMKLC_M_Compression.sh [FILE TO COMPRESS] [GPU ID] [BATCH-SIZE] [k] [s] SPuM
# [GPU ID]: GskE/train-SPrM(if have) running on this GPU, compress running on GPU0-3
### Users can modify the GPU by modifying the script: PMKLC_M_Compression.sh
# [k]: window size of GskE, default 3
# [s]: step size of GskE, default 3
```

##### 2.2 decompress with PMKLC-M

```shell
bash PMKLC_S_Decompression.sh [COMPRESSED FILE] [GPU ID] [k] [s] SPuM
### Users can modify the GPU by modifying the script: PMKLC_M_Compression.sh
# [COMPRESSED FILE]: The output file of compressor.
# [k]: window size of GskE, default 3
# [s]: step size of GskE, default 3
```

### Examples

#### 1.PMKLC-S

```shell
# compress
bash PMKLC_S_Compression.sh /home/xxx/PMKLC/DataSets/ScPo 0 320 3 3 SPuM
# decompress
bash PMKLC_S_Decompression.sh ScPo_3_3.pmklc.combined 0 3 3 SPuM
```

#### 2.PMKLC-M

```shell
# compress
bash PMKLC_M_Compression.sh /home/xxx/PMKLC/DataSets/ScPo 0 320 3 3 SPuM
# decompress
bash PMKLC_M_Decompression.sh ScPo_3_3.pmklc.combined 0 3 3 SPuM
```

### Dataset and SPuM

| File     | Link                                                         |
| -------- | ------------------------------------------------------------ |
| DataSets | https://drive.google.com/file/d/1vUMHeSYQnbSMB571EgpviDzGv4QsKCOr/view?usp=sharing |
| SPuM     | https://drive.google.com/file/d/1XVQUs_DaHr5c6CDX6wP-0cdFceba5AJ9/view?usp=sharing |


### Our Experimental Configuration

The experiments were running on a Ubuntu server (20.04.6 LTS) equipped with:

4 * Intel Xeon 4310 CPUs (2.10 GHz);

4 * NVIDIA GeForce RTX 4090 GPUs (24 GB);

512 GB of DDR4 RAM

### Credits

The arithmetic coding is performed using the code available at [Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding). The code is a part of Project Nayuki.

The cnpy code(cnpy.h and cnpy.cpp) is performed using the code available at [cnpy](https://github.com/rogersce/cnpy)
