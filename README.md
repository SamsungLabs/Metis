# Metis: Fast Automatic Distributed Training on Heterogeneous GPUs

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://i.creativecommons.org/l/by-nc/4.0/88x31.png


Metis is a system that automatically finds efficient parallelism plans for distributed deep learning training on heterogeneous GPUs. 
The auto-planner component of Metis is publicly available now. Please see the paper for further details ([paper](https://www.usenix.org/conference/atc24/presentation/um))



## Install
To run this project, you need to install the required packages. Follow the steps below to install the dependencies using the 'requirements.txt' file.

1. Clone the repository: 
```bash
git clone https://github.com/SamsungLabs/Metis.git
```

2. Navigate to the project directory:
```bash 
cd ~/Metis
```

3. Install dependencies using the requirements.txt file: 
```bash
pip install -r requirements.txt
```

4. Once all dependencies are installed, you are ready to run the project.


#### Supported Python Versions
- 3.9

## Profile Data Guide

### Overview
The project relies on profile data to make informed decisions about distributed learning strategies. The profile data must be collected for different combinations of device types, tensor parallelism degrees, and batch sizes.

#### Naming Convention
The profile data files must be named according to the following pattern:
```
DeviceType.{}_tp{}_bs{}.json 
```
Where: 
- {}: The placeholders for specific values. 
  - First placeholder: The identifier for the specific device(e.g. A100, H100)
  - Second placeholder: The tensor parallel degree. 
  - Third placeholder: The batch size. 

#### Example: 
```
DeviceType.A100_tp1_bs1.json
DeviceType.A100_tp1_bs2.json
DeviceType.A100_tp1_bs4.json
... 
DeviceType.H100_tp4_bs8.json 
```

### Profile Data Structure
Each profile data file is a **JSON** file containing the following key sections: 
1. **Model Information**:
   - model_name: The name of the model(e.g. "BERT-Large", "ResNet-50")
   - num_layers: the total number of layers in the model(e.g. 24)
   - parameters: Information about the model's parameters.
     - total_parameters_bytes: The total bytes of parameters in the model(in bytes)
     - parameters_per_layer_bytes: An array representing the bytes of parameters per layer(in bytes) 
     - activation_paramters_bytes: An array representing the bytes of activation parameters after each layer(in bytes)
2. **Performance Data**:
   - layers: a list of layers in the model
     - For each layer:
       - layer_compute_total_ms: An array of total computation times for each layer, calculated as the sum of forward compute and backward_compute(in milliseconds, listed in order of layers) 
       - layer_memory_total_mb: An array of the total memory used by each layer, including activations, gradients and optimizer states(in megabytes, listed in order of layers)
       
3. **Overall Metrics**:
    - total_time_ms: The total time taken for 1 iteration.(in milliseconds)
    - forward_backward_time_ms: The time taken for the forward and backward passes of the all layers. This includes the time between the completion of the forward pass of the last layer and the start of the backward pass(in milliseconds)
    - batch_generator_time_ms: The time taken to generate the batch data during 1 iteration(in milliseconds)
    - layernorm_grads_all_reduce_time_ms: The time taken to reduce LayerNorm gradients across devices during 1 iteration(in milliseconds)
    - embedding_grads_all_reduce_time_ms The time taken to reduce embedding gradients across devices during 1 iteration(in milliseconds)
    - optimizer_time_ms: the time consumed by the optimizer(e.g. for applying Adam updates)
    - total_memory_mb: The total amount of memory required to run the model with the given conditions(tp degree, batch size) during training(in megabytes)
**Note**: All metrics are measured during a single iteration, where the micro batch size, batch size and global batch size are set to the same value. Therefore, one iteration can be considered equivalent to one epoch.
    
#### Example Profile Data
```json
{
  "model": {
    "model_name": "GPT3",
    "num_layers": 10,
    "parameters": {
      "total_parameters_bytes": 601952256,
      "parameters_per_layer_bytes": [98566144, 50601984, 50601984, 50601984, 50601984, 50601984, 50601984, 50601984, 50601984, 98570240],
      "activation_parameters_bytes": [98566144, 50601984, 50601984, 50601984, 50601984, 50601984, 50601984, 50601984, 50601984, 98570240]
      
    }
  },
  "execution_time": {
    "total_time_ms": 1137.5594139099121,
    "batch_generator_time_ms": 934.1955184936523,
    "layernorm_grads_all_reduce_time_ms": 459.5518112182617,
    "embedding_grads_all_reduce_time_ms": 37.360191345214844,
    "optimizer_time_ms": 10814.285278320312,
    "layer_compute_total_ms": [1.4263919830322266, 10.216951370239258, 10.216951370239258, 10.216951370239258, 10.216951370239258, 10.216951370239258, 10.216951370239258, 10.216951370239258, 10.216951370239258, 0.3376007080078125]
  },
  "execution_memory": {
    "total_memory_mb": 15150.69,
    "layer_memory_total_mb": [2366.8, 1195.9, 1195.9, 1195.9, 1195.9, 1195.9, 1195.9, 1195.9, 1195.9, 3216.7]
  }
}

```

### Data Collection Guidelines
- Profile data must be collected for each combination of device type, tensor parallel size, and batch size that you intend to optimize.
- Ensure consistency in the device identifiers and tensor parallel settings for reproducibility.
- Collect data for a representative set of model layers to capture accurate performance metrics. 

### File Structures
organize the profile data files in a structured directory for easy access: 
```
/profile_data
  ├── DeviceType.A100_tp1_bs1.json
  ├── DeviceType.A100_tp1_bs2.json
  ├── DeviceType.A100_tp1_bs4.json
  ├── DeviceType.A100_tp1_bs8.json
  ├── DeviceType.A100_tp1_bs16.json
  ├── DeviceType.A100_tp2_bs1.json
  ├── DeviceType.A100_tp2_bs2.json
  ├── DeviceType.A100_tp2_bs4.json
  ...
  ├── DeviceType.H100_tp4_bs4.json
  ├── DeviceType.H100_tp4_bs8.json
  └── DeviceType.H100_tp4_bs16.json
```

### How to Use the Profile Data
Once you have collected the necessary profile data, the optimizer will use these files to calculate the optimal distributed learning strategy for your model. Ensure that all relevant configurations are covered, as missing data may result in suboptimal strategy suggestions. 
By following this guide, you ensure the profile data is correctly formatted and useful for optimizing distributed learning strategies across different hardware and parallelism settings. 

## Getting Profile Data
This section explains how to collect model profile data necessary for finding the optimal distributed training strategy. It provides methods for measuring the model's **execution time** and **memory usage**, which are crucial optimizing distributed training performance.
This guide explains how to collect model profile data using PyTorch's Hook functions, memory measurement techniques and Megatron's Timer module. By collecting accurate execution time and memory usage data, you can find the optimal distributed training strategy.
**Note**: For more details on PyTorch hooks and memory measurement, refer to the [official PyTorch documentation](https://pytorch.org/docs/stable/index.html)

### 1. Required Profile Data
The essential profile data for optimizing distributed training are as follows: 
- **Execution Time**: The time taken for the forward and backward passes of each layer. 
- **Memory Usage**: The GPU memory used during the execution of each layer. 

### 2. Measuring Execution Time
To measure the execution time of each layer, we use **PyTorch's Hook functions.** PyTorch provides register_forward_pre_hook, register_forward_hook, register_backward_pre_hook and register_backward_hook to register custom actions at the start and end of the forward and backward passes of each layer.

#### Implementation Steps
1. Register hooks for each layer to record the start and end times of the forward and backward passes. 
2. Calculate the time difference between the start and end times to measure the execution time of each layer. 
3. For accurate measurement, call **torch.cuda.synchronize()** before recording the time. This ensures that any asynchronous GPU operations are completed, reducing timing discrepancies.
**Note**: For more details on how to add hooks, refer to the [PyTorch official documentation.](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)

#### Tips for Accurate Performance Measurement
- **warm-up**: To ensure accurate measurements, it is recommended to **warm up the model by running it several times** before starting the actual time measurements. This helps account for factors like GPU caching, which can cause irregular time results during the initial runs. 

### 3. Measuring Memory Usage 
Memory usage can be measured using torch.cuda.max_memory_reserved, which returns the maximum amount of memory allocated on the GPU. This value helps track the peak memory usage during the execution of each layer. 
 
#### Implementation Steps
1. After each layer's execution, call **torch.cuda.max_memory_reserved()** to record the maximum memory used during that layer. 
2. For more accurate memory measurements, allocate **one layer per device**, allowing independent measurement of each layer's memory usage. 

#### Specifics of Memory Measurement
- **Device Allocation**: If the same layer is repeated multiple times, it's possible to assign multiple layers to the same device and measure the total memory usage. Ideally, having sufficient GPU resources allocate each layer to a separate device will provide more precise measurements, but in cases where resources are limited, measuring multiple layers on one device is still a valid approach.

### 4. Measuring Key Metrics
This project used Megatron's Timer module to collect key metrics. The Timer module precisely measures the time spent in different stages of the training process, especially during parameter updates. 

#### Key Metrics
- **forward_backward_time**
- **batch_generator_time** 
- **grads_all_reduce_time** 
- **optimizer_time**

#### Using the Timer Module
1. For each key metric, use **Megatron's Timer** to measure that time taken for specific stages of the training process. 
2. For instance, measure the **time before and after the optimizer functions** to record the time spent updating the model's parameters. 
3. The recorded time data allows for detailed analysis of time consumption at each stage of training, providing insights for performance optimization.

## How to Use Metis
Metis is a project that finds the optimal distributed training strategy based on the given cluster environment and profile dta. Users must first configure the resource and environment information for each node in the cluster, and then execute the script to optimize the training strategy. Below are the instructions for preparing the necessary data and running the script. 

### 1. Pre-requisite Data 
Before running Metis, you need to prepare the following data: 

- host_name: This contains the IP addresses and the number of GPUs for each node in the cluster where the training will be performed. It defines the cluster environment to be optimized.
```
IP1 8 
IP2 8
IP3 8
IP4 8 
```
- clusterfile: This contains the environment information for each node, such as the device type, bandwidth and memory capacity.

```json
{
  "IP1": {
    "instance_type": "V100",
    "inter_bandwidth": 312500000.0,
    "intra_bandwidth": 5312500000.0,
    "memory": 16
  },
  "IP2": {
    "instance_type": "P100",
    "inter_bandwidth": 312500000.0,
    "intra_bandwidth": 5312500000.0,
    "memory": 15
  },
  "IP3": {
    "instance_type": "T4",
    "inter_bandwidth": 312500000.0,
    "intra_bandwidth": 5312500000.0,
    "memory": 15
  },
  "IP4": {
    "instance_type": "A100",
    "inter_bandwidth": 312500000.0,
    "intra_bandwidth": 5312500000.0,
    "memory": 80
  }
}
```

- profile_data: This data consists of pre-collected profile information based on the device type, tensor parallel degree and batch size. Each profile data file is used to determine the optimal distributed training strategy based on the performance characteristics of the device. For more details about the profile data structure, refer to the previous sections.

### 2. How to Run the Script
After preparing the necessary data, you can run Metis's main script to start distributed training. 

**Main Script: cost_het_cluster.sh**

Below are the main parameters and their descriptions: 

- MODEL_NAME
- MODEL_SIZE
- NUM_LAYERS 
- GBS(Global Batch Size)
- HOME_DIR: The base directory path where project files and results are saved. All script-related operations taken place within this directory. 
- MAX_PROFILED_TP: the maximum tensor parallel degree recorded during profiling.
- MAX_PROFILED_BATCH_SIZE: the maximum batch size recorded during profiling. 
- SCALE_VARIANCE: A value used to adjust the size of the device_group when constructing the search space.  
- MAX_PERMUTE_LEN: A value that limits the size of the device_group, helping to create an efficient search space. 

  
### 3. Example Execution 
Once all the data is prepared, you can execute the script with the following command: 
```bash 
cd ~/Metis/scripts
source ./cost_het_cluster.sh MODEL_NAME=GPT MODEL_SIZE=1.5B NUM_LAYERS=10 GBS=128 HOME_DIR='/home/user' MAX_PROFILED_TP=4 MAX_PROFILED_BATCH_SIZE=4 SCALE_VARIANCE=1 MAX_PERMUTE_LEN=4 
```

This command will explore and execute the optimal distributed training strategy based on the pre-configured node and device information and profile data.





---------------


[![CC BY NC 4.0][cc-by-nc-image]][cc-by-nc]
This work is licensed under a
[Creative Commons Attribution Non Commercial 4.0 International License][cc-by-nc].

