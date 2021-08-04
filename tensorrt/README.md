# TensorRT
Steps for converting, building, and evaluating TensorRT crowd counting models.

## PyTorch --> ONNX conversion

PyTorch library contains embedded package for exporting model to ONNX representation. To convert trained crowd models one can use [`pytorch2onnx.py`](./pytorch2onnx.py) converter script as follow:

```bash
python3 pytorch2onnx.py --checkpoint_path <PATH TO PyTorch CHECKPOINT> --model_architecture <UNet or UNet++> --encoder <ONE OF AVAILABLE ENCODERS> --in_channels <NUMBER OF INPUT CHANNELS> --input_size <SIZE OF INPUT IMAGE>
```

## TensorRT engine

### Calibration dataset (only for int8 mode)

Building TensorRT model in `int8` mode requires usage of [`crowd_trt_calibrator.py`](./crowd_trt_calibrator.py) script which prepare a cache file ensuring the best quantization mapping between FLOAT32 and INT8 data type. The original code of calibrator could be found in TensorRT-7.x sample code: [`samples/python/int8_caffe_mnist/calibrator.py`](https://github.com/NVIDIA/TensorRT/blob/master/samples/python/int8_caffe_mnist/calibrator.py). The modifications allow to handle with 5-channels input (R,G,B,Fx,Fy) which contains 3-channels from RGB image and 2-channels optical flow. Due to usage 
of several sequences, corresponding optical flow data should be prepared before calibration. Structure of calibration data directory:

```bash
.
├── imgs
└── imgs_flows
```

In order to prepare data one can use [`prepare_calibration_data.py`](./prepare_calibration_data.py) script with VisDrone Crowd Counting dataset as shown below:

```bash
python3 prepare_data.py --sequences_path <PATH TO VisDrone CC SEQUENCES>
```

### ONNX --> TensorRT conversion

For TensorRT engine building [`crowd_trt_converter.py`](./crowd_trt_converter.py) script was used. It allows to convert from ONNX model to TensorRT representation in one of three quantization modes (`fp32`, `fp16`, `int8`). Building TensorRT model in `int8` mode requires a path to calibration dataset (`--img_dir` parameter).

```bash
python3 crowd_trt_converter.py -v --quant_mode <QUANTIZATION MODE> --img_dir <PATH TO INT8 QUANTIZATION IMG DIR> --model <ONNX MODEL PATH> --output <TensorRT OUTPUT MODEL PATH>
```

## Evaluation

The benchmark was performed on [NVIDIA Jetson Xavier NX](https://www.nvidia.com/pl-pl/autonomous-machines/embedded-systems/jetson-xavier-nx/) with JetPack 4.4.1. The system was set to second power mode with command `sudo nvpmodel -m 2`, which configurate power budget to 15 W and set maximum freguency of GPU to 1100 MHz and enables 6 CPU cores with 1400 MHz. Moreover, the maximum frequency of fan, clocks and memory was ensured with command `sudo jetson clocks --fan`.

For evaluation `eval.py` script was used, the command line call is showed below.

```bash
python3 eval.py --model_path <MODEL PATH> --flow_type <FLOW TYPE>
```

## Benchmark results

