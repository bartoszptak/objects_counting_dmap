# OpenVINO

Steps for converting, and evaluating OpenVINO crowd counting models.

## PyTorch --> ONNX conversion

PyTorch library contains embedded package for exporting model to ONNX representation. To convert trained crowd models one can use [`pytorch2onnx.py`](./pytorch2onnx.py) converter script, which is in tensorrt directory, as follow:

```bash
python3 pytorch2onnx.py --checkpoint_path <PATH TO PyTorch CHECKPOINT> --model_architecture <UNet or UNet++> --encoder <ONE OF AVAILABLE ENCODERS> --in_channels <NUMBER OF INPUT CHANNELS> --input_size <SIZE OF INPUT IMAGE>
```

## ONNX --> Intermediate Representation (IR) conversion

To change format from ONNX to Intermediate Representation (IR) [OpenVINO toolkit in version 2021.4](https://docs.openvinotoolkit.org/latest/index.html) was used. This framework allows to convert and optimize neural network models for inferencing on Intel hardware such as CPU, GPU, VPU (MYRIAD), and FPGA. To convert models toolkit should be installed and Model Optimizer package requirements should be met.

```bash
python3 ~/intel/openvino_2021/deployment_tools/model_optimizer/mo.py --input_model <PATH TO ONNX MODEL> --model_name <OUTPUT MODEL NAME> --data_type <FP16 or FP32> --batch 1
```

## Evaluation

The benchmark was performed on [Intel Neural Compute Stick 2](https://software.intel.com/content/www/us/en/develop/hardware/neural-compute-stick.html) with use of [Raspberry Pi 4B](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/).

## Benchmark results
