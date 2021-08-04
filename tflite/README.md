# TensorFlow Lite

Steps for converting, building, and evaluating TensorRT crowd counting models.

## PyTorch --> ONNX conversion

PyTorch library contains embedded package for exporting model to ONNX representation. To convert trained crowd models one can use [`pytorch2onnx.py`](./pytorch2onnx.py) converter script, which is in tensorrt directory, as follow:

```bash
python3 pytorch2onnx.py --checkpoint_path <PATH TO PyTorch CHECKPOINT> --model_architecture <UNet or UNet++> --encoder <ONE OF AVAILABLE ENCODERS> --in_channels <NUMBER OF INPUT CHANNELS> --input_size <SIZE OF INPUT IMAGE>
```

## ONNX --> TensorFlow

According to [official ONNX implementation](https://github.com/onnx/onnx-tensorflow), ONNX ensures command line interface for conversion from Open Neural Network Exchange (ONNX) to TensorFlow format.

```bash
onnx-tf convert -i /path/to/input.onnx -o /path/to/output
```

## TensorFlow --> TensorFlow Lite

TensorFlow library supports lightweight subpackage for mobile and edge devices in form of TensorFlow Lite. In the moment of createing tflite models, only the conversion to FP32 and FP16 modes was available. In order to change model from TensorFlow SavedModel to TensorFlow Lite FlatBuffer format [`tf2tflite.py`](./tf2tflite.py)script was used.

```bash
python3 tf2tflite.py --model_path <PATH TO TF SavedModel> --quantization_mode <FP32 or FP16> --output_path <OUTPUT PATH FOR TFLITE MODEL>
```

## Evaluation


## Benchmark results
