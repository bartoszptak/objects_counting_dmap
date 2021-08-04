from __future__ import print_function

import os
import cv2
import argparse

import tensorrt as trt


MAX_BATCH_SIZE = 1


def load_onnx(onnx_path):
    """Read the ONNX file."""
    if not os.path.isfile(onnx_path):
        print('ERROR: file (%s) not found!' % onnx_path)
        return None
    else:
        with open(onnx_path, 'rb') as f:
            return f.read()


def set_net_batch(network, batch_size):
    """Set network input batch size.

    The ONNX file might have been generated with a different batch size,
    say, 64.
    """
    if trt.__version__[0] >= '7':
        shape = list(network.get_input(0).shape)
        shape[0] = batch_size
        network.get_input(0).shape = shape
    return network


def build_engine(onnx_model_path, channels, net_h, net_w, img_dir, quant_mode, dla_core, verbose=False):
    """Build a TensorRT engine from ONNX using the older API."""

    print('Loading the ONNX file...')
    onnx_data = load_onnx(onnx_model_path)
    if onnx_data is None:
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    EXPLICIT_BATCH = [] if trt.__version__[0] < '7' else [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if quant_mode == 'int8' and not builder.platform_has_fast_int8:
            raise RuntimeError('INT8 not supported on this platform')
        if not parser.parse(onnx_data):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

        network = set_net_batch(network, MAX_BATCH_SIZE)

        print('Building an engine.  This would take a while...')
        print('(Use "--verbose" or "-v" to enable verbose logging.)')
        
        builder.max_batch_size = MAX_BATCH_SIZE
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            'input',                          # input tensor name
            (MAX_BATCH_SIZE, channels, net_h, net_w),  # min shape
            (MAX_BATCH_SIZE, channels, net_h, net_w),  # opt shape
            (MAX_BATCH_SIZE, channels, net_h, net_w))  # max shape
        config.add_optimization_profile(profile)
        if quant_mode == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif quant_mode == 'int8':
            config.set_flag(trt.BuilderFlag.FP16)
            from crowd_trt_calibrator import CrowdEntropyCalibrator
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = CrowdEntropyCalibrator(
                img_dir, channels, (net_h, net_w), onnx_model_path.replace('.onnx', '_calib.bin'))
            config.set_calibration_profile(profile)
        if dla_core >= 0:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = dla_core
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            print('Using DLA core %d.' % dla_core)
        engine = builder.build_engine(network, config)

        if engine is not None:
            print('Completed creating engine.')
        return engine


def main():
    """Create a TensorRT engine for ONNX-based YOLO."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('Path to onnx model'))
    parser.add_argument(
        '-c', '--channels', type=int, default=5,
        help='Number of channels; 3: RGB input; 5: RGB + optical flow')
    parser.add_argument(
        '--net_h', type=int, default=608,
        help='Network input height')
    parser.add_argument(
        '--net_w', type=int, default=608,
        help='Network input width')
    parser.add_argument(
        '-i', '--img_dir', type=str,
        help='path to images directory')
    parser.add_argument(
        '-q', '--quant_mode', type=str, default='fp32',
        help='quantization mode, available: {fp32, fp16, int8}')
    parser.add_argument(
        '--dla_core', type=int, default=-1,
        help='id of DLA core for inference (0 ~ N-1)')
    parser.add_argument(
        '-o', '--output', type=str, default='./crowd_int8.trt',
        help='TensorRT output model path')
    args = parser.parse_args()

    engine = build_engine(
        args.model, args.channels, args.net_h, args.net_w, args.img_dir, args.quant_mode, args.dla_core, args.verbose)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    with open(args.output, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % args.output)


if __name__ == '__main__':
    main()
