import click
import tensorflow as tf


@click.command()
@click.option('--model_path', help='Path to TensorFlow SavedModel', default='./visdrone_dis_unet_resnet18_adam_aug_mosaic_flow_dis/')
@click.option('--quantization_mode', help='Model quantization mode, available: {fp32, fp16}', default='fp32')
@click.option('--output_path', help='Path to output TF Lite model', default='./visdrone_dis_unet_resnet18_adam_aug_mosaic_flow_dis.tflite')
def convert(model_path, quantization_mode, output_path):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    if quantization_mode == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save the model.
    with open(output_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    convert()