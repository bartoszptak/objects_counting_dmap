import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from typing import List

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class ModelTRT:
    def __init__(self, config: dict):
        self.config = config

        self.TRT_LOGGER = trt.Logger()
        self.engine = self.get_engine('engine.trt')
        self.context = self.engine.create_execution_context()
        #self.buffers = self.allocate_buffers(self.engine, self.batch_size)

    def get_engine(self, engine_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_path))
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    @staticmethod
    def GiB(val):
        return val * 1 << 30

    @staticmethod
    def allocate_buffers(engine, batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in engine:

            size = trt.volume(engine.get_binding_shape(binding))
            dims = engine.get_binding_shape(binding)
            
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    @staticmethod
    def do_inference(context, bindings, inputs, outputs, stream, batch_size):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def predict(self, imgs: np.ndarray) -> np.ndarray:
        """Inference bbox on image.

        Parameters
        ----------
        img : np.ndarray
            Array shape must be (height, width, channels)

        Returns
        -------
        np.ndarray
            Returns array of list: [class, x, y, width, height]
        """
             
        imgs = np.ascontiguousarray([imgs], dtype=np.float32)

        inputs, outputs, bindings, stream = self.allocate_buffers(self.engine, 1)
        
        inputs[0].host = imgs

        outputs = self.do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
    
        return outputs[0].reshape(1,608,608,1)


if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    img = np.load('test_2.npy', mmap_mode='r', allow_pickle=True).transpose((2,0,1))

    out = ModelTRT(None).predict(img)
    print(out.shape)
    plt.imshow(out[0,:,:,0])
    plt.show()
