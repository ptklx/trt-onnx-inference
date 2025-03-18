import tensorrt as trt
import numpy as np
# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
import time
# from PIL import Image
# from six import string_types
from typing import Optional, List, Union,Tuple
from collections import namedtuple
import torch
from pathlib import Path
# import torchvision.transforms as transforms
# import cv2
# from tokenizer import build_tokenizer
# tokenizer = build_tokenizer()


# 图像预处理函数，使用 OpenCV 加载图像
# def preprocess_input(img,text_queries):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式  
#     # 图像大小调整
#     img = cv2.resize(img, (768,768))
#     # 转换为浮动类型并归一化 (标准化到 [0, 1])
#     img = img.astype(np.float32) / 255.0
#     mean = np.array([0.485, 0.456, 0.406])  # ImageNet 的平均值
#     std = np.array([0.229, 0.224, 0.225])   # ImageNet 的标准差
#     img = (img - mean) / std
#     # 将图像转换为 (C, H, W) 格式
#     # img = np.transpose(img, (2, 0, 1))  # 变换维度顺序为 (channels, height, width)
#     pixel_values = np.expand_dims(img, axis=0).astype(np.float32)  # 增加一个 batch 维度，(1, C, H, W)
#     input_ids = np.array([tokenizer.encode(text_queries)  ]).reshape(-1)
#     input_ids = np.pad([49406,*input_ids,49407],(0,16-len(input_ids)-2))
#     input_ids =np.expand_dims(input_ids, axis=0)
#     attention_mask = (input_ids > 0).astype(np.int64)
#     return  input_ids, attention_mask,pixel_values


# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# def build_engine(max_batch_size, onnx_file_path="", engine_file_path="", \
#                  fp16_mode=False, int8_mode=False, save_engine=False):
#         """Takes an ONNX file and creates a TensorRT engine to run inference with"""
#         builder = trt.Builder(TRT_LOGGER)
#         network = builder.create_network(EXPLICIT_BATCH) 
#         parser = trt.OnnxParser(network, TRT_LOGGER)
        
#         config = builder.create_builder_config()
#         # "1. 设置builder一些属性"
#         # config.max_workspace_size = 1 << 30   # 7.x 版本
#         config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 GB
#         # builder.max_batch_size = max_batch_size
#         if int8_mode:
#             # To be updated
#             raise NotImplementedError
#         if fp16_mode:
#             config.set_flag(trt.BuilderFlag.FP16)
#         # "2. 解析模型"
#         print('Loading ONNX file from path {}...'.format(onnx_file_path))
#         """with open(onnx_file_path, 'rb') as model:
#             print('Beginning ONNX file parsing')
#             parser.parse(model.read())"""
#         with open(onnx_file_path, "rb") as model:
#             if not parser.parse(model.read()):
#                 print("ERROR: Failed to parse the ONNX file.")
#                 for error in range(parser.num_errors):
#                     print(parser.get_error(error))
#                 return None
#         print('Completed parsing of ONNX file')
#         # "3. 构建engine"
#         print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
#         engine = builder.build_engine(network, config)  # Use build_serialized_network instead.
#         print("Completed creating Engine")
# 	# "4. 保存engine"
#         if save_engine:
#             with open(engine_file_path, "wb") as f:
#                 f.write(engine.serialize())
#         return engine
#############
# engine = build_engine(0,"onnx_p32/owlvit-image.onnx","onnx_p32/owlvit-image_fp16.engine",True,)

tensorrt_version = trt.__version__
major_version = int(tensorrt_version.split('.')[0])
minor_version = int(tensorrt_version.split('.')[1])
device = torch.cuda.current_device()
total_memory = torch.cuda.get_device_properties(device).total_memory

class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=self.device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        if major_version >= 10:
            num_io_tensors = model.num_io_tensors
            names = [model.get_tensor_name(i) for i in range(num_io_tensors)]
            num_inputs = sum(
                1 for name in names
                if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
            num_outputs = num_io_tensors - num_inputs
        else:
            num_bindings = model.num_bindings
            names = [model.get_binding_name(i) for i in range(num_bindings)]
            num_inputs = sum(1 for i in range(num_bindings)
                             if model.binding_is_input(i))
            num_outputs = num_bindings - num_inputs

        self.bindings: List[int] = [0] * (num_inputs + num_outputs
                                          )  # คงไว้เพื่อ TensorRT 8
        self.num_bindings = num_inputs + num_outputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]
        self.idx = list(range(self.num_outputs))

    def __init_bindings(self) -> None:
        idynamic = odynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            if major_version >= 10:
                dtype = self.dtypeMapping[self.model.get_tensor_dtype(name)]
                shape = tuple(self.model.get_tensor_shape(name))
            else:
                assert self.model.get_binding_name(i) == name
                dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
                shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                idynamic |= True
            inp_info.append(Tensor(name, dtype, shape))
        for i, name in enumerate(self.output_names):
            j = i + self.num_inputs
            if major_version >= 10:
                dtype = self.dtypeMapping[self.model.get_tensor_dtype(name)]
                shape = tuple(self.model.get_tensor_shape(name))
            else:
                assert self.model.get_binding_name(j) == name
                dtype = self.dtypeMapping[self.model.get_binding_dtype(j)]
                shape = tuple(self.model.get_binding_shape(j))
            if -1 in shape:
                odynamic |= True
            out_info.append(Tensor(name, dtype, shape))

        if not odynamic:
            self.output_tensor = [
                torch.empty(info.shape, dtype=info.dtype, device=self.device)
                for info in out_info
            ]
        self.idynamic = idynamic
        self.odynamic = odynamic
        self.inp_info = inp_info
        self.out_info = out_info

    def set_profiler(self, profiler: Optional[trt.IProfiler]):
        self.context.profiler = profiler if profiler is not None else \
            trt.Profiler()

    def set_desired(self, desired: Optional[Union[List, Tuple]]):
        if isinstance(desired,
                      (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs) -> Union[Tuple, torch.Tensor]:
        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[torch.Tensor] = [
            i.contiguous() for i in inputs
        ]

        if major_version >= 10:
            for i, name in enumerate(self.input_names):
                self.context.set_tensor_address(
                    name, contiguous_inputs[i].data_ptr())
                if self.idynamic:
                    self.context.set_input_shape(
                        name, tuple(contiguous_inputs[i].shape))

            outputs: List[torch.Tensor] = []
            for i, name in enumerate(self.output_names):
                if self.odynamic:
                    shape = tuple(self.context.get_tensor_shape(name))
                    output = torch.empty(size=shape,
                                         dtype=self.out_info[i].dtype,
                                         device=self.device)
                else:
                    output = self.output_tensor[i]
                self.context.set_tensor_address(name, output.data_ptr())
                outputs.append(output)

            success = self.context.execute_async_v3(self.stream.cuda_stream)
            if not success:
                raise RuntimeError('TensorRT execution failed')
        else:
            for i in range(self.num_inputs):
                self.bindings[i] = contiguous_inputs[i].data_ptr()
                if self.idynamic:
                    self.context.set_binding_shape(
                        i, tuple(contiguous_inputs[i].shape))

            outputs: List[torch.Tensor] = []
            for i in range(self.num_outputs):
                j = i + self.num_inputs
                if self.odynamic:
                    shape = tuple(self.context.get_binding_shape(j))
                    output = torch.empty(size=shape,
                                         dtype=self.out_info[i].dtype,
                                         device=self.device)
                else:
                    output = self.output_tensor[i]
                self.bindings[j] = output.data_ptr()
                outputs.append(output)

            self.context.execute_async_v2(self.bindings,
                                          self.stream.cuda_stream)
        self.stream.synchronize()
        return tuple(outputs[i]
                     for i in self.idx) if len(outputs) > 1 else outputs[0]



# def predicts(model_path,images):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = TRTModule(model_path,device)
#     t0=time.time()
#     images = torch.from_numpy(images)

#     result = model.forward(images.to(device))
#     t1=time.time()
#     cost_time = (t1-t0)*1000
#     print(f" inference cost time {cost_time} ms")
#     return result

if __name__ == '__main__':
    # model_path = 'onnx_p32\owlvit-image.trt' #50ms
    # model_path = 'onnx_p32\owlvit-image-bf16.trt' # 
    # model_path = 'onnx_p32\owlvit-image-fp16.trt' #33ms
    # image_path = 'bottle.jpg'
    # image = cv2.imread(image_path)
    # # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # target= 'a photo of red bottle'
    # # input_ids,attention_mask,pixel_values=preprocess_input(image,'a photo of  '+text_input[0])
    # input_ids,attention_mask,pixel_values=preprocess_input(image,target)
    # pixel_values = pixel_values.transpose(0,3,1,2)
    # pred = predicts(model_path,pixel_values)
    print("ok")

