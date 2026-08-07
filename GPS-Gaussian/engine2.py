import os
import tensorrt as trt

def build_engine(onnx_file_path, engine_file_path):
    if os.path.exists(engine_file_path):
        os.remove(engine_file_path)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    flags = 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    #메모리 8gb할당코드 필요없지 않나 싶음
    # try:
    #     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (1024 ** 3))
    # except AttributeError:
    #     config.max_workspace_size = 8 * (1024 ** 3)

    if not parser.parse_from_file(onnx_file_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return
    engine_bytes = builder.build_serialized_network(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    print("saved:", engine_file_path)

if __name__ == "__main__":
    os.makedirs("trt_engines", exist_ok=True)
    build_engine("onnx_models/gs_regresser.onnx", "trt_engines/gs_regresser.engine")