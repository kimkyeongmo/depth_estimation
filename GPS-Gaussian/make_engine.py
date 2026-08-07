import os
import tensorrt as trt

def build_engine(onnx_file_path, engine_file_path):
    if os.path.exists(engine_file_path):
        os.remove(engine_file_path)
        
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # 최신 TRT API 규격에 맞춘 명시적 배치(Explicit Batch) 플래그
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # 작업 공간 메모리 할당 (8GB)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (1024 ** 3))
    except AttributeError:
        config.max_workspace_size = 8 * (1024 ** 3)
    
    #💡 [핵심 수정 2] Blackwell 텐서 코어를 위한 FP16 정밀도 활성화
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"Building TRT Engine with FP16 + TF32 Optimization: {onnx_file_path}")
    else:
        print(f"Building TRT Engine with Native TF32 Optimization: {onnx_file_path}")
    
    if not parser.parse_from_file(onnx_file_path):
        print("[Error] ONNX parsing failed")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return
            
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("[Error] Engine build failed: " + onnx_file_path)
        return
        
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    print("✅ Engine saved: " + engine_file_path + "\n")

if __name__ == "__main__":
    os.makedirs("trt_engines", exist_ok=True)
    #build_engine("./onnx_models/rvm_resnet50_static.onnx", "./trt_engines/rvm_resnet50_static1.engine")
    build_engine("onnx_models/vda_model.onnx", "trt_engines/vda_model.engine") # fp16가능
    build_engine("onnx_models/unet_extractor.onnx", "trt_engines/unet_extractor.engine")# unet은 fp16가능
    build_engine("onnx_models/gs_regresser.onnx", "trt_engines/gs_regresser.engine")# fp16 불가능