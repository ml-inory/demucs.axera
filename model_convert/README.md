# 模型转换

## 导出ONNX
```
python export_onnx.py
```

## Pulsar2
```
pulsar2 build --input htdemucs_ft.onnx --config htdemucs.json --output_dir htdemucs_ft --output_name htdemucs_ft.axmodel --target_hardware AX650 --compiler.check 2 --npu_mode NPU3
```