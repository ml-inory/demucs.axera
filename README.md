# demucs.axera

音轨分离模型，用于将音乐分成 鼓、贝斯、人声和其他乐器 四轨 [官方repo](https://github.com/facebookresearch/demucs)

## 下载模型
```
./download_models.sh
```

如需自行转换模型请参考[模型转换](model_convert/README.md)

## Python运行 
```
cd python
```

安装依赖  
```
pip3 install -r requirements.txt
```

运行(axmodel)
```
python3 main.py -i 输入音频文件 -m AX模型 --segment 模型输入是几秒的
```

运行(onnx)
```
python3 main_onnx.py -i 输入音频文件 -m ONNX模型 --segment 模型输入是几秒的
```

## Cpp运行

下载BSP
```
./download_bsp.sh
```

编译  
```
./build.sh
```

运行
```
./install/demucs -m ../models/htdemucs.axmodel -i 输入音频.wav -o 输出目录
```