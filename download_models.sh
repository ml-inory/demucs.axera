#!/bin/bash
if [ ! -f models/htdemucs_ft.axmodel ] || [ ! -f models/htdemucs_ft.onnx ]; then
    wget -c https://github.com/ml-inory/demucs.axera/releases/download/v1.0/models.tar.gz
    tar zxvf models.tar.gz
fi