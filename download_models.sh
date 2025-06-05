#!/bin/bash
if [ ! -f models/htdemucs.axmodel ]; then
    wget -c https://github.com/ml-inory/demucs.axera/releases/download/v1.0/htdemucs.axmodel
    mv htdemucs.axmodel models/
fi