#!/bin/bash
if [ ! -d 3rdparty/ax650n_bsp_sdk ]; then
  echo "clone ax650 bsp to ax650n_bsp_sdk, please wait..."
  cd 3rdparty
  git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
fi