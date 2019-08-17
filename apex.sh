#!/bin/sh
rm -rf apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
cd ..  
mv apex/apex apex1
rm -rf apex
mv apex1 apex
