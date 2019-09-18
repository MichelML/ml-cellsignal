#!/bin/bash
bash /ml-cellsignal/prequirements.txt;
pip install -q -r /ml-cellsignal/requirements.txt;
jupyter nbconvert --to script /ml-cellsignal/experiment13-cellline.ipynb;
ipython /ml-cellsignal/experiment13-cellline.py;
