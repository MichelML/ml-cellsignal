#!/bin/bash
jupyter nbconvert --to script experiment13-cellline.ipynb;
ipython experiment13-cellline.py;
