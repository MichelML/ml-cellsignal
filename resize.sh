#!/bin/bash
jupyter nbconvert --to script edaresize.ipynb;
ipython edaresize.py;
