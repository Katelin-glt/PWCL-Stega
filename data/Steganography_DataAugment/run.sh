#!/bin/bash

python HC.py -dataset Twitter -generate-num 100000 -idx-gpu 0
python HC.py -dataset IMDB -generate-num 100000 -idx-gpu 0
python AC.py -dataset Twitter -generate-num 10 -idx-gpu 0
python AC.py -dataset IMDB -generate-num 100000 -idx-gpu 0
python ADG.py -dataset IMDB -generate-num 10 -idx-gpu 0
python ADG.py -dataset Twitter -generate-num 1000 -idx-gpu 0
python ADG.py -dataset News -generate-num 100 -idx-gpu 0