#!/bin/bash
### Remember to change ROOT to /YOUR/PATH ###
ROOT=/home/shuzhao/Data
#############################################


# make a list
split=val
root=${ROOT}/dataset/imagenet/${split}
python ./im2rec.py ${split} ${root} \
                    --list \
                    --recursive

# create a database
python ./im2rec.py ${split} ${root} \
                    --recursive \
                    --num-thread 16


split=train
root=${ROOT}/dataset/imagenet/${split}
python ./im2rec.py ${split} ${root} \
                    --list \
                    --recursive

# create a database
python ./im2rec.py ${split} ${root} \
                    --recursive \
                    --num-thread 16