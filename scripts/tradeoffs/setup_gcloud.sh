#!/bin/bash

curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
cd ~
sudo apt -y install python3-pip
sudo apt -y install python3-virtualenv
virtualenv compressai
source compressai/bin/activate
pip install pandas
pip install compressai==1.2.4
cd ~
mkdir results_vae
cd results_vae
nohup python3 /home/divpunb/perceptronac/scripts/tradeoffs/train_vaes.py /home/lucas/vimeo90k_img_nosl > /home/divpunb/train_vaes.log &