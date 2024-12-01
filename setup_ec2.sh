#!/bin/bash

# 시스템 업데이트
sudo apt-get update
sudo apt-get upgrade -y

# Python 및 pip 설치
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-venv

# Chrome 설치
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt-get install -y ./google-chrome-stable_current_amd64.deb

# ChromeDriver 설치
CHROME_DRIVER_VERSION=`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`
wget -N https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/bin/chromedriver
sudo chown root:root /usr/bin/chromedriver
sudo chmod +x /usr/bin/chromedriver

# 필요한 시스템 라이브러리 설치
sudo apt-get install -y xvfb
sudo apt-get install -y libgconf-2-4
sudo apt-get install -y libxss1
sudo apt-get install -y libnss3
sudo apt-get install -y libasound2

# Python 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 필요한 Python 패키지 설치
pip install -r requirements.txt 