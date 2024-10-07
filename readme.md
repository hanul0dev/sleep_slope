<h1 align="center">YOLOv8 steep slope collapse prediction</h1>

## Introduction
Developed based on YOLOv8.<br><br>

In the past 5 years, steep slope collapse accidents in mountainous areas of Korea have been steadily increasing.<br><br>

I was looking for a way to solve this problem,
and I am studying how to build a system at low cost by receiving existing CCTV as RTSP using Jetson.<br><br>

First, I built a mock test environment for the project.<br><br>

I implemented ground distortion using Arduino, and developed a tracking system by installing 9 reference pillars on the ground.<br><br>




## :hammer_and_wrench: Installing requirements and running the repo
Here is the installation method for using Python 3.8:

1. update & upgrade<br>
sudo apt update
sudo apt upgrade<br><br>

2. Install required files<br>
sudo apt install build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev libc6-dev<br><br>

3. Get python3.8 source code<br>
cd /
wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz<br><br>

4. Unzip<br>
tar -xf Python-3.8.12.tar.xz
cd Python-3.8.12<br><br>

5. Build<br>
./configure --enable-optimizations
make -j4<br><br>

6. finish<br>
sudo make altinstall
python3.8 --version<br><br>

7. Virtual Environment (IMPORTANT!!)<br>
python3.8 -m venv myenv                                     
source myenv/bin/activate


