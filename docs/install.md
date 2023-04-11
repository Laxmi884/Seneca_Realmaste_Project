# Install Python envirionment

as root
'''
yum install libffi-devel bzip2-devel
echo 1 > /proc/sys/vm/overcommit_memory
yum install gcc gcc-c++ zlib-devel zlib-static openssl-devel openssl-static
'''

as user
'''
mkdir python3
curl -O https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz
tar -xvf Python-3.9.16.tgz
cd Python-3.9.16
./configure --enable-optimizations --prefix=$HOME/python3
make altinstall
cd ../bin
python3.9 --version

cd $HOME
python3/bin/python3.9 -m venv py3
source ./py3/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy lightgbm pymongo regex sympy pyrsistent psutil

'''
