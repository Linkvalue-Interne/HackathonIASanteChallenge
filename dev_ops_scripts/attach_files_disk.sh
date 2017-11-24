sudo apt-get update
sudo apt-get install cifs-utils

sudo mkdir /sharedfiles

sudo mount -t cifs //3e25b31e87cfdsvm.file.core.windows.net/alibaba /sharedfiles -o vers=2.1,username=3e25b31e87cfdsvm,password=AtggZC/KcwE2EL57gFch+HWdZWfqweSfZws43AkQ0+1310dzHOq+mm5dMlwjGJvlTJ3cO9bp2q69j8K1s5NcsQ==,dir_mode=0777,file_mode=0777,serverino

sudo bash -c 'echo "//3e25b31e87cfdsvm.file.core.windows.net/alibaba /sharedfiles -o vers=2.1,username=3e25b31e87cfdsvm,password=AtggZC/KcwE2EL57gFch+HWdZWfqweSfZws43AkQ0+1310dzHOq+mm5dMlwjGJvlTJ3cO9bp2q69j8K1s5NcsQ==,dir_mode=0777,file_mode=0777,serverino"'