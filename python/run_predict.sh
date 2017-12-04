for i in $(ls /sharedfiles/outputs/models/resnet*.h5); do python main.py predict resnet50 "$i"; done
