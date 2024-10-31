FILE=edges2shoes
# URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
mkdir -p $TARGET_DIR
# echo "Downloading $URL dataset..." to $TARGET_DIR
# wget -N $URL -O $TAR_FILE
# curl -L $URL -o $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE

find "${TARGET_DIR}train" -type f -name "*.jpg" |sort -V > ./edges2shoes_train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" |sort -V > ./edges2shoes_val_list.txt
