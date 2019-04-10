#/usr/bin/bash

BNC_ZIP_ARCHIVE='2554.zip'
BNC_UNZIPPED_DIR='bnc'
BNC_FILELIST='2554.filelist.txt'
BNC_CHECKSUM='2554.checksum.txt'

if [ -f $BNC_ZIP_ARCHIVE ]; then
	unzip -l $BNC_ZIP_ARCHIVE > $BNC_FILELIST
	mkdir -p $BNC_UNZIPPED_DIR
	unzip -qu $BNC_ZIP_ARCHIVE -d $BNC_UNZIPPED_DIR
	find $BNC_UNZIPPED_DIR -type f -exec md5sum {} \; > $BNC_CHECKSUM
else
	echo "$BNC_ZIP_ARCHIVE does not exist"
fi
