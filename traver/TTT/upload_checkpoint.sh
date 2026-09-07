
document=Llama-3.2-3B-Instruct-sft-200_TutorModel1759386593_epoch-2.tar.gz
upload_dir=/home/wangjian/Coding-Tutor-Extension/checkpoints/ppo


# 解压缩文件
echo "extracting $document in $upload_dir"
ssh wangjian@somea6k "cd $upload_dir && tar -xzf epoch-0.tar.gz && echo 'extraction completed'"
echo "finish extracting $document in $upload_dir"

# 删除原始文件
echo "deleting $document in $upload_dir"
ssh wangjian@somea6k "cd $upload_dir && rm $document && echo 'deletion completed'"
echo "finish deleting $document in $upload_dir"
