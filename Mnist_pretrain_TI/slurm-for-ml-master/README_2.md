./run_file.sh MNIST_jobs.txt 

To filter out jobs when rerunning:

./filter_failed.sh Mnist_pretrain_TI_jobs.txt > failed_jobs.txt
./run_file.sh failed_jobs.txt
