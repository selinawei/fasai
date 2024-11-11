The file is used for explain specific use of everything

welcome!


## Dataset

1. For All-in-focus, the raw dataset is at ./SAI-data-align/complex_align and ./SAI-data-align/complex_align
   1. these data include these: (please align before this!!!)
        1. k, Shape: (3, 3)
            Values: [[421.2654   0.     189.1559]
            [  0.     420.3611 141.4478]
            [  0.       0.       1.    ]]

        2. p, Shape: (5,)
            Values: [-0.3242  0.1516  0.      0.      0.    ]

        3. v, Shape: ()
            Values: 0.1775

        4. size, Shape: (2,)
            Values: [260 346]

        5. events, Shape: (5262484, 4)
            Values: 1699334819.7283132

        6. occ_free_aps, Shape: (25, 260, 346)
   
        7.  occ_free_aps_ts, Shape: (25,)
            Values: ['1699334819.7212212' '1699334819.7575622' '1699334819.7939031'
            '1699334819.8302453' '1699334819.8665862' '1699334819.9029272'
            '1699334819.9392681' '1699334819.9756093' '1699334820.0119503'
            '1699334820.0482912' '1699334820.0846322' '1699334820.1209731'
            '1699334820.1573143' '1699334820.1936553' '1699334820.2299972'
            '1699334820.2663381' '1699334820.3026793' '1699334820.3390203'
            '1699334820.3753612' '1699334820.4117022' '1699334820.4480431'
            '1699334820.4843843' '1699334820.5207253' '1699334820.5570662'
            '1699334820.5934081']

2. use ./prepare_data.py to split Train and Test data and do data preprocess
    specifically, to achieve it, you should change the config in ./arguments/prepare_data.py, the "data_path" is supposed to be the file contains files of different kinds of dataset
    the "content" is supposed to be the kind of data that you want to preprocess
    the data_name is supposed to be the file name that contains the Train and Test file of the preprocessed dataset.
    by the way, the "data_name" is supposed to be like"./ahaha", while content is supposed to belike [simple_data]
    (go ./.vscode/launch.json to see instance)

3. then, your preprocessed data will be stored in "data_name"
    


## train
1. you need to change the data file in the training config: ./SAI-data-align./arguments/__init__.py
   same as the data preprocess process, the "data_path" is supposed to be the file contains files of different kinds of dataset
    the "content" is supposed to be the kind of data that you want to preprocess

2. use two terminal, one is for running the ./train.py, the other is for run:
    ### tensorboard --logdir=./results/exp_test/0 --port 6998 --bind_all
    while the is supposed to be identical as your log_dir in 
    ### SummaryWriter(log_dir=f"{results_dir}/{rank}",flush_secs=10)
    specifically, you can change zhe training config exp_name, therefore you can look at the outcome of each exp

    in this instance, the data and image will be stored in the file ./results/exp_test/0 and will be read by tensorboard
    to see it use the reply url to see it.

## For nd server
1. you can use two command, one in nd server:
   ### tensorboard --logdir=/scratch365/lwei5/results/FEA_exp_withdepth3_20241020-022701/0 --port 6998 --host=0.0.0.0
    another in your local machine:
   ### ssh -L 16006:localhost:6998 lwei5@crcfe02.crc.nd.edu
    起到一个转发的作用，然后在浏览器的地址栏输入localhost:16006即可看到tensorboard的界面