XNDL GIA UPC
Zero model training code (training_MAMepy) for the CTE-P9 cluster (launcher.psh). Before you run this code, data must be downloaded (MAMe 256x256 dataset[1]) and set up (folderfy).

You can run the training (training_MAMe.py) in the CTE-P9 cluster through the launcher.py script. "training_MAMe.py" requires the data to be available in a given location (DATASET_PATH), and structure ("partition/classname"). Check if data is available on the default location (i.e., check if you have read rights on DATASET_PATH). Otherwise you'll need to run folderby.

[1] https://hpai.bsc.es/MAMe-dataset/
