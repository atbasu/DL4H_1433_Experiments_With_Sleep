# DL4H_1433_Experiments_With_Sleep

This repository contains the results of experiments with learning from sleep data undertaken by Team 1433s as part of the CS 598 Deep Learning for Healthcare course offered by Prof. Jimeng Sun at UIUC. To get a quick overview of our experiments check out our [presentation](https://www.youtube.com/watch?v=L5BjQNSeVqA) on Youtube!

Included in this repostory are:
1. A white paper describing our experiments - [Automatic Sleep Stage Classification with CNN and Seq2Seq Models.pdf](https://github.com/atbasu/DL4H_1433_Experiments_With_Sleep/blob/main/Automatic%20Sleep%20Stage%20Classification%20with%20CNN%20and%20Seq2Seq%20Models.pdf)
2. A presentation summarising the results of our experments - [Automatic Sleep Stage Classification with CNN and Seq2Seq Models.pptx](https://github.com/atbasu/DL4H_1433_Experiments_With_Sleep/blob/main/Automatic%20Sleep%20Stage%20Classification%20with%20CNN%20and%20Seq2Seq%20Models.pptx)
3. A python notebook documenting our data experiments - [DL4H_1433_Experiments_With_Sleep.ipynb.ipynb]()
4. A python executable file that you can download and run - [DL4H_1433.py]()



## How to run the python notebook using Google Collab

1. Download a compressed zip file with all the [Physionet Sleep-EDF datasets](https://physionet.org/physiobank/database/sleep-edfx/) published between 2013 and 2018 as that will be used for this experimental notebook.
2. Store it in a folder in your google drive
3. Download the python notebook and also store it in your Google Drive location and then open with Google Collab
4. Execute the two code cells in step 1 sequentially to allow Google Collab access to the data you downloaded and stored in step 2
![alt text](https://github.com/atbasu/DL4H_1433_Experiments_With_Sleep/blob/main/enable%20google%20drive%20access%201.png?raw=true)
![alt text](https://github.com/atbasu/DL4H_1433_Experiments_With_Sleep/blob/main/enable%20google%20drive%20access%202.png?raw=true)
5. Now you can run the rest of the notebook sequentially

## How to run the python executable

1. Download a compressed zip file with all the [Physionet Sleep-EDF datasets](https://physionet.org/physiobank/database/sleep-edfx/) published between 2013 and 2018 as that will be used for this experimental notebook.
2. Decompress and store it in a folder named *'eeg_fpz_cz'* on your local machine at a location where you want to run the executable
3. Download the DL4H_1433.py file and store it in the same location where you created the eeg_fpz_cz folder
4. Ensure that your python environment supports all of the following python libraries:
 a. os
 b. csv
 c. pickle
 d. random
 e. numpy
 f. torch
 g. torchvision
 h. typing
 i. collections
 j. sklearn
 k. time
 l. matplotlib
 m. datetime
 You'll know one is missing if you see an error like this:
 ```
 (base) ATBASU-M-45BA:Project atbasu$ python3 DL4H_1433.py 
  Traceback (most recent call last):
    File "DL4H_1433.py", line 13, in <module>
      import torchvision.models as models
  ModuleNotFoundError: No module named 'torchvision'
 ```
 To fix it, simply install that library. If you have [anaconda](https://docs.anaconda.com/anaconda/install/) installed you can simply execute the following command:
 ```
 (base) ATBASU-M-45BA:Project atbasu$ conda install torchvision -c pytorch
 ```
