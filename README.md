# AICM
aka **A**lternating **I**mputation and **C**orrection **M**ethod

2018 Zhiyue Tom Hu, Yuting Ye, Patrick A. Newbury, Haiyan Huang and Bin Chen

Full paper available at: https://www.biorxiv.org/content/early/2018/08/07/386896

## Data:

If you are simply interested in fetching the corrected and/or the raw data, please see our corrected data and raw data in ```/data```.

## Algorithm:

Current version is 0.1. We will be upgrading soon, stay tuned!

Please make sure you have all dependencies installed in your python environment.

You can run it in the following way:

```python aicm.py /data/original_data/GDSC_CTRP_auc.csv /data/original_data/CTRP_GDSC_auc.csv /imputed [Options]```

The first two arguments are the path to your desired-to-correct .csv files. The third one is where the output will be automatically saved to. They will be named as ```df1_imp.csv``` and ```df2_imp.csv```.

If you would like to tune parameters, please use ```python aicm.py /data/original_data/GDSC_CTRP_auc.csv /data/original_data/CTRP_GDSC_auc.csv /imputed -h``` to find out what option is, what do they mean and their default values.

An example would be:

```python aicm.py /data/original_data/GDSC_CTRP_auc.csv /data/original_data/CTRP_GDSC_auc.csv /imputed --dropping_rate 0.1``` to change the dropping rate to 0.1 instead of default 0.05.