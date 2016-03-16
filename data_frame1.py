#-*- coding: utf-8 -*-

import pandas as pd
from sklearn.decomposition import PCA

raw_data = pd.read_csv('test_users.csv', header= None)
raw_data.columns =['ID', 'Date_account_created', 'timestamp_first_active', 'date_first_booking', 'gender', 'age', 'signup_method', 'signup_flow', 
					'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
raw_data.dropna(how='all', inplace = True)


