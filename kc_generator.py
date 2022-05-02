"""
@author: Alexander Carruth
"""

import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def oli_kc_data(dataset):
    '''
    Load OLI data from excel file (cached h5f files used to speed up)
    Args:
        dataset (str): short name for dataset ('full')

    Raises:
        ValueError: if dataset doesn't match an allowed type

    Returns:
        oli (DataFrame): OLI log file data, with columns renamed to shortened 
                   forms, and some columns removed (see source) 

    '''
    excel_files = {'full': 'ds5100_student_step_All_Data.xlsx'}
    
    if dataset not in excel_files:
        raise ValueError(str(dataset)+' not an allowed value to load_data()')
    data_location = os.path.join('datasets')
    
    #Add folder to datapath
    hdf_path = os.path.join(data_location,dataset)
    
    # Check if it is a regular file otherwise use the read excel file function
    if os.path.isfile(hdf_path):
        oli = pd.read_hdf(hdf_path)
    else:    
        excel_path = os.path.join(data_location, excel_files[dataset])
        oli = pd.read_excel(excel_path)
        oli.to_hdf(hdf_path, key='oli')
    
    # Map columns in excel file to shortened names
    name_map = {
        'Anon Student Id' : 'stud',
        'Problem Name' : 'pname',
        'First Attempt': 'first',
        'Incorrects': 'incorrects',
        'Hints': 'hints',
        'Corrects': 'corrects',
        'Opportunity (chemistry_general1-4_5)': 'opp',
        'KC (chemistry_general1-4_5)': 'kc'
        }
    
    # Drop columns that are not in name map and rename columns to be the same
    # As name map
    drop_columns = [x for x in oli.columns.tolist() if x not in name_map]
    oli = oli.drop(columns = drop_columns)
    oli = oli.rename(columns = name_map)
    oli = oli.dropna()
    
    return oli
#%% Load in the data
oli_df = oli_kc_data('full')
#%% Create dataframe where rows only map to one knowledge concept
kc_rm = []
# Sort out which questions have multiple kcs
for col,kc in oli_df['kc'].items():
    if('~~' in kc):
        kc_rm.append(kc)
# Create separate dataframes for one with multiple kcs and one with single kcs
oli_rm = oli_df[oli_df['kc'].isin(kc_rm)]
oli_app = oli_df[~oli_df['kc'].isin(kc_rm)]
# Split up the kcs and their respective observations into arrays
oli_rm['kc'] = oli_rm['kc'].str.split("~~")
oli_rm['opp'] = oli_rm['opp'].str.split("~~")
# Convert the arrays to single rows and match the rows for kc to their opportunity
oli_rm = oli_rm.explode(list(('kc','opp')))
# Add the removed kcs back in
oli_kc = oli_app.append(oli_rm)

# Get the unique kcs
kc_uni = oli_kc['kc'].unique()
oli_kc = oli_kc[oli_kc['kc'].isin(kc_uni)]
# Convert opportunities from strings to numbers
oli_kc['opp'] = pd.to_numeric(oli_kc['opp'])
# Sort the dataframe by opportunities in ascending order
oli_kc = oli_kc.sort_values('opp', axis=0)

#%% Generate knowledge curve DataFrame for all knowledge curves
# Drop nans and get the opportunity counts
opp_list = oli_kc['opp'].dropna().unique()
def generate_kc_curve(kc):
    kc_curve = pd.DataFrame()
    # Get dataframe for the current kc
    df1 = oli_kc[oli_kc['kc'] == kc]
    for opp in opp_list:
        # Get dataframe for the current opportunity
        df2 = df1[df1['opp'] == opp]
        # count the total corrects and incorrects and divide incorrect by total
        # save the answer of opportunity number, error rate, and knowledge concept
        incorrects = np.sum(df2['incorrects'])
        total_answers = np.sum(df2['corrects']) + incorrects
        if(total_answers != 0):
            error_rate = incorrects/total_answers * 100
        else:
            error_rate = math.nan
        append_frame = {'kc': kc, 'opp': opp, 'er': error_rate}
        kc_curve = kc_curve.append(append_frame, ignore_index = True)
    return kc_curve

kc_all_curve = []
# Create list of knowledge curves for every knowledge concept
for kc in kc_uni:
    kc_all_curve.append(generate_kc_curve(kc))

#%% Plot knowledge curve for a single kc
plt.figure(1)
test = kc_all_curve[0]['er']
plt.plot(kc_all_curve[0]['opp'],kc_all_curve[0]['er'],'.r-')

#%% Generate average error rate per observation over all knowledge curves
# Sort kc dataframe in ascending order and get list of unique values
oli_kc = oli_kc.sort_values('opp', axis=0)
opp_list = oli_kc['opp'].dropna().unique()

# Calculate the error rate for all opportunities in kc dataframe
df_toter = pd.DataFrame()
for opp in opp_list:
    # Get dataframe for the current opportunity
    df1 = oli_kc[oli_kc['opp'] == opp]
    incorrects = np.sum(df1['incorrects'])
    total_answers = np.sum(df1['corrects']) + incorrects
    er_mean = incorrects/total_answers * 100
    append_frame = {'opp':opp, 'erm':er_mean}
    df_toter = df_toter.append(append_frame, ignore_index = True)


#%% Plot knowledge curve for all observations
plt.figure(1)
plt.plot(df_toter['opp'],df_toter['erm'],'.r-')
plt.ylabel('Error Rate (%)')
plt.xlabel('Observation')
plt.title('Knowledge Curve')
plt.show()


