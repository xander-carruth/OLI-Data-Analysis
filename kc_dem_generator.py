"""
@author: Alexander Carruth
"""

import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def oli_dem_data(dataset):
    '''
    Load OLI demographic data from excel file (cached h5f files used to speed up)
    Args:
        dataset (str): short name for dataset ('dem')

    Raises:
        ValueError: if dataset doesn't match an allowed type

    Returns:
        oli (DataFrame): OLI log file data, with columns renamed to shortened 
                   forms, and some columns removed (see source) 

    '''
    excel_files = {'dem': 'ds5100_tx_All_Data.xlsx'}
    
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
        'Selection' : 'sname',
        'Input' : 'ans'
        }
    
    # Drop columns that are not in name map and rename columns to be the same
    # As name map
    drop_columns = [x for x in oli.columns.tolist() if x not in name_map]
    oli = oli.drop(columns = drop_columns)
    oli = oli.rename(columns = name_map)
    oli = oli.dropna()
    
    return oli

def oli_quest_data(dataset):
    '''
    Load OLI question data from excel file (cached h5f files used to speed up)
    Args:
        dataset (str): short name for dataset ('quest')

    Raises:
        ValueError: if dataset doesn't match an allowed type

    Returns:
        oli (DataFrame): OLI log file data, with columns renamed to shortened 
                   forms, and some columns removed (see source) 

    '''
    excel_files = {'quest': 'ds5100_student_step_All_Data.xlsx'}
    
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

#%% Create a dataframe for the demographic data
oli_ddf = oli_dem_data('dem')

#%% Get the demographic survey
# Get the questions that correspond to the problem name of the demographics survey
dg_df = oli_ddf[oli_ddf['pname']=='inline_pre_student_survey3']
# Filter the tags out of the answer
dg_df['ans'] = dg_df['ans'].replace({'<material>':''}, regex=True)
dg_df['ans'] = dg_df['ans'].replace({'</material>':''}, regex=True)

#%% Find the minorities in the dataframe
#Get the problems that correspond to minorities (one, three, four, five, six)
min_df = dg_df[(dg_df['sname']=='q2_table_one') | (dg_df['sname']=='q2_table_three') |
              (dg_df['sname']=='q2_table_four') | (dg_df['sname']=='q2_table_five') |
              (dg_df['sname']=='q2_table_six')]

# Remove white space from the answer
min_df['ans'] = min_df['ans'].str.strip()

# Get the responses that responded yes to being a minority
minorities = min_df[min_df['ans']=='yes']

#%% Get lists of minority and majority students who answered the survey
min_stud = minorities['stud'].unique()
maj_stud = np.array([])
for stud in dg_df['stud'].unique():
    if(not (stud in min_stud)):
        maj_stud = np.append(maj_stud, stud)
        
#%% Print the number of students in each minority category
print("Number of minority students: "+str(len(min_stud)))
print("Number of non-minority students: "+str(len(maj_stud)))

#%% Create a dataframe for the question data
oli_qdf = oli_quest_data('quest')

#%% Create dataframe where rows only map to one knowledge concept
kc_rm = []
# Sort out which questions have multiple kcs
for col,kc in oli_qdf['kc'].items():
    if('~~' in kc):
        kc_rm.append(kc)
# Create separate dataframes for one with multiple kcs and one with single kcs
oli_rm = oli_qdf[oli_qdf['kc'].isin(kc_rm)]
oli_app = oli_qdf[~oli_qdf['kc'].isin(kc_rm)]
# Split up the kcs and their respective observations into arrays
oli_rm['kc'] = oli_rm['kc'].str.split("~~")
oli_rm['opp'] = oli_rm['opp'].str.split("~~")
# Convert the arrays to single rows and match the rows for kc to their opportunity 
oli_rm = oli_rm.explode(list(('kc','opp')))
# Add the removed kcs back in
oli_kc = oli_app.append(oli_rm)
kc_uni = oli_kc['kc'].unique()

# Get the unique kcs
oli_kc = oli_kc[oli_kc['kc'].isin(kc_uni)]
# Convert opportunities from strings to numbers
oli_kc['opp'] = pd.to_numeric(oli_kc['opp'])
# Sort the dataframe by opportunities in ascending order
oli_kc = oli_kc.sort_values('opp', axis=0)

#%% Calculate the error rate for all observations in DataFrame
oli_kc = oli_kc.sort_values('opp', axis=0)
opp_list = oli_kc['opp'].dropna().unique()
# Minority dataframe
min_kc = oli_kc[oli_kc['stud'].isin(min_stud)]
# Get non minority dataframe
maj_kc = oli_kc[oli_kc['stud'].isin(maj_stud)]

df_min_tot = pd.DataFrame()
for opp in opp_list:
    # Get dataframe for the current opportunity
    df1 = min_kc[min_kc['opp'] == opp]
    incorrects = np.sum(df1['incorrects'])
    total_answers = np.sum(df1['corrects']) + incorrects
    if(total_answers != 0):
        er_mean = incorrects/total_answers * 100
    else:
        er_mean = math.nan
    append_frame = {'opp':opp, 'erm':er_mean}
    df_min_tot = df_min_tot.append(append_frame, ignore_index = True)

df_maj_tot = pd.DataFrame()
for opp in opp_list:
    # Get dataframe for the current opportunity
    df1 = maj_kc[maj_kc['opp'] == opp]
    incorrects = np.sum(df1['incorrects'])
    total_answers = np.sum(df1['corrects']) + incorrects
    if(total_answers != 0):
        er_mean = incorrects/total_answers * 100
    else:
        er_mean = math.nan
    append_frame = {'opp':opp, 'erm':er_mean}
    df_maj_tot = df_maj_tot.append(append_frame, ignore_index = True)


#%% Plot knowledge curve for all observations
plt.figure(1)
plt.plot(df_min_tot['opp'],df_min_tot['erm'],'.b-', label="Minority Students")
plt.plot(df_maj_tot['opp'],df_maj_tot['erm'],'.r-', label="Non-Minority Students")
plt.ylabel('Error Rate (%)')
plt.xlabel('Observation')
plt.title('Knowledge Curve')
plt.legend()
plt.show()