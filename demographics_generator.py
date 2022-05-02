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
        'Opportunity (Unique-step)': 'opp'
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

#%% Get questions dataset
oli_qdf = oli_quest_data('quest')

#%% Calculate scores on each quiz for the minority and non-minority groups
# Get the names of the quiz pools
pnames_quiz = set([x for x in oli_qdf['pname'] if x.count('quiz')])

min_score = []
maj_score = []
# Get questions for minority students
minq_df = oli_qdf[oli_qdf['stud'].isin(min_stud)]
# Get questions for non minority students
majq_df = oli_qdf[oli_qdf['stud'].isin(maj_stud)]

# Get scores on quiz pools for the minorities
for name in pnames_quiz:
    temp_minq = minq_df[minq_df['pname']==name]
    corrects = temp_minq['corrects'].sum()
    total = corrects + temp_minq['incorrects'].sum()
    if(total != 0):
        tot_score = corrects/total * 100
    else:
        tot_score = math.nan
    min_score.append(tot_score)

# Get scores on quiz pools for the non minorities
for name in pnames_quiz:
    temp_majq = majq_df[majq_df['pname']==name]
    corrects = temp_majq['corrects'].sum()
    total = corrects + temp_majq['incorrects'].sum()
    if(total != 0):
        tot_score = (corrects/total) * 100
    else:
        tot_score = math.nan
    maj_score.append(tot_score)

#%% Plot the average quiz grades for the minority groups
x_coord = [i+1 for i in range(len(min_score))]
plt.plot(x_coord, min_score, '.r', label='Minority')
plt.plot(x_coord, maj_score, '.b', label='Non-Minority')
plt.ylabel('Grade (%)')
plt.xlabel('Quiz')
plt.legend()
plt.title('Minority vs Non-Minority Scores')
plt.show()

#%% Find the first gen college students in the dataframe
#Get the problem that correspond to first_gen (q3_first)
fir_df = dg_df[dg_df['sname']=='q3_second']

#Remove whites space from the answers
fir_df['ans'] = fir_df['ans'].str.strip()

#Get the responses that responded yes to being first generation
first_gen = fir_df[fir_df['ans']=='yes']

#%% Get lists of first gen and non first gen students who answered the survey
fir_stud = first_gen['stud'].unique()
nonfir_stud = np.array([])
for stud in dg_df['stud'].unique():
    if(not (stud in fir_stud)):
        nonfir_stud = np.append(nonfir_stud, stud)

#%% Print the number of students in each first gen category
print("Number of first generation students: "+str(len(fir_stud)))
print("Number of non-minority students: "+str(len(nonfir_stud)))

#%% Calculate scores on each quiz for the first gen and non first gen groups
# Get names for the quiz pools
pnames_quiz = set([x for x in oli_qdf['pname'] if x.count('quiz')])

fir_score = []
nonfir_score = []
# Get the questions for first gen college students
firq_df = oli_qdf[oli_qdf['stud'].isin(fir_stud)]
# Get the questions for non first gen college students
nonfirq_df = oli_qdf[oli_qdf['stud'].isin(nonfir_stud)]

# Get the scores for first gen college students
for name in pnames_quiz:
    temp_firq = firq_df[firq_df['pname']==name]
    corrects = temp_firq['corrects'].sum()
    total = corrects + temp_firq['incorrects'].sum()
    if(total != 0):
        tot_score = corrects/total * 100
    else:
        tot_score = math.nan
    fir_score.append(tot_score)

# Get the scores for non first gen college students
for name in pnames_quiz:
    temp_nonfirq = nonfirq_df[nonfirq_df['pname']==name]
    corrects = temp_nonfirq['corrects'].sum()
    total = corrects + temp_nonfirq['incorrects'].sum()
    if(total != 0):
        tot_score = (corrects/total) * 100
    else:
        tot_score = math.nan
    nonfir_score.append(tot_score)

#%% Plot the average quiz grades for the first gen groups
x_coord = [i+1 for i in range(len(fir_score))]
plt.plot(x_coord, fir_score, '.r', label='First Gen')
plt.plot(x_coord, nonfir_score, '.b', label='Non First Gen')
plt.ylabel('Grade (%)')
plt.xlabel('Quiz')
plt.legend()
plt.title('First Gen vs Non First Gen Scores')
plt.show()