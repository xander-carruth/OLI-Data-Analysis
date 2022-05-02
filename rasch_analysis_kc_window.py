"""
@author: Alexander Carruth
"""

import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

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
    #dataset = 'full'
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

#%% Get questions dataset
oli_qdf = oli_quest_data('quest')

#%%
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

# Get the unique kcs
kc_uni = oli_kc['kc'].unique()
oli_kc = oli_kc[oli_kc['kc'].isin(kc_uni)]
# Convert opportunities from strings to numbers
oli_kc['opp'] = pd.to_numeric(oli_kc['opp'])
# Sort the dataframe by opportunities in ascending order
oli_qkc = oli_kc.sort_values('opp', axis=0)

#%% Get average kc grade for every student
students = oli_qkc['stud'].unique()
score_dist = {}
for stud in students:
    kc_dist = {}
    # Get dataframe for the current student
    df1 = oli_qkc[oli_qkc['stud']==stud]
    for kc in kc_uni:
        # Get dataframe for the current kc
        df2 = df1[df1['kc']==kc]
        corrects = df2['corrects'].sum()
        total = corrects + df2['incorrects'].sum()
        if(total != 0):
            kc_dist[kc] = (corrects/total) * 100
        else:
            kc_dist[kc] = math.nan
    score_dist[stud] = kc_dist


#%% Get scores of every student on every question
# Get unique problem names
pnames = oli_qdf['pname'].unique()
quest_score = {}
for pname in pnames:
    # Get dataframe for the current problem name
    df1 = oli_qdf[oli_qdf['pname']==pname]
    stus_dict = {}
    for stud in students:
        # Get dataframe for the current student
        df2 = df1[df1['stud']==stud]
        corrects = df2['corrects'].sum()
        total = corrects + df2['incorrects'].sum()
        if(not total == 0):
            stus_dict[stud] = (corrects/total) * 100
        else:
            stus_dict[stud] = math.nan
    quest_score[pname] = stus_dict

#%% Get average score of every student on all kcs for a specific question
p_dist = {}
for pname in pnames:
    # Get dataframe for the current problem
    oli_pqkc = oli_qkc[oli_qkc['pname']==pname]
    temp_dist = {}
    for stud in students:
        overall_avg = 0
        count = 0
        # For every kc that matches the question, add the student score
        for kc in oli_pqkc['kc']:
            overall_avg += score_dist[stud][kc]
            count += 1
        temp_dist[stud] = overall_avg/count
    p_dist[pname] = temp_dist
    

#%% Create dataframe of data for plotting
quest_plot = pd.DataFrame()
for quest in quest_score:
    x = []
    y = []
    for stud in quest_score[quest]:
        temp = quest_score[quest]
        if(not math.isnan(temp[stud])):
            x.append(p_dist[quest][stud])
            y.append(temp[stud])
    quest_df = pd.DataFrame({'Question':[quest], 'stu_score':[x], 'q_score':[y]})
    quest_plot = pd.concat([quest_plot, quest_df])

#%% Get the question name, student scores and question score for q_num
q_num = 5
df_quest = quest_plot.iloc[q_num]
plt_name = df_quest['Question']
stu_scores = df_quest['stu_score']
q_scores = df_quest['q_score']

#%% Generate plot sigmoid for sliding window
# Running mean function over sliding window. Taking the cumulative sum from after the
# first N and before the last N and subtracting shows the every sum over 5 values
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# Sigmoid function for curve_fit
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

# Get sliding window average over 5 students for student and question scores
stu_score = running_mean(stu_scores, 5)
q_score = running_mean(q_scores, 5)

# Notes for adjusting values found in the rasch_analysis_window.py file
p0 = [(max(q_score)-min(q_score)), np.median(stu_score),(2/3),min(q_score)] # this is an mandatory initial guess

fev = 1000
# Increase maxfev if given error maximum number of evaluations is exceeded.
# Default maxfev is 800
while True:
    try:
        popt, pcov = curve_fit(sigmoid, stu_score, q_score, p0, method='dogbox', maxfev=fev)
    except RuntimeError:
        fev += 2000
    else:
        break

#%% Generate plot of sigmoid for sliding window with kc score as x axis
# Get a range of 200 values to plot the sigmoid with the optimized parameters over
x = np.linspace(min(stu_score)-5, max(stu_score)+5, 200)
y = sigmoid(x, *popt)

plt.plot(stu_score, q_score, 'o', label='data')
plt.plot(x,y, label='fit')
plt.title(plt_name)
plt.ylabel('Question Score (%)')
plt.xlabel('KC Score (%)')
plt.legend(loc='best')
