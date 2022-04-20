# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:59:13 2022

@author: acarr
"""

'''
Steps for Knowledge Curve:
    Generate list of important features for comparing Prereq/Coreq and Scaffolded!
    Figure out how to check observation count!
    Figure out how to get error rate for each observation count
    Generate knowledge curve for one knowledge concept
    Create GUI that displays observation count for knowledge concept
    Create GUI that allows user to select knowledge concepts displayed on a graph
    Allow user to choose number of observations
    Create GUI for user to see correlation between two knowledge concepts
'''
'''
Import data correctly
Create dataframe of kc, observation count, and correct or incorrect for the observation
    Create the right name mapping
    Split the rows with ~~ into separate rows with their respective observation count

'''
#%%
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
        dataset (str): short name for dataset ('full' or 'stoich')

    Raises:
        ValueError: if dataset doesn't match an allowed type

    Returns:
        oli (DataFrame): OLI log file data, with columns renamed to shortened 
                   forms, and some columns removed (see source) 

    '''
    excel_files = {'full': 'ds_4594_chem1_combined.xlsx',
                   'stoich': 'chem1_data_stoich_only_ds4594.xlsx'}
    
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
        'KC (chemistry_general1-1_9)': 'kc',
        'Opportunity (chemistry_general1-1_9)': 'opp'
        }
    
    # Drop columns that are not in name map and rename columns to be the same
    # As name map
    drop_columns = [x for x in oli.columns.tolist() if x not in name_map]
    oli = oli.drop(columns = drop_columns)
    oli = oli.rename(columns = name_map)
    oli = oli.dropna()
    
    return oli
#%%
oli_df = oli_kc_data('full')
#%%
#print(oli_df['kc'])
# chedck = oli_df['kc']
# for col, kc in oli_df['kc'].items():
#     print(kc)

kc_rm = []
for col,kc in oli_df['kc'].items():
    if('~~' in kc):
        kc_rm.append(kc)
oli_rm = oli_df[oli_df['kc'].isin(kc_rm)]
oli_app = oli_df[~oli_df['kc'].isin(kc_rm)]
oli_rm['kc'] = oli_rm['kc'].str.split("~~")
oli_rm['opp'] = oli_rm['opp'].str.split("~~")
oli_rm = oli_rm.explode(list(('kc','opp')))
oli_kc = oli_app.append(oli_rm)
kc_uni = oli_kc['kc'].unique()

# # Checking for Did I Get This problems
# pnames_digt = set([x for x in oli_kc['pname'] if x.count('digt') and not x.count('pool')])
# # Checking for Quiz problems
# pnames_quiz = set([x for x in oli_kc['pname'] if x.count('quiz') and not x.count('pool')])
# pnames_keep = pnames_digt.copy()
# pnames_keep.update(pnames_quiz)

# oli_kc = oli_kc[oli_kc['pname'].isin(pnames_keep)]


oli_kc = oli_kc[oli_kc['kc'].isin(kc_uni)]
oli_kc['opp'] = pd.to_numeric(oli_kc['opp'])
oli_kc = oli_kc.sort_values('opp', axis=0)

# for kc in kc_uni:
#     df1 = oli_kc[oli_kc['kc'] == kc]
#     for 
# Sum up corrects and incorrects for knowledge concept at certain observation

#%% Generate knowledge curve DataFrame for all knowledge curves
#stu_uni = oli_kc['stud'].dropna().unique()
opp_list = oli_kc['opp'].dropna().unique()
def generate_kc_curve(kc):
    kc_curve = pd.DataFrame()
    df1 = oli_kc[oli_kc['kc'] == kc]
    for opp in opp_list:
        df2 = df1[df1['opp'] == opp]
        # count the total corrects and incorrects and divide incorrect by total
        # save the answer of opportunity number, error rate, and knowledge concept
        incorrects = np.sum(df2['incorrects'])
        total_answers = np.sum(df2['corrects']) + incorrects
        error_rate = incorrects/total_answers * 100
        append_frame = {'kc': kc, 'opp': opp, 'er': error_rate}
        kc_curve = kc_curve.append(append_frame, ignore_index = True)
    return kc_curve

kc_all_curve = pd.DataFrame()
for kc in kc_uni:
    kc_all_curve.append(generate_kc_curve(kc))
#%% Average the error rate for all observations in kc_all_curve
df_er_sum = pd.DataFrame()
for opp in kc_all_curve['opp']:
    df2 = kc_all_curve[kc_all_curve['opp']==opp]
    er_mean = np.mean(kc_all_curve['er'])
    append_frame = {'opp': opp, 'erm': er_mean}
    df_er_sum.append(append_frame, ignore_index = True)

#%% Plot knowledge curve for kc_all_curve
plt.figure(1)
# kc_first = np.mean(score_first, 1)
plt.plot(df_er_sum['opp'],df_er_sum['erm'],'r.')

#%%
#Calculate the error rate for all observations in DataFrame
oli_kc = oli_kc.sort_values('opp', axis=0)
opp_list = oli_kc['opp'].dropna().unique()

df_toter = pd.DataFrame()
for opp in opp_list:
    df1 = oli_kc[oli_kc['opp'] == opp]
    incorrects = np.sum(df1['incorrects'])
    total_answers = np.sum(df1['corrects']) + incorrects
    er_mean = incorrects/total_answers * 100
    append_frame = {'opp':opp, 'erm':er_mean}
    df_toter = df_toter.append(append_frame, ignore_index = True)


#%% Plot knowledge curve for kall observations
plt.figure(1)
# kc_first = np.mean(score_first, 1)
plt.plot(df_toter['opp'],df_toter['erm'],'.r-')
plt.ylabel('Error Rate (%)')
plt.xlabel('Observation')
plt.title('Knowledge Curve')
plt.show()

#%%

'''
Ask Sandy about rating question difficulty, exporting data from Datashop not as text, and how to tell student demographic as well
Open echo > select course > go to reseources or look in xml for question names
Text data can be imported into excel as csv
Three pre student survey, demographics are in the third survey
Separate minority vs Asian and white
Separate first gen and not first gen
IRT Analysis
Most current KC models?
Splitting into demographics, rasch analysis
Look at the quiz performance by demographic,
28-38 on spreadsheet would be good for historical comparison


Rasch analysis for all questions (student average in course on x axis, score on question on y axis)
Check pearson correlation code
Do both digt and quizzes and all question types separately
Show demographics and their respective scores
What is predicted on the knowledge curve?
'''