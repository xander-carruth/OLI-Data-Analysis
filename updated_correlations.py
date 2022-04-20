#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:13:10 2021

@author: yaron
"""
import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def oli_data(dataset):
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
        'Problem Hierarchy': 'phier',
        'Problem Name' : 'pname',
        'Problem View' : 'pview',
        'Step Name': 'sname',
        'Step Start Time': 'sstart',
        'Step End Time': 'send',
        'Step Duration (sec)' : 'sdur',
        'Error Step Duration (sec)': 'edur',
        'First Attempt': 'first',
        'Incorrects': 'incorrects',
        'Hints': 'hints',
        'Corrects': 'corrects',
        'Condition': 'condition',
        'KC (Single-KC)':  'kcsingle',
        'Opportunity (Single-KC)': 'opp',
        'KC (Unique-step)': 'kcstep',
        'Opportunity (Unique-KC)': 'oppstep',
        'KC (chemistry_general1-1_9)': 'kc'
        }
    
    # Drop columns that are not in name map and rename columns to be the same
    # As name map
    drop_columns = [x for x in oli.columns.tolist() if x not in name_map]
    oli = oli.drop(columns = drop_columns)
    oli = oli.rename(columns = name_map)
    
    return oli

def list_unique(oli, column, print_length=False):
    '''
    List of unique values in column, with nan removed
    Args:
        oli (DataFrame): DataFrame
        field (str): column name 
        print_length (bool): print to console

    Returns:
        List: unique values in that column, with nan removed

    '''
    res = oli[column].dropna().unique().tolist()
    if print_length:
        print('unique',column, len(res))
    return res

vals_full = dict()
oli_full = oli_data('full')
# Drop nonunique values
for col in oli_full.columns:
    vals_full[col] = list_unique(oli_full, col,True)    

#%% Filter to include only digt and quiz
# Checking for Did I Get This problems
pnames_digt = set([x for x in vals_full['pname'] if x.count('digt') and not x.count('pool')])
# Checking for Quiz problems
pnames_quiz = set([x for x in vals_full['pname'] if x.count('quiz') and not x.count('pool')])
pnames_keep = pnames_digt.copy()
pnames_keep.update(pnames_quiz)
#pnames_keep = pnames_quiz.copy()
print('digt ',len(pnames_digt),'quiz', len(pnames_quiz), 'keeping',len(pnames_keep))

# Check for questions with knowledge concepts
kc_keep = [x for x in vals_full['kc'] if x.count('~~') == 0]
print('number of kc excluding joined',len(kc_keep))
# Create dataframe of oli entries that are DIGT and Quiz
oli_temp = oli_full[oli_full['pname'].isin(pnames_keep)]
print('numer of log entries for retained problems',len(oli_temp))
# Create dataframe of oli entries that have knowledge concepts from temporary
# dataframe
oli = oli_temp[oli_temp['kc'].isin(kc_keep)]
print('number of log entries for single kc',len(oli))
print('number of log entries',len(oli.values))

# Store unique entries in vals dictionary
vals = dict()
for col in oli.columns:
    vals[col] = list_unique(oli, col,True)    

#%% Create numpy arrays with dimension kc x student, with performance of that
# student on the corresponding kc. This is done two ways
#   - score_first  : based on first interaction with a problem
#   - score_correct : based on number of correct/incorrect entries for a problem
# if number of interactions with that kc < min_number_for_inclusion, score = NaN 
#
# we have a list of student names in an array vals['stud'][istud] = stud_name
# this allows the inverse:  stud_map[stud_name] = istud

# Creates a map of student to their student number
stud_map = {x:i for i,x in enumerate(vals['stud'])}
# Fraction of the first question that was gotten correct
score_first = np.empty([len(kc_keep), len(stud_map)])
# Fraction of the total questions that was gotten correct
score_correct = np.empty([len(kc_keep), len(stud_map)])
score_first[:] = np.nan 
score_correct[:] = np.nan
# Total number of first answers
n_first = np.zeros([len(kc_keep), len(stud_map)])
# Total number of answers
n_corr= np.zeros([len(kc_keep), len(stud_map)])

# Threshold of entries for a student to be included for a knowledge concept
min_number_for_inclusion = 3

for ikc,kc in enumerate(kc_keep):
    # Get the data of knowledge concepts in kc_keep
    df1 = oli[ oli['kc'] == kc ]
    # Get the unique list of students in df1
    stud1 = list_unique(df1,'stud',False)
    for stud in stud1:
        # Get the student number
        istud =  stud_map[stud]
        # Get the dataframe of questions just for the student
        df2 = df1[ df1['stud'] == stud]

        # first
        first = df2.value_counts('first')
        needs = ['correct', 'incorrect', 'hint']
        # Make sure every need is in first otherwise set to 0
        for need in needs:
            if need not in first:
                first[need] = 0
        # Calculate total firsts
        tot_firsts = first['correct'] + first['incorrect'] + first['hint']
        # Check if total firsts is above threshold for inclusion otherwise set
        # to nan
        if tot_firsts >= min_number_for_inclusion:
            frac_first = float(first['correct'])/float(tot_firsts)
        else:
            frac_first = np.nan

        # correct / incorrect
        corrects = df2['corrects'] 
        incorrects = df2['incorrects']
        n_correct = np.sum(corrects)
        n_incorrect = np.sum(incorrects)
        tot_corrects = n_correct + n_incorrect
        # Check if the total answers for a knowledge concept is above threshold
        # for inclusion, otherwise set to nan
        if tot_corrects >= min_number_for_inclusion:
            frac_correct = float(n_correct)/float(tot_corrects)
        else:
            frac_correct = np.nan

        n_first[ikc,istud] = tot_firsts
        n_corr[ikc,istud] = tot_corrects
        score_first[ikc,istud] = frac_first
        score_correct[ikc,istud] = frac_correct

#%% Print out cross of student by kc and scores with sufficient attempts
all_first = score_first.flatten()
print('total cross of student by kc', len(all_first))
print('scores with sufficient attempts for first', len([x for x in all_first if not np.isnan(x)]))
all_correct = score_correct.flatten()
print('scores with sufficient attempts for correct', len([x for x in all_correct if not np.isnan(x)]))

#%% Filter out nan values from arrays
def filter_mean_nans(filter_arr):
    # Mask array my filling removing all nan values
    temp = np.ma.masked_array(filter_arr,np.isnan(filter_arr))
    result = np.mean(temp, axis=1)
    return result.filled(np.nan)

#%% Attempt to see if score_first and score_correct give similar results
plt.figure(1)
# kc_first = np.mean(score_first, 1)
kc_first = filter_mean_nans(score_first)
kc_corr = filter_mean_nans(score_correct)
plt.plot(kc_first,kc_corr,'r.')
plt.figure(2)
tot_first = filter_mean_nans(n_first)
tot_corr = filter_mean_nans(n_corr)
plt.plot(tot_first,tot_corr,'b.')
#%% Correlations between KCs
#  - for all pairs of kcs
#       - find list of students that have a score for both kcs
#       - get pearson correlation, and p value
import scipy
scores = score_first
nstud = scores.shape[1]
pearson = list()
highly_corr = list()
for i1,kc1 in enumerate(kc_keep):
    # list of scores on kc1 for all students
    score_kc1 = scores[i1,:]
    for i2,kc2 in enumerate(kc_keep):
        # Checks if two concepts have already been compared
        if i1 >= i2:
            continue
        # list of scores on kc2 for all students
        score_kc2 = scores[i2,:]
        # find students that have a score on both kcs, filter out nans
        ikeep = [istud for istud in range(nstud) if 
                       (not np.isnan(score_kc1[istud])) 
                   and (not np.isnan(score_kc2[istud]))]
        # evaluate correlation only if > 3 students have scores for both kcs
        if (len(ikeep) > 3):
            score_kc1_keep = score_kc1[ikeep]
            score_kc2_keep = score_kc2[ikeep]
            # Calculate pearson correlation
            res = scipy.stats.pearsonr(score_kc1_keep, score_kc2_keep)
            # save correlations if p value is small (this is way small!?)
            if res[1] < 1.0e-10:
                highly_corr.append([score_kc1_keep, score_kc2_keep,res[0],res[1]])
            # Save all pearson correlations
            pearson.append((kc1,kc2,res[0],res[1]))
            
#%% write out kcs that are most correlated, or anticorrelated
Rsig = [x for x in pearson if x[3] < 0.01]
Rvals = [x[2] for x in Rsig]
isort = np.argsort(-1.0 * np.abs(Rvals))
Rsorted_positive = [Rsig[i1] for i1 in isort if Rsig[i1][2] > 0.0]
Rsorted_negative = [Rsig[i1] for i1 in isort if Rsig[i1][2] < 0.0]

with open('positive_correlations.csv','w') as out1:
    out1.write('Rpearson,pvalue,kc1,kc2')
    for vals in Rsorted_positive:
        out1.write(str(vals[2])+','+str(vals[3])+','+str(vals[0])+','+str(vals[1])+'\n')
with open('negative_correlations.csv','w') as out1:
    for vals in Rsorted_negative:
        out1.write(str(vals[2])+','+str(vals[3])+','+str(vals[0])+','+str(vals[1])+'\n')


#%% Run PCA on the fraction of first answers gotten correct for knowledge concepts
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
PCA_score_first = np.ma.masked_array(score_first,np.isnan(score_first))
# Run PCA on the transposed scores
pca.fit(PCA_score_first.T)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

comp = pca.components_

#%% Create a dictionary of unique values in each column for each student in filtered data frame
#unique_bystudent = dict()
#for stud in vals['stud']:
#    df1 = oli[ oli['stud'] == stud]
#    unique_bystudent[stud] = {col: list_unique(df1, col, False) for col in df1.columns}

#%% Find the number of entries per student for a knowledge concept
pname1 = "multi_mole_digt"
# Create a dataframe of every problem in the filtered data frame with pname1
df1 = oli[ oli['pname'] == pname1]
# Filter for only unique students
studs = list_unique(df1,'stud',False)
print(pname1,'has', len(oli[oli['pname'] == pname1]), 'entries', 
      'from',len(studs),'students')
# Dictionary of number of entries by student on the pname1 concept
n_by_stud = dict()
for stud in studs:
    df2 = df1[ oli['stud'] == stud]
    n_by_stud[stud] = len(df2)
    # This does not do anything
    if len(df2) > 5:
        pass
for x in n_by_stud.values():
    print(x)
plt.figure(3)
plt.hist([x for x in n_by_stud.values()])
plt.title('entries per student for: '+pname1)
