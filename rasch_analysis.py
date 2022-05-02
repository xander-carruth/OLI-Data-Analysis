"""
@author: Alexander Carruth
"""

import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#%% Get questions dataset
oli_qdf = oli_quest_data('quest')

#%% Get average quiz grade for every student
# Get unique students
students = oli_qdf['stud'].unique()
score_dist = {}
for stud in students:
    # Get dataframe for the current student
    df1 = oli_qdf[oli_qdf['stud']==stud]
    # Get problems that are quiz problems
    pnames_quiz = set([x for x in df1['pname'] if x.count('quiz')])
    # Get dataframe of quiz problems for current student
    df1 = df1[df1['pname'].isin(pnames_quiz)]
    corrects = df1['corrects'].sum()
    total = corrects + df1['incorrects'].sum()
    score_dist[stud] = (corrects/total) * 100

#%% Get scores of every student on every question
# Get unique problem names
pnames = oli_qdf['pname'].unique()
quest_score = {}
for pname in pnames:
    # Get dataframe for current problem name
    df1 = oli_qdf[oli_qdf['pname']==pname]
    stus_dict = {}
    for stud in students:
        # Get dataframe for current student
        df2 = df1[df1['stud']==stud]
        corrects = df2['corrects'].sum()
        total = corrects + df2['incorrects'].sum()
        if(not total == 0):
            stus_dict[stud] = (corrects/total) * 100
        else:
            stus_dict[stud] = math.nan
    quest_score[pname] = stus_dict

#%% Create dataframe of data for plotting
quest_plot = pd.DataFrame()
for quest in quest_score:
    x = []
    y = []
    for stud in quest_score[quest]:
        temp = quest_score[quest]
        if(not math.isnan(temp[stud])):
            x.append(score_dist[stud])
            y.append(temp[stud])
    quest_df = pd.DataFrame({'Question':[quest], 'stu_score':[x], 'q_score':[y]})
    quest_plot = pd.concat([quest_plot, quest_df])

#%% Plot one question
# Function for plotting a default sigmoid over a x and y range
def plot_sig(x, y, beta=1):
    # Get list of values between the two values input into np.linspace
    x_ = np.linspace(-5, 5)
    x_adj = np.linspace(*x)
    sig = 1/(1+np.exp(-x_*beta))
    sig = (sig-sig.min())/sig.max()
    # Adjust the sigmoid to the range of y values
    sig_adj = sig *(y[1]-y[0]) + y[0]
    # Plot the adjusted sigmoid values over the range of x values
    plt.plot(x_adj, sig_adj)

# Get the question name, student scores and question score for q_num
q_num = 80
df_quest = quest_plot.iloc[q_num]
plt_name = df_quest['Question']
stu_score = df_quest['stu_score']
q_score = df_quest['q_score']

plt.plot(stu_score, q_score, '.r')
plt.ylabel('Question Score (%)')
plt.xlabel('Course Score (%)')
x_points = (min(stu_score), max(stu_score))
y_points = (min(q_score), max(q_score))
plot_sig(x_points, y_points, 1)
plt.title(plt_name)
plt.show()
#%% Generate plots for all questions    
for i in range(0, len(quest_plot)):
    df_quest = quest_plot.iloc[i]
    plt_name = df_quest['Question']
    stu_score = df_quest['stu_score']
    q_score = df_quest['q_score']

    plt.plot(stu_score, q_score, '.r')
    plt.ylabel('Question Score (%)')
    plt.xlabel('Course Score (%)')
    x_points = (min(stu_score), max(stu_score))
    y_points = (min(q_score), max(q_score))
    plot_sig(x_points, y_points, 1)
    plt.title(plt_name)
    # Save figures to images folder
    plt.savefig("images/rasch_{quest_name}.png".format(quest_name = plt_name))
    # Clear plot
    plt.clf()

