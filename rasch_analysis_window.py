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

#%% Get the question name, student scores and question score for q_num
q_num = 80
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

# There is no optimal starting parameters but everything other than k is good.
# Play around with adjusting the k value, raising it to make the line steeper and lowering
# it to make the line less steep. If sigmoids are not being generated, raise the k value
# to get a more sigmoid looking shape. (2/3) seems like it will give us roughly what we want
# in terms of slope. The only other way I could think of calculating the optimal values from the
# other sigmoid plot I created was to use the original plot sigmoid function to get x and y values,
# use the curve optimize fit function on the points it uses, and then use these parameters as a start
# for the sigmoid. I will include code for that below, but I don't think it is necessary.
p0 = [(max(q_score)-min(q_score)),np.median(stu_score),(2/3),min(q_score)] # this is an mandatory initial guess

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
#%% DONT USE THIS CODE IN A REGULAR RUN. This code is explained in the comment above
# Function for plotting a default sigmoid over a x and y range
def plot_sig(x, y, beta=1):
    # Get list of values between the two values input into np.linspace
    x_ = np.linspace(-5, 5)
    x_adj = np.linspace(*x)
    sig = 1/(1+np.exp(-x_*beta))
    sig = (sig-sig.min())/sig.max()
    # Adjust the sigmoid to the range of y values
    sig_adj = sig *(y[1]-y[0]) + y[0]
    return x_adj, sig_adj

x_points = (min(stu_score), max(stu_score))
y_points = (min(q_score), max(q_score))

p_x, p_y = plot_sig(x_points, y_points)

# Get default parameters
po = [(max(q_score)-min(q_score)),np.median(stu_score),1,min(q_score)]

# Fit a curve to the plot_sigma default curve
pst, pstc = curve_fit(sigmoid, p_x, p_y, po, method='dogbox')

print("Optimal value array: "+str(pst))

# Get a range of 200 values to plot the sigmoid with the default parameters over
x = np.linspace(min(p_x)-5, max(p_x)+5, 200)
y = sigmoid(x, *pst)

# Show what the sigmoid with default parameters looks like
plt.plot(stu_score, q_score, '.r')
plt.ylabel('Question Score (%)')
plt.xlabel('Course Score (%)')
x_points = (min(stu_score), max(stu_score))
y_points = (min(q_score), max(q_score))
plt.plot(x,y, label='fit')
plt.title("Test Sigmoid")
plt.show()

#%% Generate plot of sigmoid for sliding window with student quiz average as x axis
# Get a range of 200 values to plot the sigmoid with the optimized parameters over
x = np.linspace(min(stu_score)-5, max(stu_score)+5, 200)
y = sigmoid(x, *popt)

plt.plot(stu_score, q_score, 'o', label='data')
plt.plot(x,y, label='fit')
plt.title(plt_name)
plt.ylabel('Question Score (%)')
plt.xlabel('Course Score (%)')
plt.legend(loc='best')
