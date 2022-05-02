# OLI-Data-Analysis
<ins>Notes about the data:</ins>  
The oli_dem_data function runs on data exported as “By Transaction”  
The oli_quest_data function runs on data exported as “By Student-Step”  

<ins>File List:</ins>  
**correlations.py:** Cleaned up version of the file originally sent to me  
**demographics_generator.py:** Holds the basic code for getting demographic data from the datasheets and gives the example of plotting the averaged quiz grades for minority vs non minority and first gen vs non first gen students  
**kc_dem_generator.py:** Plots the minority and non minority students knowledge curves for comparison  
**kc_generator.py:** Allows selection of a knowledge concept for plotting its knowledge curve and also plots the knowledge curve for the average error rate for each observation for all knowledge concepts  
**rasch_analysis.py:** Runs rasch analysis on questions based on the average grade of students on all quizzes but does not fit a sigmoid or run a windowed average
**rasch_analysis_kc_window.py:** Runs rasch analysis on questions based on the average grade of students on the knowledge concepts that apply to the question using a sliding window average of 5 student grades then fits a sigmoid to the data  
**rash_analysis_window.py:** Runs rasch analysis on questions based on the average grade of students on all quizzes using a sliding window average of 5 students grades then fits a sigmoid to the data  
**updated_correlations.py:** Cleaned up version of the file originally sent to me with all the useless parts removed  

<ins>Future Work:</ins>
* Clean up demographics code to make functions for where there is repetition in the code
* Create code to add all the datasets in a folder together so all the separate datasheets can be joined into one dataframe
* Run the code on datasheets with updated knowledge concept maps and make observations about the results
* Figure out how to use the parameters in popt from the rasch analysis to estimate the difficult of a question
* Compare University to community college learning curves to use as a basis for telling if comparing minority knowledge curves provides valuable information
* Run the code on the data returned from the adaptive question study in the Spring 2022 semester and make observations about the results
* Coordinate with LearnSphere about adding useful code modules to their platform
