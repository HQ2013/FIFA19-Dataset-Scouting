# FIFA19-Dataset-Scouting
A Udacity Data Scientist Nanodegree Capstone Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Verification](#verification)

## Installation <a name="installation"></a>

Beyond the Anaconda distribution of Python 3, the following packages need to be installed:
* xgboost

## Project Motivation<a name="motivation"></a>

Being a football fan and video game lover, I'm very interested in football management simulation video games such as Football Manager developed by Sports Interactive and published by Sega. This game will have a new version every year, containing the new game features and the most important, the newly updated player data. All the game lovers wait for the release of the game every year so that they can be the first batch to player the new version.

As a player of such football management sumulation video games, the most important thing is to have a good knowledge of the players in the market. This is also the basic requirement for a soccer scout.

The idea of the project is to do some exploration on FIFA 19 Complete Player Dataset from kaggle, and create a model/APP which servers as a scouting tool for gamer, club manager, and soccer scout as well.

The model/APP may provide advice to the Football Club Manager, no matter in Video Game, or in real Professional football. And it also can be used as a tool for soccer scout to find a suitable player for a club.

The dataset I used in this project is from Kaggle. It contains detailed attributes for every player registered in the latest edition of FIFA 19 database. The url of the dataset is: https://www.kaggle.com/karangadiya/fifa19

## File Descriptions <a name="files"></a>

There are 1 notebooks available here to showcase work related to the above questions. The notebooks is exploratory in searching through the data pertaining to the questions showcased by the notebook title. Markdown cells & comments were used to assist in walking through the thought process for individual steps.

- In working_directory/data:
    * process_data.py: ETL Pipeline Script to process data, it loads and cleaned data by Check & Impute Missing Values, Convert string/date values into numbers, Dealing with categorical features, Normalization of feature "Height", "Weight", Combine Position Rating features with average value
    * data.csv:          Input File, CSV file containing FIFA19 complete dataset
    * cleaned_data.csv: Output File, CSV file containing cleaned data processed by process_data.py
    
- In working_directory/app:
    * templates/*.html: HTML templates for the web app.
    * run.py: Start the Python server for the web app and prepare visualizations.

### Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and WebAPP.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/data.csv data/cleaned_data.csv`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results<a name="results"></a>

About the prediction model, the MAE score 165.68 is about 7% of the average number of Players Value. The model works at acceptable accuracy.

And the App works as expected:

First, the Web APP shows a page that display the bar that to be used to find a list (number = 4) of similar players base on the provided template player.

The lower part of the page are some data visualization of the FIFA19 complete dataset.
![alt text](https://github.com/HQ2013/FIFA19-Dataset-Scouting/blob/master/screenshots/mainpage_screenshot.JPG)
![alt text](https://github.com/HQ2013/FIFA19-Dataset-Scouting/blob/master/screenshots/datavisualization1.JPG)
![alt text](https://github.com/HQ2013/FIFA19-Dataset-Scouting/blob/master/screenshots/datavisualization2.JPG)
![alt text](https://github.com/HQ2013/FIFA19-Dataset-Scouting/blob/master/screenshots/datavisualization3.JPG)

After fill in the template player name and click the button, the page returns four players that most similar to the template player.
![alt text](https://github.com/HQ2013/FIFA19-Dataset-Scouting/blob/master/screenshots/findsimilarplayers.JPG)

## Verification<a name="verification"></a>
To verify the result: four similar players comparing to the provided template player, can use the function plot_radar() in notebook to generate the radar chart of the five players ( 1 template player + 4 similar players ). The radar chart shows that the returned list of players are very similar to the template players based on the six ability values considered.
![alt text](https://github.com/HQ2013/FIFA19-Dataset-Scouting/blob/master/screenshots/verificationresults.JPG)
