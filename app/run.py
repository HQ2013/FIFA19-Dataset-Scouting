import json
import plotly
import pandas as pd
import numpy  as np
from math  import sqrt
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from plotly.graph_objs import Heatmap
from plotly.graph_objs import Scatterpolar
from sklearn.externals import joblib

app = Flask(__name__)

# load data
cleaned_df = pd.read_csv('../data/cleaned_data.csv')
#feature list for graph five: FIFA 19 dataset important feature correlation heatmap display
feature_list = ['Age','Overall','Potential','International Reputation','PAC','SHO','PAS','DRI','DEF','PHY',
                'DIV','HAN','KIC','REF','SPD','POS','Value_Number_K','Wage_Number_K','ReleaseClause_Number_K']

################################################################################
#Use this function to measure the similarity by Pearson correlation coefficient.
def sim_pearson(data, feature_to_compare, player1, player2):

    # Find the number of elements
    n = len(feature_to_compare)
    
    sum_1    = 0
    sum_2    = 0
    sum_1_sq = 0
    sum_2_sq = 0
    p_sum    = 0

    for it in feature_to_compare:
        value1 = data.loc[data['Name']==player1, it].values[0]
        value2 = data.loc[data['Name']==player2, it].values[0]
        
        # Add up all the preferences
        sum_1 += value1
        sum_2 += value2

        # Sum up the squares
        sum_1_sq += pow(value1,2)
        sum_2_sq += pow(value2,2)

        # Sum up the products
        p_sum += value1 * value2

    # Calculate Pearson score
    num = p_sum - (sum_1 * sum_2/n)
    den = sqrt((sum_1_sq - pow(sum_1, 2)/n) * (sum_2_sq - pow(sum_2, 2)/n))

    if den == 0:
        return 0

    r = num/den

    return r

################################################################################
def most_similar(data, player, n=4):
    """
    This function find the most similar player to the provided player
    
    Parameter:
    data:   input dataset
    player: a player name as the Recommendation template

    return:
    another player who is most similar to the Recommendation template player
    """
    scores = [ ]
    
    player_position = data.loc[data['Name']==player, 'Position'].values[0]
    player_overall  = data.loc[data['Name']==player, 'Overall'].values[0]
    
    if player_position == "GK":
        feature_to_compare = ['DIV', 'HAN', 'KIC', 'REF', 'SPD', 'POS']
    else:
        feature_to_compare = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']
    
    for candidate in data['Name']:
        if candidate != player:
            candidate_position = data.loc[data['Name']==candidate, 'Position'].values[0]
            candidate_overall  = data.loc[data['Name']==candidate, 'Overall'].values[0]
            if ((candidate_position == player_position) and (candidate_overall > player_overall*0.8)):
                scores += [(sim_pearson(data, feature_to_compare, player, candidate), candidate)]
    
    # Sort the list so the highest scores appear at the top scores.sort( )
    scores.sort( )
    scores.reverse( )
    return scores[0:n]



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # create visuals
    graphs = [
        # Graph 1: FIFA19 Players' Age Distribution 
        {
            'data': [
                Pie(
                    labels = cleaned_df.Age.value_counts().to_frame().index,
                    values = cleaned_df.Age.value_counts().to_frame().Age
                )
            ],
            'layout': { 'title': 'FIFA 19 Players Age Distribution' }
        },
        
        # Graph 2: FIFA19 Players' Position Distribution 
        {
            'data': [
                Pie(
                    labels = cleaned_df.Position.value_counts().to_frame().index,
                    values = cleaned_df.Position.value_counts().to_frame().Position
                )
            ],
            'layout': { 'title': 'FIFA 19 Players Position Distribution' }
        },
        
        # Graph 3: FIFA 19 Top 20 clubs with highest total player market value
        {
            'data': [
                Bar(
                    x=cleaned_df.groupby("Club")["Value_Number_K"].sum().sort_values(ascending=False).head(20).to_frame().index,
                    y=cleaned_df.groupby("Club")["Value_Number_K"].sum().sort_values(ascending=False).head(20).to_frame().Value_Number_K
                )
            ],

            'layout': {
                'title': 'FIFA 19 Top 20 clubs with highest total player market value',
                'yaxis': { 'title': "Total Market Value of the Players (in thousands)" },
                'xaxis': { 'title': "Club" }
            }
        },

        # Graph 4: FIFA 19 Top 20 clubs with highest average wage
        {
            'data': [
                Bar(
                    x=cleaned_df.groupby("Club")["Wage_Number_K"].mean().sort_values(ascending=False).head(20).to_frame().index,
                    y=cleaned_df.groupby("Club")["Wage_Number_K"].mean().sort_values(ascending=False).head(20).to_frame().Wage_Number_K
                )
            ],

            'layout': {
                'title': 'FIFA 19 Top 20 clubs with highest average wage',
                'yaxis': { 'title': "Average wage of the Players (in thousands)" },
                'xaxis': { 'title': "Club" }
            }
        },

        # Graph 5: Average value for each position
        {
            'data': [
                Bar(
                    x=cleaned_df.groupby("Position")["Value_Number_K"].mean().sort_values(ascending=False).to_frame().index,
                    y=cleaned_df.groupby("Position")["Value_Number_K"].mean().sort_values(ascending=False).to_frame().Value_Number_K
                )
            ],

            'layout': {
                'title': 'Average value for each position',
                'yaxis': { 'title': "Average value for each position (in thousands)" },
                'xaxis': { 'title': "Position" }
            }
        },

        # Graph 6: Average wage for each position
        {
            'data': [
                Bar(
                    x=cleaned_df.groupby("Position")["Wage_Number_K"].mean().sort_values(ascending=False).to_frame().index,
                    y=cleaned_df.groupby("Position")["Wage_Number_K"].mean().sort_values(ascending=False).to_frame().Wage_Number_K
                )
            ],

            'layout': {
                'title': 'Average wage for each position',
                'yaxis': { 'title': "Average wage for each position (in thousands)" },
                'xaxis': { 'title': "Position" }
            }
        },

        # Graph 7: Average value for each age
        {
            'data': [
                Bar(
                    x=cleaned_df.groupby("Age")["Value_Number_K"].mean().sort_values(ascending=False).to_frame().index,
                    y=cleaned_df.groupby("Age")["Value_Number_K"].mean().sort_values(ascending=False).to_frame().Value_Number_K
                )
            ],

            'layout': {
                'title': 'Average value for each age',
                'yaxis': { 'title': "Average value for each age (in thousands)" },
                'xaxis': { 'title': "Age" }
            }
        },

        # Graph 8: Average wage for each age
        {
            'data': [
                Bar(
                    x=cleaned_df.groupby("Age")["Wage_Number_K"].mean().sort_values(ascending=False).to_frame().index,
                    y=cleaned_df.groupby("Age")["Wage_Number_K"].mean().sort_values(ascending=False).to_frame().Wage_Number_K
                )
            ],

            'layout': {
                'title': 'Average wage for each age',
                'yaxis': { 'title': "Average wage for each Age (in thousands)" },
                'xaxis': { 'title': "Age" }
            }
        },
        
        # Graph 9: FIFA 19 dataset important feature correlation heatmap display
        {
            'data': [
                Heatmap(
                    z=np.array(cleaned_df[feature_list].corr()),
                    x=feature_list,
                    y=feature_list,
                    colorscale='Jet'
                )
            ],

            'layout': { 'title': 'FIFA19 dataset important feature correlation heatmap display' }
        },
        

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user input, a player's name and displays similar players
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use most_similar() function to find the most similar players
    Results = most_similar(cleaned_df, query)

    # create radar visuals
    # Basic Abailties
    cols_nGK  = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']
    cols_GK   = ['DIV', 'HAN', 'KIC', 'REF', 'SPD', 'POS']
    position_type = cleaned_df.loc[cleaned_df['Name'] == query, 'Position'].values[0]
    
    if position_type == "GK":
        cols = cols_GK
    else:
        cols = cols_nGK

    graphs = [
        # Graph 1: FIFA19 Players' Age Distribution 
        {
            'data': [
                Scatterpolar(
                	r = [
                	cleaned_df.loc[cleaned_df['Name'] == query, cols[0]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == query, cols[1]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == query, cols[2]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == query, cols[3]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == query, cols[4]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == query, cols[5]].values[0]
                	],
                    theta = cols,
                    name  = query
                ),
                Scatterpolar(
                	r = [
                	cleaned_df.loc[cleaned_df['Name'] == Results[0][1], cols[0]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[0][1], cols[1]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[0][1], cols[2]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[0][1], cols[3]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[0][1], cols[4]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[0][1], cols[5]].values[0]
                	],
                    theta = cols,
                    name  = Results[0][1]
                ),
                Scatterpolar(
                	r = [
                	cleaned_df.loc[cleaned_df['Name'] == Results[1][1], cols[0]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[1][1], cols[1]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[1][1], cols[2]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[1][1], cols[3]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[1][1], cols[4]].values[0],
                	cleaned_df.loc[cleaned_df['Name'] == Results[1][1], cols[5]].values[0]
                	],
                    theta = cols,
                    name  = Results[1][1]
                ),
            ],
            'layout': {
            	'title': 'FIFA 19 Players Age Distribution',
            	'polar': dict(radialaxis = dict(visible = True,range = [0, 50])),
            	'showlegend': { True }
            }
        },
    ]

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        TPlayer1=Results[0][1],
        TPlayer2=Results[1][1],
        TPlayer3=Results[2][1],
        TPlayer4=Results[3][1],
    )


################################################################################
def main():
    #app.run(host='0.0.0.0', port=3001, debug=True)
    app.run()

################################################################################
if __name__ == '__main__':
    main()