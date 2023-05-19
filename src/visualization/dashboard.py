import pandas as pd
from pathlib import Path
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import sys, os
import plotly.graph_objects as go
import numpy as np

def merge_vehicles_make_vehicle_type(dataframe):

    # Merge the vehicles and make dataframes
    vehicle_type_count = dataframe.groupby('make_')['vehicle_type'].value_counts().unstack().reset_index().fillna(0)

    # Count the unique models for each make
    total_models_df = dataframe.groupby('make_')['model.1_'].nunique().reset_index()

    # Rename the columns to make them more descriptive
    total_models_df.columns = ['make_', 'total_models']

    # Merge this with your current dataframe
    master_df_d = pd.merge(dataframe, total_models_df, on='make_', how='left')

    # Then you can use numpy's average function to compute the weighted average
    make_total_avg_score = master_df_d.groupby('make_').apply(lambda x: np.average(x['predicted_co2_rating'], 
                                                                                weights=x['total_models'])).reset_index().rename(columns={0:'weighted_avg_predicted_co2_rating_by_make'})
    make_total_avg_score.sort_values(by='weighted_avg_predicted_co2_rating_by_make', ascending=False, inplace=True)


    # Merge the total models dataframe with the make_total_avg_score dataframe
    weighted_avg_df = pd.merge(total_models_df, make_total_avg_score, on='make_', how='left')

    # sort the dataframe by the weighted average
    weighted_avg_df.sort_values(by='weighted_avg_predicted_co2_rating_by_make', ascending=False, inplace=True)

    final_df = pd.merge(weighted_avg_df, vehicle_type_count, on='make_', how='left').sort_values(by='weighted_avg_predicted_co2_rating_by_make', ascending=False)
    return final_df


def display_top_vehicle_scores(dataframe,  view='top'):

    final_df = merge_vehicles_make_vehicle_type(dataframe)

    if view=='top':
        final_df = final_df.head(10)
        title_str = "Top 10 Vehicle Makes - by their average CO2 rating score"
    else:
        final_df = final_df.tail(10)
        title_str = "Bottom 10 Vehicle Makes - by their average CO2 rating score"

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(final_df.columns),
                    fill_color='rgb(47, 15 , 61)',
                    align='center'),
        # make_	total_models	weighted_avg_predicted_co2_rating_by_make	electric	fuel-only	hybrid
        cells=dict(values=[final_df["make_"],
                        final_df["total_models"],
                        final_df["weighted_avg_predicted_co2_rating_by_make"],
                        final_df["electric"],
                        final_df["fuel-only"],
                        final_df["hybrid"]
                        ],
                                    fill_color='rgb(107, 24, 63)',
                                    align='left'))
    ])

    fig.update_layout(
            title=f"{title_str}",
            font=dict(
                color="white"
            ),
            title_x=0.5,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color='white'
        )

    return fig

def display_table(dataframe, title_str):

    df = dataframe[["model_year", "model.1_", "vehicleclass_", "co2emissions_(g/km)", "predicted_co2_rating"]]
    df.rename(columns = {"model_year": "Year",
                        "model.1_": "Model name",
                        "vehicleclass_": "Class",
                        "co2emissions_(g/km)": "CO2 Emissions (g/km)",
                        "predicted_co2_rating":"Rating" }, inplace=True)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='rgb(47, 15 , 61)',
                    align='center'),
        cells=dict(values=[df["Year"],
                        df["Model name"],
                        df["Class"],
                        df["CO2 Emissions (g/km)"],
                        df["Rating"]],
                                    fill_color='rgb(107, 24, 63)',
                                    align='left'))
    ])

    fig.update_layout(
            #title=f"{title_str}.<br><sup>On a scale from 1 (worst) to 10 (best)</sup>",
            font=dict(
                color="white"
            ),
            title_x=0.5,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color='white'
        )

    return fig

def generate_bar_chart_with_models(dataframe):

    model_average_co2 = dataframe.groupby(["model.1_",'model_year',"predicted_co2_rating"])['co2emissions_(g/km)'].mean().reset_index().sort_values(by='co2emissions_(g/km)', \
                                                                                           ascending=False).rename(columns={"co2emissions_(g/km)":"Average CO2 emissions",
                                                                                                                           "model.1_":"Model name",
                                                                                                                               'model_year': "Model year",
                                                                                                                               "predicted_co2_rating": "CO2 rating"})
    model_average_co2['Model year'] = model_average_co2['Model year'].astype(str)
    year_min = model_average_co2['Model year'].min() 
    year_max = model_average_co2['Model year'].max() 
    model_average_co2 = model_average_co2.sort_values(
                        by="Model year", 
                        ascending=False)
    
    make_name = dataframe['make_'].unique()[0]

    fig = px.histogram(data_frame=model_average_co2, 
                #x='Model name', 
                x='CO2 rating',
                color_discrete_sequence= px.colors.sequential.matter,
                color='Model year',
                title=f"Distribution of CO2 rating({year_min} - {year_max}) by make ({make_name.upper()})")

    fig.update_layout(
            title_x=0.5,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
    fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})

    return fig


def generate_count_plot(attribute, vehicle_type, dataframe):
    """
    This function generates a histogram of an attribute for 
    vehicle dataframes
    """

    try:
        dataframe = dataframe.sort_values(
                        by="model_year", 
                        ascending=False)

        
        year_min = dataframe['model_year'].min() 
        year_max = dataframe['model_year'].max() 
        rename_attr = attribute.replace("_"," ").capitalize()
        fig = px.histogram(data_frame=dataframe,
                        x=attribute, labels={attribute:rename_attr},
                        color_discrete_sequence= px.colors.sequential.matter,
                        color='model_year',
                        title=f"Frequency of {rename_attr} ({year_min} - {year_max}), {vehicle_type}")
        fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
        fig.update_layout(
            title_x=0.5,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )


        return fig
    except KeyError:
        print("Key not found. Make sure that 'vehicle_type' is in ['electric', 'hybrid', 'fuel-only']")
    except ValueError:
        print("Dimension is not valid. ")
        
        
# Set data read path
clean_data = os.path.abspath(os.path.join(os.getcwd(), 'data', 'predicted-data'))

# Assign variables 
file_name_2022_1995 = "vehicle_data_with_clusters.csv"
pure_electric = "predicted_co2_rating_electric.csv"
hybric_vehicle = "predicted_co2_rating_hybrid.csv"

# Read data
master_df = pd.read_csv(Path(clean_data,f'{file_name_2022_1995}'))
electric_df = pd.read_csv(Path(clean_data,f'{pure_electric}'))
hybrid_df = pd.read_csv(Path(clean_data,f'{hybric_vehicle}'))

dataframe_dictionary = {"electric": electric_df,
                       "hybrid": hybrid_df,
                       "fuel-only": master_df}

all_makes = master_df['make_'].unique()

year_min = master_df['model_year'].min() 
year_max = master_df['model_year'].max() 

# App section        
        

# Stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



# Intialize app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="Fuel consumption of vehicles dashboard")
server = app.server

colors = {
    'background': '#003f5c',
    'text': 'white'
}

card_text_style = {
    'textAlign' : 'left',
    'color' : 'black',
    'backgroundColor': colors['background']
}

header_style = {'textAlign' : 'center','color':"white"}

header_menu_style = {'textAlign' : 'left','color':"white","maxWidth": "80em", "fontSize":"16px"}

# ----------------------------------------------------------------------------------#
text_card = dbc.Card(
    dbc.CardBody(
        [
            html.H2("Vehicle fuel consumption dashboard", className="card-title",style=header_style),
            html.H5("Fuel-only, hybrid and electric vehicles", className="card-subtitle", style=header_style),
            
        ]
    ),style={'backgroundColor': colors['background']}
)

footer_card = dbc.Card(
    dbc.CardBody(
        [
            html.P(
               "To help consumers in Canada find fuel-efficient vehicles, the Government of Canada released a fuel consumption ratings search tool.\
                In it, they provide users the ability to search vehicles by model, class and make and obtain information on the fuel consumption of \
                various vehicles in three settings: city, highway and combined. Vehicles undergo a 5-cycle fuel consumption testing in each of these \
                settings, where the vehicle's CO2 emissions are estimated. Additionally, they provide\
                access through their open data portal as part of the Open Government License Canada. ", className="card-text",style=header_menu_style,),
            html.P(
                "Whereas this tool allows consumers to obtain information via the website, it would be great if it could also show data insights on \
               scores from different manufactures, as well as trends for which vehicles in Canada and US are more popular and in turn what this means \
                for fuel emissions and air quality. This dashboard's goal is to provide insights into different vehicle's fuel consumption under the 5-cycle\
                    testing, provide insights into consumer trends in Canada, and forecast CO2 emissions from vehicles.", className="card-text",style=header_menu_style,
            ),
            dbc.CardLink(
                "Learn about fuel consumption testing", 
            href="https://www.nrcan.gc.ca/energy-efficiency/transportation-alternative-fuels/fuel-consumption-guide/understanding-fuel-consumption-ratings/fuel-consumption-testing/21008",
            style={'textAlign' : 'center'}),
            html.P(""),
            dbc.CardLink("Data source",
                href="https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64"
                )
        ]
    ),style={'backgroundColor': colors['background']}
)

menu_card = dbc.Card(
    dbc.CardBody(
        [
           html.Div([
                html.Div([
                    html.P("Type or select a make", style=header_menu_style),
                    dcc.Dropdown(
                        id='select-make',
                        options=all_makes,
                        value= 'bmw',
                        style={'backgroundColor':"white"}),
                ],  className="two columns"),

            ], className="row"),

        ]
    ), style={'backgroundColor': colors['background']}
)

menu_card2 = dbc.Card(
    dbc.CardBody(
        [
           html.Div([
                html.Div([
                    html.P("Type or select a make", style=header_menu_style),
                    dcc.Dropdown(
                        id='select-make-2',
                        options=all_makes,
                        value= 'bmw',
                        style={'backgroundColor':"white"}),
                ],  className="two columns"),

            ], className="row"),

        ]
    ), style={'backgroundColor': colors['background']}
)

plots_card = dbc.Card(

    dbc.CardBody([ 
        html.Div([
                    html.Div([
                            dcc.Graph(id='graph-dist_fuel')
                        ], className="six columns"),

                    
                    html.Div([
                            dcc.Graph(id='graph-scatter-box')
                        ], className="six columns"),
                    
                    
                ], className="row"),

        html.Div([
                    html.Div([
                            dcc.Graph(id='table-of-scores')
                        ], className="six columns", ),

                 

                html.Div([
                            dcc.Graph(id='graph-models-released')
                        ], className="six columns"),
                    
                ], className="row"),
    ]), style={'backgroundColor': colors['background']}
)

plots_card_2 = dbc.Card(

    dbc.CardBody([ 
        html.Div([
                   
                    
                    html.Div([
                           dcc.Graph(id='top_10_cars')
                        ], className="six columns"),

                    html.Div([
                           dcc.Graph(id='bottom_10_cars')
                        ], className="six columns"),
                    
                    
                ], className="row"),

        html.Div([
                        html.Div([
                html.Div([
                    html.P("Type or select a make", style=header_menu_style),
                    dcc.Dropdown(
                        id='select-make-2',
                        options=all_makes,
                        value= 'bmw',
                        style={'backgroundColor':"white"}),
                ],  className="two columns"),

            ], className="row"),
        ]),

        html.Div([

                    html.Div([
                            dcc.Graph(id='graph-vehicle-type-count')
                        ], className="six columns", ),

                 

                html.Div([
                            dcc.Graph(id='vehicle-type-bar')
                        ], className="six columns"),
                    
                ], className="row"),
    ]), style={'backgroundColor': colors['background']}
)

cards = dbc.Container([ 
    dbc.Row(
    [   
        dbc.Col(menu_card, width='auto'),
        dbc.Col(plots_card, width='auto'),
    ]
)
], fluid=True,style={'backgroundColor': "black"})


cards2 = dbc.Container([ 
    dbc.Row(
    [   
        dbc.Col(plots_card_2, width='auto'),
    ]
)
], fluid=True,style={'backgroundColor': "black"})


app.layout = html.Div([
    dbc.Col(text_card, width='auto'),
    dcc.Tabs([
        dcc.Tab(label='Insights about vehicle makes', children=[
            html.Div(
                children=[cards],
                style={'backgroundColor': "black"}
            )
        ]),
        dcc.Tab(label='Insights about vehicles by their type: electric, fuel based and hybrid', children=[
            html.Div(
                children=[cards2],
                style={'backgroundColor': "black"}
            )
        ]),
        # dcc.Tab(label='Supervised learning results and clustering analysis', children=[
        #     dcc.Graph(
        #         figure={
        #             'data': [
        #                 {'x': [1, 2, 3], 'y': [2, 4, 3],
        #                     'type': 'bar', 'name': 'SF'},
        #                 {'x': [1, 2, 3], 'y': [5, 4, 3],
        #                  'type': 'bar', 'name': u'Montréal'},
        #             ]
        #         }
        #     )
        # ]),
    ]),
    dbc.Col(footer_card, width='auto')
])

@app.callback(
    Output('graph-dist_fuel', 'figure'),
    Input('select-make', 'value'))
def show_avg_predicted_co2_rating_by_make(value):
    filtered_df = master_df[master_df['make_'] == value]


    viz_table = pd.DataFrame(filtered_df.groupby(["make_",'model_year'])['predicted_co2_rating'].mean()).reset_index().rename(columns={'predicted_co2_rating':'avg_predicted_co2_rating_by_make'})

    fig = px.line(viz_table, x='model_year', 
                  y='avg_predicted_co2_rating_by_make', 
                  title=f'Average Predicted CO2 Rating by Make ({value.upper()})',
                  color_discrete_sequence= px.colors.sequential.matter,)

    fig.update_layout(
        title_x=0.5,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis={'categoryorder': 'total descending', 'showgrid': False},
        yaxis={'showgrid': False}
    )

    return fig

@app.callback(
    Output('graph-scatter-box', 'figure'),
    Input('select-make', 'value'))
def show_predicted_co2_rating_by_model(value):
    filtered_df = master_df[master_df['make_'] == value]

    # create line chart
    line_fig = px.scatter(filtered_df, 
                        x='model_year', 
                        y='predicted_co2_rating', 
                        title=f'Predicted CO2 ratings over time by make {value.upper()} and model (hover for model name)',
                        labels={'model_year':'Model Year', 'co2emissions_(g/km)':'CO2 Emissions (g/km)'}, 
                        hover_name='model.1_',
                        color_discrete_sequence= ['rgb(253, 237, 176)',
                                                'rgb(195, 56, 90)',
                                                'rgb(47, 15, 61)'],
                        color='vehicle_type')
    
    line_fig.update_layout(
        title_x=0.5,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis={'categoryorder': 'total descending', 'showgrid': False},
        yaxis={'showgrid': False}
    )
    return line_fig

@app.callback(
    Output('graph-models-released', 'figure'),
    [
    Input('select-make', 'value'),])
def update_frequency_chart(make):
    vehicle_type = "fuel-only"
    sel_dataframe = dataframe_dictionary[vehicle_type]
    dataframe = sel_dataframe[(sel_dataframe['make_']==make) & 
                              (sel_dataframe['model_year']>=1995)  & 
                              (sel_dataframe['predicted_co2_rating']>=1) ]
    fig0 = generate_bar_chart_with_models(dataframe)
    return fig0

@app.callback(
    Output('table-of-scores', 'figure'),
    [
    Input('select-make', 'value')])
def update_frequency_chart(make):
    vehicle_type = "fuel-only"
    sel_dataframe = dataframe_dictionary[vehicle_type]
    dataframe = sel_dataframe[(sel_dataframe['make_']==make) & 
                              (sel_dataframe['model_year']>=1995)  &
                               (sel_dataframe['predicted_co2_rating']>=7) ]
    title_str = f"Models with a CO2 rating of 7 or above for make {make.upper()}"
    fig0 = display_table(dataframe, title_str)
    return fig0

@app.callback(
    Output('graph-vehicle-type-count', 'figure'),
    [Input('select-make-2', 'value')])
def plot_vehicle_type_count(make):
    total_models_df = merge_vehicles_make_vehicle_type(master_df)
    df_filtered = total_models_df[total_models_df['make_'] == make]
    
    fig = go.Figure(data=[
        go.Bar(name='Electric', x=df_filtered['make_'], y=df_filtered['electric'], marker_color='rgb(26, 118, 255)'),
        go.Bar(name='Fuel-only', x=df_filtered['make_'], y=df_filtered['fuel-only'], marker_color='rgb(55, 83, 109)'),
        go.Bar(name='Hybrid', x=df_filtered['make_'], y=df_filtered['hybrid'], marker_color='rgb(26, 188, 156)')
    ])

    # Change the bar mode
    fig.update_layout(
        title=f'Number of Vehicle Types for {make.upper()}',
        title_x=0.5,
        xaxis=dict(title='Make'),
        yaxis=dict(title='Number of Vehicles'),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )

    return fig

@app.callback(
    Output('top_10_cars', 'figure'),
    Input('select-make-2', 'value')
    )
def show_top_ten_cars(value):
    figure = display_top_vehicle_scores(master_df,  view='top')

    return figure

@app.callback(
    Output('bottom_10_cars', 'figure'),
    Input('select-make-2', 'value')
    )
def show_bottom_ten_cars(value):
    figure = display_top_vehicle_scores(master_df, view='bottom')

    return figure

@app.callback(
    Output('vehicle-type-bar', 'figure'),
    Input('select-make-2', 'value')
    )
def show_all_cars(value):
    vehicle_type_count = master_df['vehicle_type'].value_counts().reset_index()
    vehicle_type_count.columns = ['vehicle_type', 'count']

    figure = px.bar(vehicle_type_count, x='vehicle_type', y='count', title='Vehicle Type Count')
    # Change the bar mode
    figure.update_layout(
        title=f'Number of Vehicle Types in the entire dataset',
        title_x=0.5,
        xaxis=dict(title='Make'),
        yaxis=dict(title='Number of Vehicles'),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )

    return figure

if __name__ == '__main__':  
    
    app.run_server(debug=True) 