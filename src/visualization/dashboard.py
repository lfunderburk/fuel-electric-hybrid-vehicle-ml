import pandas as pd
from pathlib import Path
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import sys, os
import plotly.graph_objects as go


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
            title=f"{title_str}.<br><sup>On a scale from 1 (worst) to 10 (best)</sup>",
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

    fig = px.histogram(data_frame=model_average_co2, 
                #x='Model name', 
                x='CO2 rating',
                color_discrete_sequence= px.colors.sequential.matter,
                color='Model year',
                title=f"Distribution of CO2 rating({year_min} - {year_max}) by make")

    fig.update_layout(
            title_x=0.5,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
    fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})

    return fig

def generate_scatter_or_box(dataframe, y_var):
    
    if dataframe[y_var].dtype=='float64' or dataframe[y_var].dtype=='int64':
        if "fuelconsumption" in y_var:
            type_ = y_var.split("_")[-1]
            label = "Fuel consumption " + type_ 
        else:
            label = y_var.replace("_"," ").capitalize()
        fig = px.scatter(data_frame=dataframe, 
               y=y_var,
               x='model_year', 
               size='co2emissions_(g/km)',
                color='predicted_co2_rating',
               color_discrete_sequence= px.colors.sequential.matter,
               hover_data=['model.1_', 'model_year', 'vehicleclass_'],
             labels={y_var: label, 'model_year': "Year of manufacture"},
             title= f"{label} over time")
    else:
        label = y_var.replace("_"," ").capitalize()
        fig = px.scatter(data_frame=dataframe, 
               y=y_var,
               x='model_year',
               size='co2emissions_(g/km)', 
               color='predicted_co2_rating',
               color_discrete_sequence= px.colors.sequential.matter,
                hover_data=['model.1_', 'model_year', 'vehicleclass_'],
             labels={y_var: label, 'model_year': "Year of manufacture"},
             title= f"{label} over time")
    fig.update_layout(
            title_x=0.5,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )

    fig.update_yaxes(showgrid=False)
        
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
file_name_2022_1995 = "predicted_co2_rating.csv"
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
                    html.P("Select attribute", style=header_menu_style),
                    dcc.Dropdown(
                        id='attribute',
                        options=[
                                {'label': 'Vehicle class', 'value': 'vehicleclass_'},
                                {'label':"Fuel type", "value":"mapped_fuel_type"},
                                 {"label": "Fuel consumption in city (l/100km)", "value": "fuelconsumption_city(l/100km)"},
                                 {"label": "Fuel consumption in highway (l/km)", "value": "fuelconsumption_hwy(l/100km)"},
                                 {"label": "Fuel consumption combined (l/km)", "value": "fuelconsumption_comb(l/100km)"},
                                 {"label": "Fuel consumption combined (mpg)", "value": "fuelconsumption_comb(mpg)"}],
                        value= 'vehicleclass_',
                        style={'backgroundColor':"white"}),
                ], className="four columns"),
                html.Div([
                    html.P("Type or select a make", style=header_menu_style),
                    dcc.Dropdown(
                        id='select-make',
                        options=all_makes,
                        value= 'bmw',
                        style={'backgroundColor':"white"}),
                ],  className="two columns"),

                html.Div([
                    html.P("Select a year", style=header_menu_style),
                    dcc.Dropdown(
                        id='start-year',
                        options=[i + 1 for i in range(year_min-1, year_max)],
                        value= 2020,
                        style={'backgroundColor':"white"}),
                ],  className="two columns"),
                html.Div([
                    html.P("Select a CO2 rating", style=header_menu_style),
                    dcc.Dropdown(
                        id='co2-rating',
                        options=[i + 1 for i in range(0, 11)],
                        value= 5,
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

cards = dbc.Container([ 
    dbc.Row(
    [   
        dbc.Col(menu_card, width='auto'),
        dbc.Col(plots_card, width='auto'),
    ]
)
], fluid=True,style={'backgroundColor': "black"})


app.layout = html.Div([
    dbc.Col(text_card, width='auto'),
    dcc.Tabs([
        dcc.Tab(label='Fuel-only vehicle insights', children=[
            html.Div(
                children=[cards],
                style={'backgroundColor': "black"}
            )
        ]),
        dcc.Tab(label='Hybrid vehicle insights', children=[
            dcc.Graph(
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [1, 4, 1],
                            'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [1, 2, 3],
                         'type': 'bar', 'name': u'Montréal'},
                    ]
                }
            )
        ]),
        dcc.Tab(label='Electric vehicle insights', children=[
            dcc.Graph(
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [2, 4, 3],
                            'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [5, 4, 3],
                         'type': 'bar', 'name': u'Montréal'},
                    ]
                }
            )
        ]),
    ]),
    dbc.Col(footer_card, width='auto')
])


@app.callback(
    Output('graph-dist_fuel', 'figure'),
    [Input('attribute', 'value'),
    Input('select-make', 'value'),
    Input('start-year', 'value'),
    Input('co2-rating', 'value')])
def update_frequency_chart(attribute, make, start_year, co2_rating):
    vehicle_type = "fuel-only"
    sel_dataframe = dataframe_dictionary[vehicle_type]
    dataframe = sel_dataframe[(sel_dataframe['make_']==make) & 
                              (sel_dataframe['model_year']>=start_year) &
                              (sel_dataframe['predicted_co2_rating']>=co2_rating)]
    fig0 = generate_count_plot(attribute, vehicle_type, dataframe)
    return fig0

@app.callback(
    Output('graph-scatter-box', 'figure'),
    [Input('attribute', 'value'),
    Input('select-make', 'value'),
    Input('start-year', 'value'),
    Input('co2-rating', 'value')])
def update_frequency_chart(attribute, make, start_year, co2_rating):
    vehicle_type = "fuel-only"
    sel_dataframe = dataframe_dictionary[vehicle_type]
    dataframe = sel_dataframe[(sel_dataframe['make_']==make) & 
                              (sel_dataframe['model_year']>=start_year)  &
                               (sel_dataframe['predicted_co2_rating']>=co2_rating) ]
    fig0 = generate_scatter_or_box( dataframe, attribute)
    return fig0

@app.callback(
    Output('graph-models-released', 'figure'),
    [
    Input('select-make', 'value'),
    Input('start-year', 'value'),
    Input('co2-rating', 'value')])
def update_frequency_chart(make, start_year, co2_rating):
    vehicle_type = "fuel-only"
    sel_dataframe = dataframe_dictionary[vehicle_type]
    dataframe = sel_dataframe[(sel_dataframe['make_']==make) & 
                              (sel_dataframe['model_year']>=start_year)  & 
                              (sel_dataframe['predicted_co2_rating']>=co2_rating) ]
    fig0 = generate_bar_chart_with_models(dataframe)
    return fig0

@app.callback(
    Output('table-of-scores', 'figure'),
    [
    Input('select-make', 'value'),
    Input('start-year', 'value')])
def update_frequency_chart(make, start_year):
    vehicle_type = "fuel-only"
    sel_dataframe = dataframe_dictionary[vehicle_type]
    dataframe = sel_dataframe[(sel_dataframe['make_']==make) & 
                              (sel_dataframe['model_year']>=start_year)  &
                               (sel_dataframe['predicted_co2_rating']>=7) ]
    title_str = "Models with a CO2 rating of 7 or above"
    fig0 = display_table(dataframe, title_str)
    return fig0




if __name__ == '__main__':  
    
    app.run_server(debug=True) 