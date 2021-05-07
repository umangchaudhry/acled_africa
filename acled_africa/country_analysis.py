import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pylab import rcParams
import folium
import geopandas as gpd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

def yearly_events(data, country):
    """
    This function creates a bar plot of the number of events 
    in the given dataset.
    
    Parameters
    ----------
    data : a pandas dataframe that countains the column 'event_type' containing 'Protests' and 'Battles'
    as values and a column called country to subset the data
        
    country : the country for which the figure needs to be generated 
        

    Returns
    -------
    A seaborn-matplotlib figure

    """
    country_data = data[data['country']==country]
    country_data = country_data.loc[country_data['event_type'].isin(['Protests','Battles'])]
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    sns.countplot(data=country_data,x='year', zorder=2)
    ax.set_title("Number of Events per Year", fontsize=18)
    ax.set_ylabel("Number of Events", fontsize=15)
    ax.set_xlabel("Year",fontsize=15)
    plt.xticks(rotation = 0)
    ax.grid(zorder=0)
    plt.show()

def bp_yearly(data,country):
    """
    This function creates a bar plot of the number of battles and protests per year
    for the selected country

    Parameters
    ----------
    data : a pandas dataframe that countains the column 'event_type' containing 'Protests' and 'Battles'
    as values and a column called country to subset the data
        
    country : the country for which the figure needs to be generated 
        

    Returns
    -------
    A seaborn-matplotlib figure

    """
    country_data = data[data['country']==country]
    country_data = country_data.loc[country_data['event_type'].isin(['Protests','Battles'])]    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    sns.countplot(data=country_data,x='year', hue = 'event_type', zorder=2)
    ax.set_title("Number of Battles and Protests per Year", fontsize=18)
    ax.set_ylabel("Number of Events", fontsize=15)
    ax.set_xlabel("Year",fontsize=15)
    ax.legend(title = 'Event Type')
    plt.xticks(rotation = 0)
    ax.grid(zorder=0)
    plt.show()

def protests_network(data, country, actor_type):
    """
    This function generates and draws a networkx graph object for the actors involved in protests
    in the given country

    Parameters
    ----------
    data : a pandas dataframe that contains the columns ['country', 'event_type', 'actor1', 'actor2', 
    assoc_actor1 and assoc_actor2]
        
    country : the country for which the network is being made from the dataset
        
    actor_type : two options:
                        1. "main actor" to generate a network of the main actors (actor1,actor2)
                            in the acled dataset
                        2. "associated actor" to generate a network of the associated actors
                            (assoc_actor_1, assoc_actor_2) in the acled dataset

    Returns
    -------
    A networkx graph figure

    """
    country_data = data[data['country']==country]
    if actor_type == 'main actor':
        country_protests = country_data[country_data['event_type']=='Protests'][['actor1', 'actor2']].dropna()
        G = nx.from_pandas_edgelist(country_protests, source='actor1', target='actor2')
        rcParams['figure.figsize'] = 14, 10
        pos = nx.spring_layout(G, scale=100, k=3/np.sqrt(G.order()))
        colors = [['lightgrey', 'lightblue'][node.startswith('Pr')] 
                  for node in G.nodes()]
        d = dict(G.degree)
        nx.draw(G, pos, 
                with_labels=True, 
                nodelist=d, 
                node_size=[d[k]*300 for k in d],
                node_color=colors)
        
    elif actor_type == 'associated actor':
        country_protests = country_data[country_data['event_type']=='Protests'][['assoc_actor_1', 'assoc_actor_2']].dropna()
        G = nx.from_pandas_edgelist(country_protests, source='assoc_actor_1', target='assoc_actor_2')
        rcParams['figure.figsize'] = 14, 10
        pos = nx.spring_layout(G, scale = 10, k=3/np.sqrt(G.order()))
        colors = [['lightgrey', 'lightblue'][node.startswith(('Police', 'Military'))] 
                  for node in G.nodes()]
        d = dict(G.degree)
        nx.draw(G, pos, 
                with_labels=True, 
                nodelist=d, 
                node_size=[d[k]*300 for k in d],
                node_color=colors)

def country_maps(data, country, event_type):
    """
    This function generates a map of the event locations in the acled 
    dataset for the selected country

    Parameters
    ----------
    data : a pandas dataframe that contains the following columns ['country', 'event_type', 'latitude', 'longitude']
        
    country : the country for which the map needs to be generated
        
    event_type : 3 options:
                    1.'both' to generate a map containing battles and protests locations
                    2.'protests' to generate a map containing the locations of protests
                    3.'battles' to generate a map containing the locations of battles
        

    Returns
    -------
    A folium map object

    """
    country_data = data[data['country']==country]
    country_data = country_data.loc[country_data['event_type'].isin(['Protests','Battles'])]
    #get map data
    countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    africa = countries[countries['continent'] == 'Africa']
    africa = africa.dropna()
    africa[['name', 'geometry']].to_file('data/africa.json', driver='GeoJSON', encoding='utf-8')
    geojson_data = json.load(open('data/africa.json','r'))
    for feature in geojson_data['features']:
         properties = feature['properties']
         feature.update(id=properties['name'])
    colors = {'Protests' : 'royalblue', 'Battles' : 'red'}
    m = folium.Map(location=[9.546647, 9.373178], zoom_start = 6.4, tiles='cartodbpositron')

    if event_type == 'both':
        country_data.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]],radius=1, color=colors[row['event_type']],fill_color=colors[row['event_type']],opacity = 0.5).add_to(m), axis=1)

        choropleth = folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            key_on="feature.id",
            fill_color = 'white',
            line_opacity=.5,
            popup = 'Name',
            fill_opacity = 0
        ).add_to(m)

    if event_type == 'protests':
        country_data[country_data['event_type']=='Protests'].apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=1, color=colors[row['event_type']], fill_color=colors[row['event_type']], opacity = 0.5).add_to(m), axis=1)

        choropleth = folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            key_on="feature.id",
            fill_color = 'white',
            line_opacity=.5,
            popup = 'Name',
            fill_opacity = 0
        ).add_to(m)

    if event_type == 'battles':
        country_data[country_data['event_type']=='Battles'].apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=1, color=colors[row['event_type']], fill_color=colors[row['event_type']], opacity = 0.5).add_to(m), axis=1)

        choropleth = folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            key_on="feature.id",
            fill_color = 'white',
            line_opacity=.5,
            popup = 'Name',
            fill_opacity = 0
        ).add_to(m)

    return m

def battles_network(data, country, actor_type):
    """
    This function generates and draws a networkx graph object for the actors involved in battles
    in the given country

    Parameters
    ----------
    data : a pandas dataframe that contains the columns ['country', 'event_type', 'actor1', 'actor2', 
    assoc_actor1 and assoc_actor2]
        
    country : the country for which the network is being made from the dataset
        
    actor_type : two options:
                        1. "main actor" to generate a network of the main actors (actor1,actor2)
                            in the acled dataset
                        2. "associated actor" to generate a network of the associated actors
                            (assoc_actor_1, assoc_actor_2) in the acled dataset

    Returns
    -------
    A networkx graph figure

    """
    country_data = data[data['country']==country]
    if actor_type == 'main actor':
        country_battles = country_data[country_data['event_type']=='Battles'][['actor1', 'actor2']].dropna()
        country_counts = country_battles.groupby(['actor1', 'actor2'],as_index=False).size().nlargest(20,'size',keep='all')
        n_battles = country_battles[country_battles['actor1'].isin(country_counts['actor1']) & country_battles['actor2'].isin(country_counts['actor2'])]
        G = nx.from_pandas_edgelist(n_battles, source='actor1', target='actor2')
        rcParams['figure.figsize'] = 14, 10
        pos = nx.spring_layout(G, scale=100, k=3/np.sqrt(G.order()))
        colors = [['lightgrey', 'lightblue'][node.startswith(('Police', 'Military'))] 
                  for node in G.nodes()]
        d = dict(G.degree)
        nx.draw(G, pos, 
                with_labels  =True, 
                nodelist=d, 
                node_size=[d[k]*300 for k in d],
                node_color=colors)

    elif actor_type == 'associated actor':
        country_battles = country_data[country_data['event_type']=='Battles'][['assoc_actor_1', 'assoc_actor_2']].dropna()
        G = nx.from_pandas_edgelist(country_battles, source='assoc_actor_1', target='assoc_actor_2')
        rcParams['figure.figsize'] = 14, 10
        pos = nx.spring_layout(G, scale=10, k=3/np.sqrt(G.order()))
        colors = [['lightgrey', 'lightblue'][node.startswith(('Police', 'Military'))] 
                  for node in G.nodes()]
        d = dict(G.degree)
        nx.draw(G, pos, 
                with_labels=True, 
                nodelist=d, 
                node_size=[d[k]*300 for k in d],
                node_color=colors)

