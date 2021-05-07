import json
import geopandas as gpd
import folium
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def protests_maps(data):
    """
    This function generates a map of the number of protests by country in Africa
    
    Parameters
    ----------
    data : a pandas dataframe containing the following columns ['country', 'protests', 'region', 'iso3']
        

    Returns
    -------
    A folium map object

    """
    #subset data
    p_data=data.loc[data['event_type'].isin(['Protests'])]
    p_counts=p_data.groupby(['country', 'region', 'iso3'], as_index=False).size()
    #get map data
    countries=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    africa=countries[countries['continent']=='Africa']
    africa_protests=pd.merge(africa,p_counts,how='left',left_on='iso_a3',right_on='iso3')
    africa_protests=africa_protests.dropna()
    n_protests=africa_protests[['name','size']]
    africa_protests[['name','geometry','size']].to_file('data/africa_protests.json',driver='GeoJSON',encoding='utf-8')
    geojson_data=json.load(open('data/africa_protests.json','r'))
    for feature in geojson_data['features']:
        properties=feature['properties']
        feature.update(id=properties['name'])
    m=folium.Map(location=[4.546647, 22.373178], zoom_start = 4, tiles='cartodbpositron')
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        name="choropleth",
        data=n_protests,
        columns=["name", "size"],
        key_on="feature.id",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=.1,
        popup = 'Name',
        legend_name="Number of Protests",
        highlight = True
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(["name" , "size"],labels=False)
    )
    return m


def battles_maps(data):
    """
    This function generates a map of the number of battles by country in Africa
    
    Parameters
    ----------
    data : a pandas dataframe containing the following columns ['country', 'protests', 'region', 'iso3']
        

    Returns
    -------
    A folium map object

    """
    #subset data
    b_data=data.loc[data['event_type'].isin(['Battles'])]
    b_counts=b_data.groupby(['country', 'region', 'iso3'], as_index=False).size()
    #get map data
    countries=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    africa=countries[countries['continent']=='Africa']
    africa_battles=pd.merge(africa,b_counts,how='left',left_on='iso_a3',right_on='iso3')
    africa_battles=africa_battles.dropna()
    n_battles=africa_battles[['name','size']]
    africa_battles[['name','geometry','size']].to_file('data/africa_battles.json',driver='GeoJSON',encoding='utf-8')
    geojson_data=json.load(open('data/africa_battles.json','r'))
    for feature in geojson_data['features']:
        properties=feature['properties']
        feature.update(id=properties['name'])
    m=folium.Map(location=[4.546647, 22.373178], zoom_start = 4, tiles='cartodbpositron')
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        name="choropleth",
        data=n_battles,
        columns=["name", "size"],
        key_on="feature.id",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=.1,
        popup = 'Name',
        legend_name="Number of Battles",
        highlight = True
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(["name" , "size"],labels=False)
    )
    return m

def fatalities_maps(data):
    """
    This function generates a map of the number of fatalities by country in Africa
    
    Parameters
    ----------
    data : a pandas dataframe containing the following columns ['country', 'protests', 'region', 'iso3', 'fatalities']
        

    Returns
    -------
    A folium map object

    """
    bp_data = data.loc[data['event_type'].isin(['Protests','Battles'])]
    f_counts = bp_data.groupby(['country', 'region', 'iso3'], as_index = False).sum('fatalities')[['country', 'region', 'iso3', 'fatalities']]
    #get map data
    countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    africa = countries[countries['continent'] == 'Africa']
    africa_fatalities = pd.merge(africa, f_counts, how = 'left', left_on = 'iso_a3', right_on = 'iso3')
    africa_fatalities = africa_fatalities.dropna()
    n_fatalities = africa_fatalities[['name', 'fatalities']]
    africa_fatalities[['name', 'geometry', 'fatalities']].to_file('data/africa_fatalities.json', driver='GeoJSON', encoding='utf-8')
    geojson_data = json.load(open('data/africa_fatalities.json','r'))
    for feature in geojson_data['features']:
         properties = feature['properties']
         feature.update(id=properties['name'])
    m=folium.Map(location=[4.546647, 22.373178], zoom_start = 4, tiles='cartodbpositron')
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        name="choropleth",
        data=n_fatalities,
        columns=["name", "fatalities"],
        key_on="feature.id",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=.1,
        popup = 'Name',
        legend_name="Number of Fatalities",
        highlight = True
    ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(["name","fatalities"],labels=False)
    )
    return m
    