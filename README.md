# acled_africa
A Python Package for performing analysis on the relationships between protests and battles in Africa Nations using the ACLED data

# Quick navigation
[Installation Instructions](#installation-instructions)  
[Dependencies](#dependencies)  
[Package Structure](#package-structure)

## Installation Instructions

To use this package, please clone this repository in the working directory for your project. Once the acled_africa folder is in your working directory, you will be able to import the package as any other python package:

```python
import acled_africa
```

Additionally, each module within the package can be imported as follows:

```python
from acled_africa import eda
from acled_africa import maps
from acled_africa import country_analysis
from acled_africa import textanalysis
```

Functions in each module can be called as follows:
```python
eda.region_event(params)
```

## Dependencies

This package uses several pacakges to run the build in functions. You will be required to install these dependencies before being able to use this package. However, once installed, you are not required to import the dependencies. The required packages are listed below:

* numpy
* pandas
* matplotlib
* seaborn
* sklearn
* time
* wordcloud
* itertools
* warnings
* re
* json
* geopandas
* folium
* networkx
* pylab

## Package Structure

This package has 4 separate modules:

1. eda
2. maps
3. country_analysis
4. textanalysis

Each module has several built in functions that use given data to perform analysis. Descriptions of each of the functions within these modules can be found below:

### eda

```python
region_event(data)
```
This function creates a bar plot of the regional distribution of events in the given dataset.


```python
event_dist(data)
```
This function creates a bar plot of the distribution of event types in the given dataset.


```python
event_reg_dist(data)
```
This function creates a bar plot of the distribution of event types by region in the given dataset.


```python
protest_type_dist(data)
```
This function creates a bar plot of the distribution of protest types by region in the given dataset.


```python
event_reg_year(data)
```
This function creates a line plot of the distribution of event by year and region in the given dataset.


```python
bat_reg_year(data)
```
This function creates a line plot of the distribution of battles by year and region in the given dataset.


```python
pro_reg_year(data)
```
This function creates a line plot of the distribution of protests by year and region in the given dataset.


```python
pro_reg_year(data)
```
This function creates a line plot of the number of fatalities by year in the given dataset as a result of battles and protests.


```python
fatalities_yearly(data)
```
This function creates a line plot of the number of fatalities by year in the given dataset as a result of battles and protests.


### maps

```python
protests_maps(data)
```
This function generates a map of the number of protests by country in Africa.


```python
battles_maps(data)
```
This function generates a map of the number of battles by country in Africa.


```python
fatalities_maps(data)
```
This function generates a map of the number of fatalities by country in Africa.


## country_analysis

```python
yearly_events(data, country)
```
This function creates a bar plot of the number of events in the given country.


```python
bp_yearly(data, country)
```
This function creates a bar plot of the number of battles and protests per year for the selected country.


```python
protests_network(data, country, actor_type)
```
This function generates and draws a networkx graph object for the actors involved ('main actor' or 'associated actor') in protests in the given country.


```python
battles_network(data, country, actor_type)
```
This function generates and draws a networkx graph object for the actors involved ('main actor' or 'associated actor') in battles in the given country.


```python
country_maps(data, country, event_type)
```
This function generates a map of the event locations in the acled dataset for the selected country and event type


### textanalysis

```python
keyword_extraction(data,event_type,n_gram_range=(1,3),keyword_n=5,mss_n=5,mss_nr=10,mmr_n=5,mmr_diversity=0.7)
```
This function performs keyword extraction for the given country and event type. NOTE: each row of data takes around 1.5 seconds to process, so running this on a large dataset can take several hours. 


```python
cluster_analysis(data, country, event_type, word_type, k, plot_silhouette=True, word_cloud=True, save=True)
```
This function performs cluster_analysis on the extracted keywords (you may specify the keyword type being used) for the given country and outputs the silhouette scores for each cluster. You are required to then select (you will be prompted) the best value for 'k' and word clouds will be generated using kmeans analysis to separate the events into clusters. Each row of data is also assigned the cluster number. 

**For more information on these functions, please use the help function in python to explore the documentation for each function**






