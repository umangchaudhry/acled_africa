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











