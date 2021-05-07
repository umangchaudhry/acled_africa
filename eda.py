import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def region_event(data):
    """
    This function creates a bar plot of the regional distribution of events 
    in the given dataset.
    
    Parameters
    ----------
    data : a pandas dataframe that contains a column called 'region' to group by
        

    Returns
    -------
    A matplotlib figure

    """
    df = data.groupby('region').size()
    ax = df.plot(kind='bar', figsize=(10,6), color='green', fontsize=13,zorder=2);
    ax.set_alpha(0.8)
    ax.set_title("Distribution of Events Across Regions", fontsize=18)
    ax.set_ylabel("Number of Events", fontsize=15)
    ax.set_xlabel("Region",fontsize=15)
    plt.xticks(rotation = 0)
    ax.grid(zorder=0)
    plt.show()
    

def event_dist(data):
    """
    This function creates a bar plot of the distribution of event types 
    in the given dataset.
    
    Parameters
    ----------
    data : a pandas dataframe that contains a column called 'event_type' to group by
        

    Returns
    -------
    A matplotlib figure

    """
    df = data.groupby('event_type').size()
    ax = df.plot(kind='bar', figsize=(10,6), color='green', fontsize=13,zorder=2);
    ax.set_alpha(0.8)
    ax.set_title("Distribution of Event Types", fontsize=18)
    ax.set_ylabel("Number of Instances", fontsize=15)
    ax.set_xlabel("Event Type",fontsize=15)
    plt.xticks(rotation = 45)
    ax.grid(zorder=0)
    plt.show()

def event_reg_dist(data):
    """
    This function creates a bar plot of the distribution of event types by 
    region in the given dataset.
    
    Parameters
    ----------
    data : a pandas dataframe that contains the following columns ['region', 'event_type']
        

    Returns
    -------
    A seaborn-matplotlib figure

    """
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    sns.countplot(data=data,x='region',hue='event_type', zorder=2)
    ax.set_alpha(0.8)
    ax.set_title("Distribution of Event Types by Region", fontsize=18)
    ax.set_ylabel("Number of Events", fontsize=15)
    ax.set_xlabel("Region",fontsize=15)
    ax.legend(title='Event Type')
    plt.xticks(rotation = 0)
    ax.grid(zorder=0)
    plt.show()
    
def protest_type_dist(data):
    """
    This function creates a bar plot of the distribution of protest types by 
    region in the given dataset.
    
    Parameters
    ----------
    data : a pandas dataframe that contains the following columns ['region', 'sub_event_type']
        

    Returns
    -------
    A seabron-matplotlib figure

    """
    p_data = data.loc[data['event_type'].isin(['Protests'])]
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    sns.countplot(data=p_data,x='region',hue='sub_event_type', zorder=2)
    ax.set_alpha(0.8)
    ax.set_title("Distribution of Protest Types by Region", fontsize=18)
    ax.set_ylabel("Number of Events", fontsize=15)
    ax.set_xlabel("Region",fontsize=15)
    ax.legend(title='Event Type')
    plt.xticks(rotation = 0)
    ax.grid(zorder=0)
    plt.show()
    
def event_reg_year(data):
    """
    This function creates a line plot of the distribution of event by year and 
    region in the given dataset.

    Parameters
    ----------
    data : a pandas dataframe that contains the following columns ['region', 'year'] to group by
        

    Returns
    -------
    A seabron-matplotlib figure

    """
    bp_data = data.loc[data['event_type'].isin(['Protests','Battles'])]
    data_viz = bp_data.groupby(['year','region'], as_index=False).size()
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    sns.lineplot(data=data_viz, x='year', y='size', hue='region',zorder=2)
    ax.set_alpha(0.8)
    ax.set_title("Distribution of Events by Year and Region", fontsize=18)
    ax.set_ylabel("Number of Events", fontsize=15)
    ax.set_xlabel("Year",fontsize=15)
    ax.legend(title='Region')
    plt.xticks(rotation = 0)
    ax.grid(zorder=0)
    plt.show()

def bat_reg_year(data):
    """
    This function creates a line plot of the distribution of battles by year and 
    region in the given dataset.
    
    Parameters
    ----------
    data : a pandas dataframe that contains the following columns ['region', 'year'] to group by
        

    Returns
    -------
    A seaborn-matplotlib figure

    """
    b_data = data.loc[data['event_type'].isin(['Battles'])]    
    data_viz = b_data.groupby(['year','region'], as_index=False).size()
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    sns.lineplot(data=data_viz, x='year', y='size', hue='region',zorder=2)
    ax.set_alpha(0.8)
    ax.set_title("Distribution of Battles by Year and Region", fontsize=18)
    ax.set_ylabel("Number of Events", fontsize=15)
    ax.set_xlabel("Year",fontsize=15)
    ax.legend(title='Region')
    plt.xticks(rotation = 0)
    ax.grid(zorder=0)
    plt.show()
    
def pro_reg_year(data):
    """
    This function creates a line plot of the distribution of protests by year and 
    region in the given dataset.
    
    Parameters
    ----------
    data : a pandas dataframe that contains the following columns ['region', 'year'] to groupby
        

    Returns
    -------
    A seaborn-matplotlib figure

    """
    p_data = data.loc[data['event_type'].isin(['Protests'])]
    data_viz = p_data.groupby(['year','region'], as_index=False).size()
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    sns.lineplot(data=data_viz, x='year', y='size', hue='region',zorder=2)
    ax.set_alpha(0.8)
    ax.set_title("Distribution of Protests by Year and Region", fontsize=18)
    ax.set_ylabel("Number of Events", fontsize=15)
    ax.set_xlabel("Year",fontsize=15)
    ax.legend(title='Region')
    plt.xticks(rotation = 0)
    ax.grid(zorder=0)
    plt.show()

def fatalities_yearly(data):
    """
    This function creates a line plot of the number of fatalities by year in the given 
    dataset as a result of battles and protests.
    
    Parameters
    ----------
    data : a pandas dataframe that contains the column 'year' to groupby and 'fatalities' to calculate a sum
        

    Returns
    -------
    A seaborn-matplotlib figure

    """
    bp_data = data.loc[data['event_type'].isin(['Protests','Battles'])]    
    f_counts = bp_data.groupby(['year'], as_index = False).sum('fatalities')[['year', 'fatalities']]
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    sns.lineplot(data=f_counts, x='year', y='fatalities',zorder=2)
    ax.set_alpha(0.8)
    ax.set_title("Fatalities per Year", fontsize=18)
    ax.set_ylabel("Number of Fatalities", fontsize=15)
    ax.set_xlabel("Year",fontsize=15)
    plt.xticks(rotation = 0)
    ax.grid(zorder=0)
    plt.show()