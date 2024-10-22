import pandas as pd
import matplotlib.pyplot as plt

# Loading the data
outflow_df = pd.read_csv('countyoutflow16-21 compiled.csv')
inflow_df = pd.read_csv('countyinflow16-21 compiled.csv')

# Basic summary statistics
def summarize_data(df):
    print("Summary statistics:")
    print(df.describe(include='all'))
    print("\nMissing values:")
    print(df.isnull().sum())

# Outflow summary
print("Outflow Data Summary")
summarize_data(outflow_df)

# Inflow summary
print("Inflow Data Summary")
summarize_data(inflow_df)

# Trend over time for outflow and inflow
def plot_trends(df, title):
    trends = df.groupby('year').agg({'agi': 'sum', 'n2': 'sum'}).reset_index()
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.bar(trends['year'], trends['agi'], color='g', alpha=0.6)
    ax2.plot(trends['year'], trends['n2'], color='b', marker='o')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total AGI (in billions)', color='g')
    ax2.set_ylabel('Total Individuals', color='b')

    plt.title(title)
    plt.show()

# Plotting trends
plot_trends(outflow_df, "Outflow Trends Over Time")
plot_trends(inflow_df, "Inflow Trends Over Time")

# Top countries/states by AGI inflow/outflow
def top_counties(df, title, top_n=10):
    top = df.groupby(['y2_state', 'y2_countyname']).agg({'agi': 'sum', 'n2': 'sum'}).reset_index()
    top = top.sort_values('agi', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top['y2_countyname'], top['agi'], color='skyblue')
    plt.xlabel('AGI (in billions)')
    plt.title(f'Top {top_n} Counties by AGI - {title}')
    plt.show()

# Top 10 counties by AGI for inflow and outflow
top_counties(outflow_df, "Outflow")
top_counties(inflow_df, "Inflow")
