import pandas as pd
import matplotlib.pyplot as plt

# Loading the cleaned data
outflow_df = pd.read_csv('cleaned_countyoutflow.csv')
inflow_df = pd.read_csv('cleaned_countyinflow.csv')

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
    trends = df.groupby('year').agg({
        'adjusted_gross_income': 'sum', 
        'num_individuals': 'sum'
    }).reset_index()
    
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.bar(trends['year'], trends['adjusted_gross_income'], color='g', alpha=0.6)
    ax2.plot(trends['year'], trends['num_individuals'], color='b', marker='o')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total AGI (in billions)', color='g')
    ax2.set_ylabel('Total Individuals', color='b')

    plt.title(title)
    plt.show()

# Plotting trends
plot_trends(outflow_df, "Outflow Trends Over Time")
plot_trends(inflow_df, "Inflow Trends Over Time")

# Top counties/states by AGI inflow/outflow
def top_counties(df, title, top_n=10):
    top = df.groupby(['destination_state_fips', 'y2_countyname']).agg({
        'adjusted_gross_income': 'sum', 
        'num_individuals': 'sum'
    }).reset_index()
    
    top = top.sort_values('adjusted_gross_income', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top['y2_countyname'], top['adjusted_gross_income'], color='skyblue')
    plt.xlabel('AGI (in billions)')
    plt.title(f'Top {top_n} Counties by AGI - {title}')
    plt.show()

# Top 10 counties by AGI for inflow and outflow
top_counties(outflow_df, "Outflow")
top_counties(inflow_df, "Inflow")
