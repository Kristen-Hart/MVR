import pandas as pd

# Loading the data
outflow_df = pd.read_csv('countyoutflow16-21 compiled.csv')
inflow_df = pd.read_csv('countyinflow16-21 compiled.csv')

print(outflow_df.info())
print(outflow_df.head())

# Renaming
outflow_df = outflow_df.rename(columns={
    'y1_statefips': 'origin_state_fips',
    'y1_countyfips': 'origin_county_fips',
    'y2_statefips': 'destination_state_fips',
    'y2_countyfips': 'destination_county_fips',
    'n1': 'num_returns',
    'n2': 'num_individuals',
    'agi': 'adjusted_gross_income'
})

inflow_df = inflow_df.rename(columns={
    'y1_statefips': 'origin_state_fips',
    'y1_countyfips': 'origin_county_fips',
    'y2_statefips': 'destination_state_fips',
    'y2_countyfips': 'destination_county_fips',
    'n1': 'num_returns',
    'n2': 'num_individuals',
    'agi': 'adjusted_gross_income'
})

outflow_df = outflow_df.dropna()  # Drop any rows with missing data
inflow_df = inflow_df.dropna()  # Drop any rows with missing data

# Convert AGI to numeric
outflow_df['adjusted_gross_income'] = pd.to_numeric(outflow_df['adjusted_gross_income'], errors='coerce')
inflow_df['adjusted_gross_income'] = pd.to_numeric(inflow_df['adjusted_gross_income'], errors='coerce')

# Removing duplicates
outflow_df = outflow_df.drop_duplicates()
inflow_df = inflow_df.drop_duplicates()


# Exporting cleaned data
outflow_df.to_csv('cleaned_countyoutflow.csv', index=False)
inflow_df.to_csv('cleaned_countyinflow.csv', index=False)
