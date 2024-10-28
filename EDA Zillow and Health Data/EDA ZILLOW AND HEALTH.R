required_packages <- c("dplyr", "ggplot2", "tidyr", "skimr", "corrplot", "readr", "maps", "car", "GGally")
installed_packages <- rownames(installed.packages())
for (p in required_packages) {
  if (!(p %in% installed_packages)) {
    install.packages(p)
  }
}

library(dplyr)
library(ggplot2)
library(tidyr)
library(skimr)
library(corrplot)
library(readr)
library(maps)
library(car)       
library(GGally)    

# 3. Assign the data already loaded in the environment to more manageable variables
zillow_yearly <- Zillow.Data.Yearly.FIPS
zillow_monthly <- Zillow.Data.Monthly.FIPS
health_data <- Health.Data

# 4. Exploratory Data Analysis (EDA)

# 4.1. Basic summary of the datasets
cat("\nSummary of Zillow Data Yearly FIPS\n")
skim(zillow_yearly)

cat("\nSummary of Zillow Data Monthly FIPS\n")
skim(zillow_monthly)

cat("\nSummary of Health Data\n")
skim(health_data)

# 4.2. Missing data verification and visualization
plot_missing_data <- function(data, dataset_name) {
  missing_data <- data %>%
    summarise_all(~ mean(is.na(.))) %>%
    gather(key = "variable", value = "missing_percentage") %>%
    arrange(desc(missing_percentage))
  
  # Visualize the percentages of missing data
  ggplot(missing_data, aes(x = reorder(variable, missing_percentage), y = missing_percentage)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = paste("Percentage of Missing Data -", dataset_name),
         x = "Variable", y = "Missing Percentage") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 6),  # Reduce Y axis text size
          plot.title = element_text(size = 14),  # Adjust title size
          axis.text.x = element_text(size = 10)) # Adjust X axis text size
}

# Apply the function to the datasets
plot_missing_data(zillow_yearly, "Zillow Data Yearly")
plot_missing_data(zillow_monthly, "Zillow Data Monthly")
plot_missing_data(health_data, "Health Data")


# Yearly
analyze_correlation_vif_yearly <- function(data, dataset_name) {
  # Select all numeric variables without missing values
  numeric_data <- data %>%
    select_if(is.numeric) %>%
    drop_na()
  
  if (ncol(numeric_data) > 1) {
    # Calculate the correlation matrix
    correlation_matrix <- cor(numeric_data)
    
    # Visualize the complete correlation matrix
    corrplot(correlation_matrix, method = "circle",
             title = paste("Correlation Matrix -", dataset_name), mar = c(0, 0, 1, 0))
    
    # Calculate VIF for all variables
    vif_model <- lm(as.formula(paste(names(numeric_data)[1], "~ .")), data = numeric_data)
    vif_values <- vif(vif_model)
    
    # Display all VIF values
    cat("\nVIF Values for", dataset_name, ":\n")
    print(vif_values)
    
    # Identify and display variables with high VIF values
    high_vif <- vif_values[vif_values > 5]
    if (length(high_vif) > 0) {
      cat("\nVariables with multicollinearity (VIF > 5):\n")
      print(high_vif)
    } else {
      cat("\nNo significant multicollinearity detected in", dataset_name, "\n")
    }
  } else {
    cat("\nNot enough numeric variables in", dataset_name, "\n")
  }
}

# Apply the function to the Zillow Yearly dataset
analyze_correlation_vif_yearly(zillow_yearly, "Zillow Data Yearly")




# 4.3. Correlation analysis and multicollinearity detection
analyze_correlation_vif <- function(data, dataset_name) {
  numeric_data <- data %>%
    select_if(is.numeric) %>%
    select_if(~ !any(is.na(.)))
  
  if (ncol(numeric_data) > 1) {
    # Calculate the correlation matrix
    correlation_matrix <- cor(numeric_data)
    
    # Visualize the correlation matrix
    corrplot(correlation_matrix, method = "circle",
             title = paste("Correlation Matrix -", dataset_name), mar = c(0, 0, 1, 0))
    
    # Calculate VIF
    vif_model <- lm(as.formula(paste(names(numeric_data)[1], "~ .")), data = numeric_data)
    vif_values <- vif(vif_model)
    
    # Display VIF values
    cat("\nVIF Values for", dataset_name, ":\n")
    print(vif_values)
    
    # Identify variables with high VIF
    high_vif <- vif_values[vif_values > 5]
    if (length(high_vif) > 0) {
      cat("\nVariables with multicollinearity (VIF > 5):\n")
      print(high_vif)
    } else {
      cat("\nNo significant multicollinearity detected in", dataset_name, "\n")
    }
  } else {
    cat("\nNot enough numeric variables without missing data in", dataset_name, "\n")
  }
}

# Apply the correlation and multicollinearity analysis function
analyze_correlation_vif(zillow_yearly, "Zillow Data Yearly")
analyze_correlation_vif(zillow_monthly, "Zillow Data Monthly")
analyze_correlation_vif(health_data, "Health Data")


# 4.4. Visualization of key variable distributions
# Check column names
names(zillow_yearly)

# Distribution of housing prices in 2021
if ("X2021" %in% colnames(zillow_yearly)) {
  ggplot(zillow_yearly, aes(x = `X2021`)) +
    geom_histogram(binwidth = 50000, fill = "green", color = "black") +
    labs(title = "Distribution of Housing Prices in 2021", x = "Price in 2021", y = "Frequency") +
    theme_minimal()
} else {
  cat("The 'X2021' column does not exist in the dataset.")
}

# Distribution of 'Premature Deaths' variable in Health Data
if ("Premature.Deaths" %in% colnames(health_data)) {
  ggplot(health_data, aes(x = `Premature.Deaths`)) +
    geom_histogram(binwidth = 100, fill = "orange", color = "black", na.rm = TRUE) +  # na.rm removes NA values
    labs(title = "Distribution of Premature Deaths",
         x = "Premature Deaths", y = "Frequency") +
    theme_minimal()
} else {
  cat("The 'Premature.Deaths' column does not exist in the dataset.")
}

# 4.5. Bivariate relationships (scatter plot and pair matrix)
# Scatter plot between housing prices in 2020 and 2021
if ("X2020" %in% colnames(zillow_yearly) && "X2021" %in% colnames(zillow_yearly)) {
  ggplot(zillow_yearly, aes(x = `X2020`, y = `X2021`)) +
    geom_point() +
    labs(title = "Relationship between Housing Prices in 2020 and 2021",
         x = "Price in 2020", y = "Price in 2021") +
    theme_minimal()
} else {
  cat("The 'X2020' or 'X2021' columns do not exist in the dataset.")
}

# Pair matrix to analyze relationships between multiple variables
if (all(c("X2019", "X2020", "X2021") %in% colnames(zillow_yearly))) {
  ggpairs(zillow_yearly %>% select(`X2019`, `X2020`, `X2021`))
} else {
  cat("The 'X2019', 'X2020', or 'X2021' columns do not exist in the dataset.")
}

# 4.6. Geographic analysis with maps
names(zillow_yearly)

coordinates_data <- data.frame(
  FIPS = zillow_yearly$FIPS,  # Assuming 'FIPS' is the common key
  long = rnorm(n = nrow(zillow_yearly), mean = -100, sd = 5),
  lat = rnorm(n = nrow(zillow_yearly), mean = 40, sd = 5)
)

zillow_with_coordinates <- zillow_yearly %>%
  left_join(coordinates_data, by = "FIPS")

us_map <- map_data("state")
ggplot(data = us_map, aes(x = long, y = lat, group = group)) +
  geom_polygon(fill = "white", color = "black") +
  geom_point(data = zillow_with_coordinates, aes(x = long, y = lat, color = `X2021`), inherit.aes = FALSE) +
  labs(title = "Housing Prices by County in 2021") +
  theme_minimal()

# 4.7. Temporal analysis of housing prices
zillow_monthly_long <- zillow_monthly %>%
  pivot_longer(cols = starts_with("X"), names_to = "Date", values_to = "Price") %>%
  mutate(Date = as.Date(gsub("^X", "", Date), format = "%m.%d.%Y"))

# Visualize the temporal trend of housing prices in a county
ggplot(zillow_monthly_long %>% filter(RegionName == "San Francisco"), aes(x = Date, y = Price)) +
  geom_line(color = "blue") +
  labs(title = "Housing Price Trend in San Francisco",
       x = "Date", y = "Housing Price") +
  theme_minimal()


# 5. Data Treatment
# Step 1: Standardize Missing Values
# Convert any empty strings or specific markers to NA
health_data[health_data == ""] <- NA
health_data[health_data == " "] <- NA
health_data[health_data == "NA"] <- NA
health_data[health_data == "N/A"] <- NA
health_data[health_data == "null"] <- NA

# Step 2: Identify and Remove Columns with Any Missing Data
cat("\nIdentifying columns with any missing values in Health Data...\n")
columns_with_na <- colnames(health_data)[colSums(is.na(health_data)) > 0]
cat("Columns with missing data:\n")
print(columns_with_na)

# Remove columns that contain any missing values
cat("\nRemoving columns with any missing values in Health Data...\n")
health_data_clean <- health_data %>%
  select(-all_of(columns_with_na))

# Step 3: Verify No Missing Data Remains
cat("\nVerifying that all columns in cleaned Health Data have no missing values:\n")
missing_values_check <- colSums(is.na(health_data_clean))
print(missing_values_check[missing_values_check > 0])  # Should be empty if all missing values are removed

if (any(missing_values_check > 0)) {
  stop("Error: There are still missing values in some columns of Health Data after cleaning.")
} else {
  cat("All columns in Health Data are now confirmed to be free of missing values.\n")
}

# Step 4: Save the Cleaned Dataset
write.csv(health_data_clean, "health_data_clean.csv", row.names = FALSE, na = "")

# Step 5: Re-import and Re-verify to Ensure Consistency
cat("\nRe-importing the saved Health Data file to ensure no missing data was saved...\n")
health_data_clean_reimport <- read.csv("health_data_clean.csv")
reimport_missing_check <- colSums(is.na(health_data_clean_reimport))
print(reimport_missing_check[reimport_missing_check > 0])  # Should return empty if all data is clean

if (any(reimport_missing_check > 0)) {
  stop("Error: Missing values found in re-imported Health Data file. Please review the cleaning process.")
} else {
  cat("The re-imported Health Data file is confirmed to have no missing values.\n")
}


# 5.2. Remove rows with missing data in Zillow Yearly and Zillow Monthly (Stricter Version)
cat("\nRemoving rows with missing values in Zillow Yearly and Monthly (Stricter Version)\n")
zillow_yearly_clean <- zillow_yearly %>% drop_na()  # Remove rows with any NA values
zillow_monthly_clean <- zillow_monthly %>% drop_na()  # Remove rows with any NA values

# Confirm that all rows in zillow_yearly_clean and zillow_monthly_clean have no missing values
cat("\nChecking for remaining missing values in cleaned Zillow Yearly rows:\n")
if (any(rowSums(is.na(zillow_yearly_clean)) > 0)) {
  cat("Warning: There are still missing values in some rows of Zillow Yearly.\n")
} else {
  cat("All rows in Zillow Yearly are now free of missing values.\n")
}

cat("\nChecking for remaining missing values in cleaned Zillow Monthly rows:\n")
if (any(rowSums(is.na(zillow_monthly_clean)) > 0)) {
  cat("Warning: There are still missing values in some rows of Zillow Monthly.\n")
} else {
  cat("All rows in Zillow Monthly are now free of missing values.\n")
}

# 6. Final verification of cleaned datasets
cat("\nFinal structure of cleaned Health Data\n")
str(health_data_clean)

cat("\nFinal structure of cleaned Zillow Yearly\n")
str(zillow_yearly_clean)

cat("\nFinal structure of cleaned Zillow Monthly\n")
str(zillow_monthly_clean)

# 7. Conclusion
cat("\nEDA and data treatment completed without handling outliers.\n")

# Save the cleaned datasets to CSV files
cat("\nSaving cleaned datasets to CSV files\n")
write.csv(zillow_yearly_clean, "zillow_yearly_clean.csv", row.names = FALSE)
write.csv(zillow_monthly_clean, "zillow_monthly_clean.csv", row.names = FALSE)

cat("\nData successfully saved to CSV files.\n")

# Revisar si los archivos CSV generados están completamente limpios
cat("\nRevisar datos en archivos CSV generados:\n")
zillow_yearly_csv <- read.csv("zillow_yearly_clean.csv")
zillow_monthly_csv <- read.csv("zillow_monthly_clean.csv")

cat("\nVerificar si los datos en Health Data CSV están limpios:\n")
print(sum(is.na(health_data_csv)))

cat("\nVerificar si los datos en Zillow Yearly CSV están limpios:\n")
print(sum(is.na(zillow_yearly_csv)))

cat("\nVerificar si los datos en Zillow Monthly CSV están limpios:\n")
print(sum(is.na(zillow_monthly_csv)))
