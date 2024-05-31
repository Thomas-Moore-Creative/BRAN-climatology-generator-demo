import xarray as xr
import glob
import xarray as xr

# Define the path to your netCDF files
file_pattern = '/path/to/netcdf/files/*.nc'

# Get a list of all netCDF files matching the pattern
file_list = glob.glob(file_pattern)

# Create an empty dataset to store the merged data
merged_dataset = xr.Dataset()

# Loop through each file and merge it into the dataset
for file in file_list:
    # Open the netCDF file as a dataset
    dataset = xr.open_dataset(file)
    
    # Merge the dataset into the merged_dataset
    merged_dataset = xr.merge([merged_dataset, dataset])

# Print the merged dataset
print(merged_dataset)

import matplotlib.pyplot as plt

# Define the variables to plot
variables = ['mean', 'max', 'min', 'std', 'median', '5th quantile', '95th quantile']

# Loop through each variable
for variable in variables:
    # Select the data for the equator
    equator_data = merged_dataset[variable].sel(lat=0)
    
    # Plot the data
    plt.plot(equator_data, label=variable)

# Add labels and legend
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Show the plot
plt.show()


# Print the version of xarray
print(xr.__version__)