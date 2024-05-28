import pandas as pd
import numpy as np
import math
import ast
import logging
import matplotlib.pyplot as plt
import time


# Start of the section you want to time
start_time = time.time()


def select_cell():
    # Read the data
    vehicle_data = pd.read_csv('vehicle_data.csv')

    # Get the count of vehicles in each cell
    vehicle_counts = vehicle_data.groupby('cell_id').size()

    # Find the cell ID with the maximum number of vehicles
    selected_cell_id = vehicle_counts.idxmax()

    # Save the selected cell ID to a file
    with open('selected_cell_id.txt', 'w') as file:
        file.write(str(selected_cell_id))

    # Print the selected cell ID and the number of vehicles in it
    print(f"Cell ID: {selected_cell_id} having maximum vehicles {vehicle_counts[selected_cell_id]}  is selected.'")

    # New code to filter the DataFrame and write it back to the CSV
    selected_vehicle_data = vehicle_data[vehicle_data['cell_id'] == selected_cell_id]
    selected_vehicle_data.to_csv('selected_vehicle_data.csv', index=False)

    # Print the information
    print(f"Data for {vehicle_counts[selected_cell_id]} vehicles from cell ID {selected_cell_id} is filtered and saved into 'selected_vehicle_data.csv'.")

def read_cell_size_from_file(filename):
    with open(filename, 'r') as file:
        return int(file.read().strip())

def get_cell_data(cell_id, cell_size):
    # Convert the cell ID back to a tuple
    cell_id = ast.literal_eval(cell_id)

    # Calculate the cell center
    cell_center_x = (float(cell_id[0]) * cell_size) + (cell_size / 2)
    cell_center_y = (float(cell_id[1]) * cell_size) + (cell_size / 2)
    cell_center = (cell_center_x, cell_center_y)

    # Calculate the cell radius
    cell_radius = cell_size / 2

    return cell_center, cell_radius

def calculate_distance(vehicle_data, selected_cell_id, cell_size):
    # Get the cell center and radius for the selected cell ID
    cell_center, cell_radius = get_cell_data(selected_cell_id, cell_size)

    # Filter the DataFrame to only include vehicles in the selected cell
    cell_vehicles = vehicle_data[vehicle_data['cell_id'] == selected_cell_id]

    # Initialize a counter for the number of vehicles
    vehicle_count = 0

    # Iterate over all vehicles in the cell
    for index, row in cell_vehicles.iterrows():
        # Calculate the Euclidean distance between the vehicle and the cell center
        raw_distance = math.sqrt((row['longitude'] - cell_center[0])**2 + (row['latitude'] - cell_center[1])**2)

        # Subtract the cell radius from the calculated distance
        distance = raw_distance - cell_radius

        # Save the calculated distance in the DataFrame
        vehicle_data.loc[index, 'distance'] = distance

        # Increment the vehicle counter
        vehicle_count += 1

    # Define the output filename
    output_filename = 'selected_vehicle_data.csv'

    # Save the updated DataFrame to a CSV file
    vehicle_data.to_csv(output_filename, index=False)

    # Print the information
    print(f"Distances of {vehicle_count} vehicles from cell ID {selected_cell_id} are calculated and saved into '{output_filename}'.")

    return vehicle_data



def evaluate_resources(vehicle_data, selected_cell_id):
    # Get the rows for the vehicles in the selected cell from the DataFrame
    vehicle_rows = vehicle_data[vehicle_data['cell_id'] == selected_cell_id]

    # Define the weight factors for each component of the resources evaluation
    processing_power_weight = 0.5
    available_storage_weight = 0.5

    # Iterate over each vehicle in the cell
    for index, vehicle_row in vehicle_rows.iterrows():
        # Get the processing power and available storage for the vehicle from the DataFrame
        processing_power = vehicle_row['processing_power']
        available_storage = vehicle_row['available_storage']

        # Calculate the resources evaluation
        resources = (processing_power_weight * processing_power +
                     available_storage_weight * available_storage)

        # Save the calculated resources in the DataFrame
        vehicle_data.loc[index, 'resources'] = resources

    # Define the output filename
    output_filename = 'selected_vehicle_data.csv'

    # Save the updated DataFrame to a CSV file
    vehicle_data.to_csv(output_filename, index=False)

    # Print the information
    print(f"Resources of {len(vehicle_rows)} vehicles from cell ID {selected_cell_id} are calculated and saved into '{output_filename}'.")

    return vehicle_data

# Call the functions
select_cell()

# Load the selected vehicle data
vehicle_data = pd.read_csv('selected_vehicle_data.csv')

# Read the selected cell ID from the DataFrame
selected_cell_id = vehicle_data['cell_id'].unique()[0]


# Read cell size from the file
cell_size = read_cell_size_from_file('cell_size.txt')

calculate_distance(vehicle_data, selected_cell_id, cell_size)

# Call the new function
evaluate_resources(vehicle_data, selected_cell_id)




import pandas as pd
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Define constants
INITIAL_MAX_VALUE = -1
DATA_FILE = 'selected_vehicle_data.csv'
LEADER_INFO_FILE = 'leader_info.txt'

# Load the selected vehicle data
global_vehicle_data = pd.read_csv(DATA_FILE)

# Read the selected cell ID from the DataFrame
selected_cell_id = global_vehicle_data['cell_id'].unique()[0]


def calculate_total_value(row):
    """Calculate the total value for a vehicle."""
    Centerness = 1 / row['distance'] if row['distance'] != 0 else float('inf')
    traffic_rules_obeyed = int(row['traffic_rules_obeyed'])  # Convert boolean to integer
    Resources = row['resources']
    return traffic_rules_obeyed + Centerness + Resources


def select_group_leader(vehicle_data: pd.DataFrame, selected_cell_id: int) -> pd.DataFrame:
    """Select the group leader from the vehicle data."""
    vehicle_data = vehicle_data.assign(group_leader=False)

    max_total_value = INITIAL_MAX_VALUE
    group_leader_id = None

    for index, row in vehicle_data.iterrows():
        total_value = calculate_total_value(row)
        if total_value > max_total_value:
            max_total_value = total_value
            group_leader_id = row['vehicle_id']

    vehicle_data.loc[vehicle_data['vehicle_id'] == group_leader_id, 'group_leader'] = True

    return vehicle_data, group_leader_id


def save_data(vehicle_data: pd.DataFrame, group_leader_id: int):
    """Save the vehicle data and leader info to files."""
    try:
        vehicle_data.to_csv(DATA_FILE, index=False)
        with open(LEADER_INFO_FILE, 'w') as file:
            file.write(f"Group Leader Cell ID: {selected_cell_id}\n")
            file.write(f"Number of vehicles in cell: {len(vehicle_data)}\n")
            file.write(f"Group Leader ID: {group_leader_id}\n")
    except Exception as e:
        logging.error(f"Error saving data: {e}")


def plot_vehicle_data(vehicle_data: pd.DataFrame, group_leader_id: int):
    """Plot the vehicle data and highlight the group leader."""
    plt.figure(figsize=(10, 6))

    # Plot all vehicles
    plt.scatter(vehicle_data['distance'], vehicle_data['vehicle_id'], label='Vehicles')

    # Highlight the group leader
    leader_data = vehicle_data[vehicle_data['vehicle_id'] == group_leader_id]
    plt.scatter(leader_data['distance'], leader_data['vehicle_id'], color='red', label='Group Leader')

    plt.xlabel('Distance from Center')
    plt.ylabel('Vehicle ID')
    plt.title('Vehicle Distribution with Group Leader')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    vehicle_data, group_leader_id = select_group_leader(global_vehicle_data, selected_cell_id)
    logging.info(f"Group Leader Cell ID: {selected_cell_id}")
    logging.info(f"Number of vehicles in cell: {len(vehicle_data)}")
    logging.info(f"Group Leader ID: {group_leader_id}")
    save_data(vehicle_data, group_leader_id)
    plot_vehicle_data(vehicle_data, group_leader_id)


if __name__ == "__main__":
    main()


# End of the section you want to time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Write elapsed time to file
with open('times.txt', 'a') as f:
    f.write(str(elapsed_time) + '\n')