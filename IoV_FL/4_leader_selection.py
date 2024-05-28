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