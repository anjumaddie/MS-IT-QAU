import numpy as np
import matplotlib.pyplot as plt

# Define sensitivity of the query
sensitivity = 1  # Sensitivity can be considered as 1 since we're dealing with location data

# Define privacy parameter epsilon
epsilon_values = [0.1, 0.3, 0.5, 0.75, 1.0]  # Different levels of privacy

def laplace_mechanism(query_result, sensitivity, epsilon):
    # Generate Laplace noise with scale based on sensitivity and privacy parameter
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    # Add noise to the query result
    return query_result + noise

# Example function to query next station/location (replace this with your actual logic)
def query_next_station(vehicle_data):
    # For demonstration purposes, let's say the next station is based on the average longitude of vehicles
    next_station_longitude = np.mean(vehicle_data['longitude'])
    return next_station_longitude

# Example function to handle LBS request from a vehicle
def handle_lbs_request(vehicle_data, epsilon):
    # Query next station/location
    next_station_longitude = query_next_station(vehicle_data)
    # Apply Laplace mechanism to add noise to the query result
    noisy_next_station_longitude = laplace_mechanism(next_station_longitude, sensitivity, epsilon)
    return next_station_longitude, noisy_next_station_longitude

# Example function to handle LBS response from group leader
def handle_lbs_response(vehicle_data, epsilon):
    # Query next station/location for the response
    response = query_next_station(vehicle_data)
    # Apply Laplace mechanism to add noise to the response
    noisy_response = laplace_mechanism(response, sensitivity, epsilon)
    return response, noisy_response

# Function to calculate accuracy
def calculate_accuracy(original_value, noisy_value):
    return 1 - abs(original_value - noisy_value) / original_value

# Main Function
if __name__ == "__main__":
    # Vehicle data
    vehicle_data = {
        'datetime': "2024-05-13 09:43:42",
        'cell_id': "(3, 5)",
        'vehicle_id': "veh69",
        'speed': 0.0,
        'longitude': [127.03479642747426, 127.02535849939764, 127.02756162079504],  # Example longitude values
        'latitude': [37.49814967247717, 37.50272146446248, 37.49857750954583],  # Example latitude values
    }

    accuracies = []
    noises = []
    for epsilon in epsilon_values:
            print(f"\nLevel of Privacy (Epsilon): {epsilon}")

            # Handle LBS request from a vehicle
            next_station_longitude, noisy_next_station_longitude = handle_lbs_request(vehicle_data, epsilon)
            print("Next station longitude (Vehicle request):", next_station_longitude)
            print("Noisy next station longitude (Vehicle request):", noisy_next_station_longitude)
            
            # Handle LBS response from group leader
            group_leader_response, noisy_group_leader_response = handle_lbs_response(vehicle_data, epsilon)
            print("Group leader response:", group_leader_response)
            print("Noisy group leader response:", noisy_group_leader_response)

            # Calculate and print accuracy
            request_accuracy = calculate_accuracy(next_station_longitude, noisy_next_station_longitude)
            response_accuracy = calculate_accuracy(group_leader_response, noisy_group_leader_response)
            print("Accuracy of vehicle request:", request_accuracy)
            print("Accuracy of group leader response:", response_accuracy)

            # Append accuracies and noises to their respective lists
            accuracies.append((request_accuracy, response_accuracy))
            noises.append((abs(next_station_longitude - noisy_next_station_longitude), abs(group_leader_response - noisy_group_leader_response)))

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_values, [acc[0] for acc in accuracies], label='Request Accuracy')
    plt.plot(epsilon_values, [acc[1] for acc in accuracies], label='Response Accuracy')
    plt.xlabel('Privacy Level (Epsilon)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=False)

    # Plot noise
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_values, [noise[0] for noise in noises], label='Request Noise')
    plt.plot(epsilon_values, [noise[1] for noise in noises], label='Response Noise')
    plt.xlabel('Privacy Level (Epsilon)')
    plt.ylabel('Noise')
    plt.legend()
    plt.show(block=False)

    # Keep the script running until all plot windows are closed
    plt.show()