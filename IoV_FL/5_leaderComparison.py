import matplotlib.pyplot as plt

def read_times(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

def plot_times(times, labels):
    avg_times = [sum(time)/len(time) for time in times]
    plt.bar(labels, avg_times, color=['lightblue', 'orange'], edgecolor='black')
    plt.xlabel('Schemes', fontsize=14)
    plt.ylabel('Average Time Taken (seconds)', fontsize=14)
    plt.title('Average Time Taken by Different Schemes', fontsize=16)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    base_paper_times = read_times("../Base_Code_Generated/times.txt")
    my_paper_times = read_times("times.txt")

    plot_times([base_paper_times, my_paper_times], ['Base Paper', 'Proposed Scheme'])
