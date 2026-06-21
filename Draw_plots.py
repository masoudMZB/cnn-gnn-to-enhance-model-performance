import pickle
import numpy as np
import matplotlib.pyplot as plt

def select_best_run_final_epoch(logs, is_loss=True):
    selected_runs = []
    for log in logs:
        # Choose the run with the minimum/maximum value in the last epoch
        best_run_index = np.argmin([run[-1] for run in log]) if is_loss else np.argmax([run[-1] for run in log])
        selected_runs.append(log[best_run_index])
    return selected_runs

def select_best_run_overall(logs_train, logs_test, is_loss=True):
    selected_runs_train = []
    selected_runs_test = []
    for log, train_log in zip(logs_test, logs_train):
        # Choose the run with the minimum/maximum average value based on test logs
        best_run_index = np.argmin([np.mean(run) for run in log]) if is_loss else np.argmax([np.mean(run) for run in log])
        selected_runs_test.append(log[best_run_index])
        selected_runs_train.append(train_log[best_run_index])
        print(f'train was {len(train_log[best_run_index])}')
        print(f'acc was was {(log[best_run_index][-1])}')
    return selected_runs_test, selected_runs_train

def mean_performance(logs):
    mean_runs = []
    for log in logs:
        mean_run = np.mean(log, axis=0)  # Average over runs
        mean_runs.append(mean_run)
    return mean_runs

def plot_logs(logs1, logs2, title1='Experiment 1', title2='Experiment 2'):
    # Assuming logs1 and logs2 are structured as:
    # [train_losses, train_accuracies, test_losses, test_accuracies]

    # Plotting training and testing accuracy and loss for Experiment 1
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    for i, acc in enumerate(logs1[1]):  # Train accuracies
        plt.plot(acc, label=f'Run {i+1}')
    plt.title(f'{title1} - Training Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(2, 2, 2)
    for i, loss in enumerate(logs1[0]):  # Train losses
        plt.plot(loss, label=f'Run {i+1}')
    plt.title(f'{title1} - Training Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plotting training and testing accuracy and loss for Experiment 2
    plt.subplot(2, 2, 3)
    for i, acc in enumerate(logs2[1]):  # Train accuracies
        plt.plot(acc, label=f'Run {i+1}')
    plt.title(f'{title2} - Training Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(2, 2, 4)
    for i, loss in enumerate(logs2[0]):  # Train losses
        plt.plot(loss, label=f'Run {i+1}')
    plt.title(f'{title2} - Training Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        with open('/content/drive/MyDrive/tmp/logsconcat5.pkl', 'rb') as file:
                logsconcat5 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logsconcat10.pkl', 'rb') as file:
                logsconcat10 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logsconcat32.pkl', 'rb') as file:
                logsconcat32 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logssq1.pkl', 'rb') as file:
                logssq1 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logssq5.pkl', 'rb') as file:
                logssq5 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logssq32.pkl', 'rb') as file:
                logssq32 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logssq100.pkl', 'rb') as file:
                logssq100 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logssq200.pkl', 'rb') as file:
                logssq200 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logssq300.pkl', 'rb') as file:
                logssq300 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logsef1.pkl', 'rb') as file:
                logsef1 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logsef5.pkl', 'rb') as file:
                logsef5 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logsef32.pkl', 'rb') as file:
                logsef32 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logsef100.pkl', 'rb') as file:
                logsef100 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logsef200.pkl', 'rb') as file:
                logsef200 = pickle.load(file)

        with open('/content/drive/MyDrive/tmp/logsef300.pkl', 'rb') as file:
                logsef300 = pickle.load(file)

        logs_train_acc = [logssq1[1], logssq5[1], logssq32[1], logssq100[1], logssq200[1], logssq300[1],
                           logsef1[1], logsef5[1], logsef32[1], logsef100[1], logsef200[1], logsef300[1],
                           logsconcat5[1], logsconcat10[1], logsconcat32[1]
                           ]


        logs_test_acc = [logssq1[3], logssq5[3], logssq32[3], logssq100[3], logssq200[3], logssq300[3],
                           logsef1[3], logsef5[3], logsef32[3], logsef100[3], logsef200[3], logsef300[3],
                           logsconcat5[3], logsconcat10[3], logsconcat32[3]
                           ]

        logs_exp_names = ['logssq1', 'logssq5','logssq32', 'logssq100','logssq200', 'logssq300',
                          'logsef1', 'logsef5','logsef32', 'logsef100','logsef200', 'logsef300',
                          'logsconcat5', 'logsconcat10','logsconcat32',
                          ]
        logs_exp_marker = ['.', 'p','o', 'x','v', '<',
                           '.', 'p','o', 'x','v', '<',
                           '.', 'p','o'
                           ]

        best_overall_runs_test, best_overall_runs_train = select_best_run_overall(logs_train_acc,logs_test_acc, is_loss=False)

        # Plotting
        plt.figure(figsize=(14, 8))
        for i, run_loss in enumerate(best_overall_runs_test):
            epochs = range(1, len(run_loss) + 1)
            plt.plot(epochs, run_loss, label=f'Experiment {logs_exp_names[i]}', marker=f'{logs_exp_marker[i]}')

        plt.title('TEST accurcay for Different Experiments')
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

        # Plotting
        plt.figure(figsize=(14, 8))
        for i, run_loss in enumerate(best_overall_runs_train):
            epochs = range(1, len(run_loss) + 1)
            plt.plot(epochs, run_loss, label=f'Experiment {logs_exp_names[i]}', marker=f'{logs_exp_marker[i]}')

        plt.title('TRAIN accurcay for Different Experiments')
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
    except FileNotFoundError:
        print("Log files not found. Skipping plot generation.")
