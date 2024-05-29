import os,json
from matplotlib import pyplot as plt
# %matplotlib inline
import numpy as np
import pickle as pkl
import pandas as pd
from copy import deepcopy
import seaborn as sns
from scipy import stats
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader,WeightedRandomSampler,RandomSampler

# Check if CUDA is available (for GPU usage)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Constants
- TIME_PERIODS: the length of the time segment
- STEP_DISTANCE: the amount of overlap between two consecutive time segments
"""

BATCH_SIZE = 8
FIGURE_FOLDER = '/home/mila/c/chris.emezue/representation-learning-assignment/figs/random-sampler/'
if not os.path.exists(FIGURE_FOLDER):
  os.makedirs(FIGURE_FOLDER)

sns.set() # Default seaborn look and feel
plt.style.use('ggplot')

# Class labels
LABELS = ['Downstairs',
          'Jogging',
          'Sitting',
          'Standing',
          'Upstairs',
          'Walking']

TIME_PERIODS = 80  # The number of steps within one time segment

# The steps to take from one segment to the next; if this value is equal to TIME_PERIODS,
# then there is no overlap between the segments
STEP_DISTANCE = 40

"""# Data

I used `/kaggle/input/activitydetectionimusensor/WISDM_ar_v1.1.1_raw.txt` data.
> (another dataset (later): `/kaggle/input/human-activity-recognition/time_series_data_human_activities.csv`)
"""

# Define some functions to read the data and show some basic info about the data

def read_data(file_path):
    column_names = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv(file_path, header=None, names=column_names)
    # Last column has a ";" character which must be removed
    df['z-axis'].replace(regex=True,
                         inplace=True,
                         to_replace=r';',
                         value=r'')
    # Transform 'z-axis' column to float
    df['z-axis'] = df['z-axis'].apply(convert_to_float)

    df.dropna(axis=0, how='any', inplace=True)  # Drop NaN values

    return df

def convert_to_float(x):
    try:
        return np.float64(x)
    except:
        return np.nan


def save_json(obj,filename):
  with open(filename,'w+') as f:
    json.dump(obj,f)
    
def read_json(filename):
  with open(filename,'r') as f:
    data = json.load(f)
    return data

def show_basic_dataframe_info(dataframe):
    # Shape: #_rows, #_columns
    print("Number of rows in the dataframe: %i" % (dataframe.shape[0]))
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))

"""The data can be downloaded from **[here](https://www.kaggle.com/datasets/sosoyeong/wisdm-raw)**.

Then 'file_path' the directory to where you have put the data.
"""

#show_basic_dataframe_info(df)

# by activity type
#df['activity'].value_counts().plot(kind='bar',title='Training Examples by Activity Type')

# by user
#df['user'].value_counts().plot(kind='bar',title='Training Examples by User')

"""- We have more data for walking and jogging activities more than other activities.
- 36 participants

<br>

**Accelerometer data** for six activities
- sampling rates 20Hz (20 values per second)
    - first 180 records == 9 second interval
        - 1/200 * 180 = 9 seconds
"""

def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'X-Axis')  # x
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'Y-Axis')  # y
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'Z-Axis')  # z
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'r')
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_visible(False)

    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)



def create_segments_and_labels(df, time_steps, step, label_name):
    # x, y, z acceleration as features
    N_FEATURES = 3

    # Number of steps to advance in each iteration
    # step = time_steps # no overlap between segments

    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]

        # find the most often used label in this segment
        label_mode_result = stats.mode(df[label_name][i: i + time_steps])
        if np.isscalar(label_mode_result.mode):
            label = label_mode_result.mode
        else:
            label = label_mode_result.mode[0]

        segments.append([xs, ys, zs])
        labels.append(label)

    # bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels  # x, y

def to_categorical(y, num_classes):
    return torch.eye(num_classes)[y]


def save_pickle(obj,filename):
  with open(filename,'wb+') as f:
    pkl.dump(obj,f)
def load_pickle(filename):
  with open(filename,'rb') as f:
    data = pkl.load(f)
    return data



# Save the tensors in pickle file, if they don't exist already
if not os.path.exists('data/x_train_tensor.pkl'):

  file_path = '/home/mila/c/chris.emezue/representation-learning-assignment/WISDM_ar_v1.1_raw.txt'
  df = read_data(file_path)


  for activity in np.unique(df['activity']):
      subset = df[df['activity'] == activity][:180]  # check only for first 180 records (9 seconds)
      plot_activity(activity, subset)


  LABEL = 'ActivityEncoded'

  le = preprocessing.LabelEncoder()  # string to Integer
  df[LABEL] = le.fit_transform(df['activity'].values.ravel())

  df.head()

  # Split data into train and test set
  # train: user 1 ~ 28
  # test: user 28 ~

  df_train = df[df['user'] <= 28]
  df_test = df[df['user'] > 28]

  # normalize train data (value range: 0 ~ 1)
  # normalization should be applied to test data in the same way
  pd.options.mode.chained_assignment = None  # defual='warm'

  df_train['x-axis'] = df_train['x-axis'] / df_train['x-axis'].max()
  df_train['y-axis'] = df_train['y-axis'] / df_train['y-axis'].max()
  df_train['z-axis'] = df_train['z-axis'] / df_train['z-axis'].max()


  # round numbers
  df_train = df_train.round({'x-axis':4, 'y-axis':4, 'z-axis': 4})
  df_train.head()


  # normalize test data

  df_test['x-axis'] = df_test['x-axis'] / df_test['x-axis'].max()
  df_test['y-axis'] = df_test['y-axis'] / df_test['y-axis'].max()
  df_test['z-axis'] = df_test['z-axis'] / df_test['z-axis'].max()

  df_test = df_test.round({'x-axis':4, 'y-axis':4, 'z-axis': 4})

  """Still the dataframe is not ready yet to be fed into a neural network.

  So, we need to reshpae it.
  """

  # 80 steps => 4 sec (0.05 * 80 = 4)

  x_train, y_train = create_segments_and_labels(df_train,
                                                TIME_PERIODS,
                                                STEP_DISTANCE,
                                                LABEL)  # LABEL = 'ActivityEncoded'

  x_test, y_test = create_segments_and_labels(df_test,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)


  print('x_train shape: ', x_train.shape)
  print(x_train.shape[0], 'training samples')
  print('y_train shape: ', y_train.shape)

  """`x_train` has 20868 records of 2D-matrix of shape 80x3.

  **Dimensions we need to remeber**

  - #_time periods: the number of time periods within 1 record
      - 4 second interval => 80
  - #_sensors: 3 (x, y, z axis acceleration)
  - #_classes: the number of the nodes for output layer -> 6
  """


  """The input data is 2D (80x3).

  """

  input_shape = (TIME_PERIODS * 3)
  x_train = x_train.reshape(x_train.shape[0], input_shape)
  x_test = x_test.reshape(x_test.shape[0], input_shape)
  x_test = x_test.astype('float32')
  x_train = x_train.astype('float32')

  y_train = y_train.astype('float32')
  y_test = y_test.astype('float32')



  #y_train_hot = to_categorical(y_train, n_classes)
  #y_test = to_categorical(y_test, n_classes)
  #print('New y_train shape: ', y_train_hot.shape)

  """In PyTorch, we need to wrap these NumPy arrays into a dataset and then create a DataLoader for batch processing."""

  # Convert your numpy arrays to PyTorch tensors
  x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # long for CrossEntropyLoss
  x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
  y_test_tensor = torch.tensor(y_test, dtype=torch.long)

  # Create a small validation set

  x_train_tensor, x_valid_tensor, y_train_tensor, y_valid_tensor = train_test_split(x_train_tensor, y_train_tensor, test_size=0.20, random_state=42)


  save_pickle(x_train_tensor,'data/x_train_tensor.pkl')
  save_pickle(y_train_tensor,'data/y_train_tensor.pkl')
  save_pickle(x_valid_tensor,'data/x_valid_tensor.pkl')
  save_pickle(y_valid_tensor,'data/y_valid_tensor.pkl')
  save_pickle(x_test_tensor,'data/x_test_tensor.pkl')
  save_pickle(y_test_tensor,'data/y_test_tensor.pkl')
  save_pickle(y_test,'data/y_test.pkl')

else: 
  # Load saved pickle files
  print(f'Our input files already exist so we are reading them...')
  x_train_tensor = load_pickle('data/x_train_tensor.pkl')
  y_train_tensor = load_pickle('data/y_train_tensor.pkl')
  x_valid_tensor = load_pickle('data/x_valid_tensor.pkl')
  y_valid_tensor = load_pickle('data/y_valid_tensor.pkl')
  x_test_tensor = load_pickle('data/x_test_tensor.pkl')
  y_test_tensor = load_pickle('data/y_test_tensor.pkl')
  y_test = load_pickle('data/y_test.pkl')

n_sensors = 3
n_time_periods = TIME_PERIODS
n_classes = len(LABELS)  # Assuming y_train is available and contains your class labels



def show_confusion_matrix(validaitons, predictions, title=None,filename=None):
    assert filename is not None, '`filename` needs to be defined'
    matrix = metrics.confusion_matrix(validaitons, predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    if title: plt.title(title)
    else: plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

def plot_perfomance(train_losses, val_losses,filename):
  # Assuming you have lists `train_losses` and `val_losses` containing loss values
  plt.figure(figsize=(10, 5))
  plt.title("Training and Validation Loss")
  plt.plot(val_losses, label="Validation")
  plt.plot(train_losses, label="Training")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")  
  plt.tight_layout()
  plt.legend()
  plt.savefig(filename)
  #plt.show()

def compute_accuracy(pred,target):
    """Find the number of correct instances (where pred class)
    is equal to target class. Then divide this number by the
    total number of instances there is.
    """

    pred_arg = pred.argmax(dim=1)
    correct = sum(torch.eq(pred_arg,target))
    return correct / target.shape[0]



def train(model, device, num_epochs, optimizer,train_loader,val_loader,criterion,train_losses,val_losses,train_accs,val_accs):
    print_every_epoch = 5
    patience = 10
    model.train()
    main_loss,main_acc=0,0
    best_acc = 0
    best_model_weights = None
    count_patience = 0
    # Initialize lists to store losses and accuracies
    for epoch in range(num_epochs):
      for train_batch in train_loader:
        input, label = train_batch
        input, label = input.to(device), label.to(device)
        pred = model(input)
        #breakpoint()
        # Compute loss
        loss = criterion(pred,label)
        main_loss+=loss.item()
        # Compute accuracy
        train_acc = compute_accuracy(pred,label).item()
        main_acc += train_acc

        # Compute gradient and update params
        loss.backward()
        optimizer.step()


      train_losses.append(main_loss / len(train_loader))
      train_accs.append(main_acc/len(train_loader))
      val_loss, val_acc = validate(model,device,val_loader,criterion)
      val_losses.append(val_loss)
      val_accs.append(val_acc)
      main_loss,main_acc=0,0

      if val_acc > best_acc:
        best_model_weights = deepcopy(model.state_dict())
        print("using model weights at best val accuracy")
        best_acc = val_acc
      else:
        count_patience+=1


      if epoch % print_every_epoch ==0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Train acc: {train_acc} | Val acc: {val_acc}")
      if count_patience>=patience:
        print('Early stopping...')
        break

      
    model.load_state_dict(best_model_weights)
    
def validate(model, device,loader,criterion):
  model.eval()
  main_loss,main_acc=0,0
  for batch in loader:
    input, label = batch
    input, label = input.to(device), label.to(device)
    pred = model(input)

    # Compute loss
    loss = criterion(pred,label)
    main_loss += loss.item()

    # Compute accuracy
    acc = compute_accuracy(pred,label).item()
    main_acc += acc
  return main_loss/len(loader), main_acc / len(loader)


def test(model, device,loader,criterion):
  model.eval()
  main_loss,main_acc=0,0
  for batch in loader:
    input, label = batch
    #print(input.shape)
    input, label = input.to(device), label.to(device)
    pred = model(input)
    #breakpoint()
    # Compute loss
    loss = criterion(pred,label)
    main_loss += loss.item()

    # Compute accuracy
    acc = compute_accuracy(pred,label).item()
    main_acc += acc
  return main_loss/len(loader), main_acc / len(loader)