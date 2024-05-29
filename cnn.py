from utils import *
import sys
args = sys.argv[1:]
sample = args[0]
#sample='weighted'
#assert len(args)==2, 'The arguments passed should be two for optimizer and lr'

#FIGURE_FOLDER = os.path.join(FIGURE_FOLDER,f'{args[0]}-{args[1]}-cnn/')
FIGURE_FOLDER = os.path.join(FIGURE_FOLDER,f'cnn-{sample}')
if not os.path.exists(FIGURE_FOLDER):
  os.makedirs(FIGURE_FOLDER)


# Create TensorDatasets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor,y_test_tensor )
valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)

if sample=='random':
  sampler= RandomSampler(train_dataset,replacement=True)
else:
  print('Using weighted sampling')
  label_count = torch.unique(train_dataset.tensors[1],return_counts=True)
  label_count_dict = {k.item():v.item() for k,v in zip(label_count[0],label_count[1])}
  label_weight_dict = {k:1/v for k,v in label_count_dict.items()}

  sample_weights = torch.tensor([label_weight_dict[i.item()] for i in train_dataset.tensors[1]])
  sampler = WeightedRandomSampler(sample_weights,len(train_dataset))

# Create DataLoaders
batch_size = BATCH_SIZE  # You can change this value as per your need
train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print(f'Length of validation data: ',len(valid_dataset))

"""---

# 1D CNN

* Reference article: https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
* Reference code: https://github.com/ni79ls/har-keras-cnn/blob/master/20180903_Keras_HAR_WISDM_CNN_v1.0_for_medium.py

 A 1D CNN is very effective when you expect to derive interesting features from shorter (fixed-length) segments of the overall data set and where the location of the feature within the segment is not of high relevance.


This applies well to the analysis of time sequences of sensor data (such as gyroscope or accelerometer data). It also applies to the analysis of any kind of signal data over a fixed-length period (such as audio signals).
"""

class MyConvModel(nn.Module):
    def __init__(self, time_periods, n_sensors, n_classes):
        super(MyConvModel, self).__init__()
        self.time_periods = time_periods
        self.n_sensors = n_sensors
        self.n_classes = n_classes
        self.kernel_size = 5

        self.conv1 = nn.Conv1d(self.n_sensors, 100, self.kernel_size, stride=1, padding='valid')
        self.conv2 = nn.Conv1d(100, 100, self.kernel_size, stride=1, padding='valid')
        #self.batchnorm2 =  nn.BatchNorm1d(100)

        self.maxpool1 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(100, 160, self.kernel_size, stride=1, padding='valid')
        #self.batchnorm3 =  nn.BatchNorm1d(160)
        self.conv4 = nn.Conv1d(160, 160, self.kernel_size, stride=1, padding='valid')
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(160,self.n_classes)
        self.activation = nn.ReLU()
        #self.activation = nn.LeakyReLU()

    def forward(self, x):
        # Reshape the input to (batch_size, n_sensors, time_periods)
        #breakpoint()
        x = x.reshape(x.shape[0],self.n_sensors,self.time_periods)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        #x = self.conv_add(x)
        x = self.maxpool1(x)
        x = self.activation(self.conv3(x))
        x_last = self.activation(self.conv4(x))

        x_avg = self.avgpool1(x_last)
        x = self.dropout(x_avg)
        x = x.squeeze() # squeeze out the time dimension that is 1

        x = self.linear(x)
        x = F.log_softmax(x)
        return x

class MyConvModelDownScaled(nn.Module):
    def __init__(self, time_periods, n_sensors, n_classes):
        super(MyConvModelDownScaled, self).__init__()
        self.time_periods = time_periods
        self.n_sensors = n_sensors
        self.n_classes = n_classes
        self.kernel_size = 3

        self.conv1 = nn.Conv1d(self.n_sensors, 50, self.kernel_size, stride=1, padding='valid')
        self.conv2 = nn.Conv1d(50, 50, self.kernel_size, stride=1, padding='valid')
        #self.batchnorm2 =  nn.BatchNorm1d(100)

        self.maxpool1 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(50, 80, self.kernel_size, stride=1, padding='valid')
        #self.batchnorm3 =  nn.BatchNorm1d(160)
        self.conv4 = nn.Conv1d(80, 80, self.kernel_size, stride=1, padding='valid')
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(eval(args[0]))
        self.linear = nn.Linear(80,self.n_classes)
        self.activation = nn.ReLU()
        #self.activation = nn.LeakyReLU()

    def forward(self, x):
        # Reshape the input to (batch_size, n_sensors, time_periods)
        #breakpoint()
        x = x.reshape(x.shape[0],self.n_sensors,self.time_periods)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        #x = self.conv_add(x)
        x = self.maxpool1(x)
        x = self.activation(self.conv3(x))
        x_last = self.activation(self.conv4(x))

        x_avg = self.avgpool1(x_last)
        #breakpoint()
        x = self.dropout(x_avg)
        x = x.squeeze() # squeeze out the time dimension that is 1
        #breakpoint()
        x = self.linear(x)
        x = F.log_softmax(x)
        return x



# Assuming TIME_PERIODS, n_sensors, and n_classes are defined
model_cnn = MyConvModel(TIME_PERIODS, n_sensors, n_classes)
#model_cnn = MyConvModelDownScaled(TIME_PERIODS, n_sensors, n_classes)

# Move the model to the device (CPU or GPU)
model_cnn.to(device)

# Print model summary
print(model_cnn)
# Optimizer
criterion = nn.NLLLoss() # since we already have softmax in the forward pass

# Choose your Optimizer
optim = torch.optim.AdamW(model_cnn.parameters(),lr=0.0001, weight_decay=0.01)

# if args[0]=='adamw':
#   optim = torch.optim.AdamW(model_cnn.parameters(),lr=eval(args[1]), weight_decay=0.01)
# elif args[0]=='adam':
#   optim = torch.optim.Adam(model_cnn.parameters(),lr=eval(args[1]), weight_decay=0.01)
# else:
#   raise Exception('argument for optmiizer should be `adam` or `adamw`!')
#best_val_loss = float('inf')
patience = 100
trigger_times = 0

# Initialize lists to store losses and accuracies
train_losses = []
val_losses = []
train_accs = []
val_accs = []

print('Training the model...')
EPOCHS = 500
train(model_cnn, device, EPOCHS, optim,train_loader,val_loader,criterion,train_losses,val_losses,train_accs,val_accs)
print(f'Training ended.')

# Save model dict
model_path = os.path.join(FIGURE_FOLDER,'cnn_model_dict.pt')
torch.save(model_cnn.state_dict(), model_path)

# Plotting performance
save_json(train_losses,os.path.join(FIGURE_FOLDER,'cnn_train_losses.json'))
save_json(val_losses,os.path.join(FIGURE_FOLDER,'cnn_val_losses.json'))


# Losses and accuracy plots
loss_figure_name = os.path.join(FIGURE_FOLDER,'cnn_loss_plot.png')
plot_perfomance(train_losses,val_losses,loss_figure_name)

acc_figure_name = os.path.join(FIGURE_FOLDER,'cnn_acc_plot.png')
plot_perfomance(train_accs,val_accs,acc_figure_name)


# Testing

test_loss, test_acc = test(model_cnn,device,test_loader,criterion)
test_details_filename = os.path.join(FIGURE_FOLDER,'cnn_test_details.txt')
with open(test_details_filename,'w+') as f:
  f.write(f'Accuracy on test data: {test_acc}'+'\n')
  f.write(f'Loss on test data: {test_loss}')



y_test = to_categorical(y_test_tensor, n_classes)
y_pred_test = model_cnn(x_test_tensor.to(device)).detach().cpu()
max_y_pred_test = np.argmax(y_pred_test, axis=1)

max_y_test = np.argmax(y_test, axis=1)

cm_figure_name = os.path.join(FIGURE_FOLDER,'cnn_cm.png')
show_confusion_matrix(max_y_test, max_y_pred_test,filename=cm_figure_name)

print(classification_report(max_y_test, max_y_pred_test))