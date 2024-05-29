from utils import *
import sys
args = sys.argv[1:]
sample = args[0]
#sample='random'

#assert len(args)==2, 'The arguments passed should be two for optimizer and lr'

#FIGURE_FOLDER = os.path.join(FIGURE_FOLDER,f'{args[0]}-{args[1]}-mlp')
FIGURE_FOLDER = os.path.join(FIGURE_FOLDER,f'mlp-{sample}')
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print(f'Length of validation data: ',len(valid_dataset))

"""# Multi-layer Perceptron"""

class MyModel(nn.Module):
    def __init__(self, time_periods, n_classes):
        super(MyModel, self).__init__()
        self.time_periods = time_periods
        self.n_classes = n_classes

        self.activation = nn.ReLU()        
        self.group = nn.Sequential(nn.Linear(in_features=100,out_features=100),self.activation)
        self.flatten_layer = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.time_periods*3,out_features=100)
        self.fc2 = nn.Sequential(*[self.group for i in range(2)])
        #self.fc3 = nn.Linear(in_features=100,out_features=100)
        self.fc4 = nn.Linear(in_features=100,out_features=self.n_classes)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.flatten_layer(x)
        x = self.activation(self.fc1(x))
        #x = self.activation(self.fc2(x))
        x = self.fc2(x)

        #x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = F.log_softmax(x)
        return x
# Assuming TIME_PERIODS and n_classes are defined
model_mlp = MyModel(TIME_PERIODS, n_classes)
model_mlp.to(device)

# Print model summary
print(model_mlp)
#breakpoint()



# Use Pytorch's cross entropy Loss function for a classification task
criterion = nn.NLLLoss() # since we already have softmax in the forward pass


# Choose your Optimizer
my_optimizer = torch.optim.AdamW(model_mlp.parameters(),lr=0.0001, weight_decay=0.01)
# Choose your Optimizer
# if args[0]=='adamw':
#   my_optimizer = torch.optim.AdamW(model_mlp.parameters(),lr=eval(args[1]), weight_decay=0.01)
# elif args[0]=='adam':
#   my_optimizer = torch.optim.Adam(model_mlp.parameters(),lr=eval(args[1]), weight_decay=0.01)
# else:
#   raise Exception('argument for optmiizer should be `adam` or `adamw`!')



train_losses = []
val_losses = []
train_accs = []
val_accs = []
print_every_epoch = 5


# Added code to only take the weights of the best performing model,
# after a val evaluation

EPOCHS = 500
train(model_mlp, device, EPOCHS,my_optimizer,train_loader,val_loader,criterion,train_losses,val_losses,train_accs,val_accs)


# Save model dict
model_path = os.path.join(FIGURE_FOLDER,'mlp_model_dict.pt')
torch.save(model_mlp.state_dict(), model_path)


# Losses and accuracy plots
loss_figure_name = os.path.join(FIGURE_FOLDER,'mlp_loss_plot.png')
plot_perfomance(train_losses,val_losses,loss_figure_name)

acc_figure_name = os.path.join(FIGURE_FOLDER,'mlp_acc_plot.png')
plot_perfomance(train_accs,val_accs,acc_figure_name)

"""Result from the article

![Expectation](attachment:16797bb4-c2ae-4f1b-8a7a-e195e39da9c3.png)

## Test
"""

test_loss, test_acc = test(model_mlp,device,test_loader,criterion)
test_details_filename = os.path.join(FIGURE_FOLDER,'mlp_test_details.txt')
with open(test_details_filename,'w+') as f:
  f.write(f'Accuracy on test data: {test_acc}'+'\n')
  f.write(f'Loss on test data: {test_loss}')


# Performing categorical encoding on y_test for confusion matrix
y_test = to_categorical(y_test, n_classes)

print('Accuracy on validation data: ', val_accs[-1])
print('Loss on validation data: ', val_losses[-1])

"""The test accuray is **about 75%**."""




y_test = to_categorical(y_test_tensor, n_classes)    
y_pred_test = model_mlp(x_test_tensor.to(device)).detach().cpu()
max_y_pred_test = np.argmax(y_pred_test, axis=1)

max_y_test = np.argmax(y_test, axis=1)

cm_figure_name = os.path.join(FIGURE_FOLDER,'mlp_cm.png')

show_confusion_matrix(max_y_test, max_y_pred_test,filename=cm_figure_name)

print(classification_report(max_y_test, max_y_pred_test))
