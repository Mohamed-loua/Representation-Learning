import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np


def read_json(filename):
    with open(filename,'r') as f:
        data = json.load(f)
        return data


LOG_FOLDER = '/home/mila/c/chris.emezue/representation-learning-assignment/assignment2/assignment/log'


columns = ['Experiment','Experiment Description','Total train time','Avg eval time','Avg. eval acc','Avg. eval f1','Avg train loss','Avg. eval loss']
col1,col2,col3,col4,col5,col6,col7,col8 = [],[],[],[],[],[],[],[]
exp_desc = {
    1:'LSTM, no dropout, encoder only',
    2:'LSTM, dropout, encoder only',
    3:'LSTM, dropout, encoder-decoder, no attention',
    4:'LSTM, dropout, encoder-decoder, with attention',
    5:'Transformer, 2 layers, pre-normalization',
    6:'Transformer, 4 layers, pre-normalization',
    7:'Transformer, 2 layers, post-normalization',
    8:'Fine-tuning BERT'
    }

exps = list(exp_desc.keys())

def get_exp_table():
    for exp in exps:
        exp_log_folder = os.path.join(LOG_FOLDER,f'{exp}')
        args_filename = os.path.join(exp_log_folder,'args.json')
        json_data = read_json(args_filename)
        col1.append(exp)
        col2.append(exp_desc[exp])
        
        total_train_time = sum(json_data['epoch_train_time'])
        col3.append(total_train_time)
        
        avg_eval_time = sum(json_data['epoch_eval_time']) / len(json_data['epoch_eval_time'])
        col4.append(avg_eval_time)
        
        avg_eval_acc = sum(json_data['epoch_eval_acc']) / len(json_data['epoch_eval_acc'])
        col5.append(avg_eval_acc)

        avg_eval_f1 = sum(json_data['epoch_eval_f1']) / len(json_data['epoch_eval_f1'])
        col6.append(avg_eval_f1)
        
        avg_train_loss  = sum(json_data['epoch_train_loss']) / len(json_data['epoch_train_loss'])
        col7.append(avg_train_loss)
        
        avg_eval_loss = sum(json_data['epoch_eval_loss']) / len(json_data['epoch_eval_loss'])
        col8.append(avg_eval_loss)
        


    log_dict = {'Experiment':col1,'Experiment Description':col2,'Total train time':col3,
                'Avg eval time':col4,'Avg. eval acc':col5,'Avg. eval f1':col6,
                'Avg train loss':col7,'Avg. eval loss':col8}

    df = pd.DataFrame(log_dict)
    df.to_csv('log_results.csv',index=False)
    
    
    
# To plot the losses
def get_losses():
    losses=[]
    loss_type_arr = []
    iterations = []
    exps_arr = []
    for exp in exps:
        exp_log_folder = os.path.join(LOG_FOLDER,f'{exp}')
        args_filename = os.path.join(exp_log_folder,'args.json')
        json_data = read_json(args_filename)

        exp_train_losses =  json_data['train_losses']
        exp_eval_losses =  json_data['eval_losses']
        
        total_loss = exp_train_losses + exp_eval_losses
        losses.extend(total_loss)
        
        #loss_types = ['Train loss' for i in range(len(exp_train_losses))] + ['Eval loss' for i in range(len(exp_eval_losses))]
        #loss_type_arr.extend(loss_types)
        exp_train_iters = [i for i in range(100,100*len(exp_train_losses)+1,100)] 
        exp_eval_iters = [i for i in range(100,100*len(exp_eval_losses)+1,100)]
        #exp_exps = [exp for i in range(len(total_loss))]
        
        #iterations.extend(exp_iters)
        #exps_arr.extend(exp_exps)
        
        fig, ax  = plt.subplots()
        
        ax.plot(exp_train_iters,exp_train_losses,label='Train loss')
        ax.plot(exp_eval_iters,exp_eval_losses,label = 'Eval loss')
        #fig.legend()
        plt.legend()
        fig.suptitle(f'Loss plots for experiment {exp}')

        fig.tight_layout()
        fig.savefig(f'exp_figs/experiment_loss_{exp}.png')
        #breakpoint()
        

SAVE_PERTURBED_RESULT = '/home/mila/c/chris.emezue/representation-learning-assignment/assignment2/assignment/model_perturbed_result'
               
def analyse_perturbations():
    ptypes = ['0','A','B','C','D','E','F','G','H','I','J','K']
    exp_predictions = {i:[] for i in range(1,9)}
    exp_predictions['Perturbation'] = ptypes
    for exp in exps:

        for ptype in ptypes:
            exp_log_folder = os.path.join(SAVE_PERTURBED_RESULT,f'{exp}')

            exp_perturbed = os.path.join(SAVE_PERTURBED_RESULT,f'{exp}')
            os.makedirs(exp_perturbed,exist_ok=True)
            save_filename = os.path.join(exp_perturbed,f'{ptype}.txt')
            with open(save_filename,'r') as f:
                text = f.read()
                pred = int(text.strip())
                exp_predictions[exp].append(pred)
    #breakpoint()
    df = pd.DataFrame(exp_predictions)
    df.to_csv('perturbation_predictions.csv',index=False)
if __name__=='__main__':
    #get_losses()
    analyse_perturbations()