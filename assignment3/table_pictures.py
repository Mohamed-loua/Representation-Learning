import os




path = '/home/mila/c/chris.emezue/representation-learning-assignment/assignment3'

images = []

#breakpoint()
for epoch in range(20):
    filename = f'genImages/epoch_{epoch}_sample_999.png'


    s='''

    \\begin{'''+'''figure'''+'''}
        \centering
        \includegraphics{'''+filename+'''}
        \caption{Generated images at epoch '''+str(epoch)+'''}
        \label{fig:genimg-'''+f'''{str(epoch)}'''+'''}
    \end{'''+'''figure}
    '''
    #breakpoint()
    images.append(s)
final_string  = '\n\n'.join(images)

with open('latex_image_figures.txt','w+') as f:
    f.write(final_string)
    