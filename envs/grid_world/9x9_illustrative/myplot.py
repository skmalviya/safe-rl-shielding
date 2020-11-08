# -*- coding: utf-8 -*-
'''
Written on 19 July 2019
@author : Shrikant Malviya
'''

#Help
#Most Important : https://towardsdatascience.com/a-step-by-step-guide-for-creating-advanced-python-data-visualizations-with-seaborn-matplotlib-1579d6a1a7d0
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
#https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
#https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
#https://python-graph-gallery.com/11-grouped-barplot/

# libraries
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import matplotlib.font_manager as mfm

font_path = '/home/shrikant/Lohit-Devanagari.ttf'
prop = mfm.FontProperties(fname=font_path) # find this font

####### Full Marker List #########
### ('.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'None', None, ' ', '')
def myplot(data,index,file_name):
    # Markers
    marker = itertools.cycle(('o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'None', None, ' ', '')) 
    # Colors
    colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))
    # X-axis
    x = range(1,len(data[list(data)[0]])+1)
    fig = plt.figure()
    print (list(data),x)
    # multiple line plot
    for key in data.keys():
        #plt.plot( x, data[i], marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
        #plt.plot( x, data[key], marker = next(marker), color='black', markersize=12, linewidth=1, label=key)
        plt.plot( index, data[key], linewidth=2, label=key, marker = next(marker), color=next(colors))
        #plt.plot( x, data[i], marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
    #plt.legend()
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    fig.savefig(file_name+'.pdf', bbox_inches='tight')
    #plt.show()

def bar_plot(data,index,file_name):
    """
    given a dictionary of data in list format, it plot them in a GROUPED BAR-PLOT

    Arguments
    ---------
    data:       dictionary of lists who keys are the label and 
                the values in those list are used in the bar plotting

    index:      a list consists of feature values of the data (column names actually)

    file_name:    Name by which plot will be saved 
                  a '.pdf' format
    """

    # Markers
    colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))
    marker = itertools.cycle(('o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'None', None, ' ', '')) 
    
    # len of each list in the data
    row_len = len(list(data.values())[0])

    # set width of bar depends on number of groups in the data
    barWidth = 1.0/(len(data.keys())+1)

    # Set position of bar on X axis
    r = []
    r.append(np.arange(row_len))
    for i in range(len(data.keys())-1):
        r.append([x + barWidth for x in r[i]])

    # Make the plot
    fig = plt.figure()#figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i, key in enumerate(data):
        plt.bar(r[i], data[key], width=barWidth, edgecolor='black', label=key)
        #ax.bar(r[i], data[key], width=barWidth, align='edge', edgecolor='white', label=key)
        #plt.bar(r[1], bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')
        #plt.bar(r[2], bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')

    # Add xticks on the middle of the group bars
    #plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(row_len)], index)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    fig.savefig(file_name+'.pdf', bbox_inches='tight')

def area_plot(data,file_name):
    # Markers
    colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))
    marker = itertools.cycle(('o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'None', None, ' ', '')) 
    
    # X-axis
    x = range(1,len(data[list(data)[0]])+1)
    fig = plt.figure()
    print (list(data),x)
    # multiple line plot
    y1 = ''
    y2 = ''
    i = 0 
    for key in data.keys():
        print ('Plotting '+key+'...')
        c = next(colors)
        #plt.plot( x, data[i], marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
        #plt.plot( x, data[key], marker = next(marker), color='black', markersize=12, linewidth=1, label=key)        
        #plt.plot( x, data[i], marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
        #plt.plot( x, data[key], linewidth=1, label=key, color=c)
        plt.fill_between(x, 0, data[key], color=c,label=key, alpha=0.8)
        #if i == 0:
            #y1 = data[key]
            #plt.fill_between(x, 0, y1, color=c,label=key)#, alpha=0.5)
        #else:
            #y2 = data[key]
            #plt.fill_between(x, y1, y2, color=c,label=key)#, alpha=0.5)
            #y1=y2
        #i += 1
        
    #plt.legend()
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    fig.savefig(file_name+'.pdf', bbox_inches='tight')
    #plt.show()


#https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

def stack_plot( data,
                file_name='stack_plot'):
    """
    given a set of list in data, it plot them in a stack plot

    Arguments
    ---------
    data:       dictionary of lists who keys are the label and 
                the values in those list are used in the stack plotting

    file_name:    Name by which plot will be saved 
                  a '.pdf' format
    """
    # Colors to be iterated for plotting each list in data with different color
    #colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # Define X-axis range (1----len(a list in data))
    x = range(1,len(data[list(data)[0]])+1)        
    print (list(data),x)


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.stackplot(x, list(data.values()), labels=list(data.keys()), baseline='sym')#, colors=[:len(data.keys())])
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    fig.savefig(file_name+'.pdf', bbox_inches='tight')



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          file_name='confusion_plot'):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    file_name:    Name by which plot will be saved 
                  a '.pdf' format

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name  # title of graph
                          file_name    = 'confusion_plot')    # name for plot image to be saved

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """    

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    cax = ax.matshow(cm,interpolation='nearest',cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    fig.colorbar(cax)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        #plt.xticks(tick_marks, target_names, rotation=45)
        #plt.yticks(tick_marks, target_names)
        ax.set_xticks(tick_marks)
        plt.xticks(rotation=90)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(target_names)
        ax.set_yticklabels(target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    fig.savefig(file_name+'.pdf', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    data = {}
    data['santosh'] = np.random.randn(10)
    data['rohit'] = np.random.randn(10)+range(1,11)
    data['shri'] = np.random.randn(10)+range(11,21)
    myplot(data,'sample')

    #Example of simple bar_plot()
    data = {}
    data['var1'] = [12, 30, 1, 8, 22]
    data['var2'] = [28, 6, 16, 5, 10]
    data['var3'] = [29, 3, 24, 25, 17]
    index = ['स','स','स','स','स'] #['A', 'B', 'C', 'D', 'E']
    bar_plot(data,index,'bar_plot_demo')
    sys.exit(0)
    # Example of simple myplot()
        # Example of Stack Plot
    data = {}
    data['sleeping'] = [7,8,6,11,7]
    data['eating'] =   [2,3,4,3,2]
    data['working'] =  [7,8,7,2,2]
    data['playing'] =  [8,5,7,8,13]
    area_plot(data,'area_plot')


    # Example of Stack Plot
    data = {}
    data['sleeping'] = [7,8,6,11,7]
    data['eating'] =   [2,3,4,3,2]
    data['working'] =  [7,8,7,2,2]
    data['playing'] =  [8,5,7,8,13]
    stack_plot(data)

    # Example of Confusion Matrix Plot
    plot_confusion_matrix(cm           = np.array([[ 1098,  1934,   807],
                                              [  604,  4392,  6233],
                                              [  162,  2362, 31760]]), 
                      normalize    = False,
                      target_names = ['high', 'medium', 'low'],
                      title        = "Confusion Matrix",
                      file_name    = 'conf_matrix')

    # Example of Normalized Confusion Matrix Plot
    plot_confusion_matrix(cm           = np.array([[ 1098,  1934,   807],
                                              [  604,  4392,  6233],
                                              [  162,  2362, 31760]]), 
                      normalize    = True,
                      target_names = ['high', 'medium', 'low'],
                      title        = "Confusion Matrix, Normalized",
                      file_name    = 'conf_matrix_normalized')

