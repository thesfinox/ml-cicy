import os
import matplotlib.pyplot as plt

def savefig(filename, fig, root='./img', show=False):
    '''
    Save a Matplotlib figure to file.
    
    Needed arguments:
        filename: the path to the saved file in the img directory (no extension),
        fig:      the Matplotlib figure object.
        
    Optional arguments:
        root: root directory,
        show: show the plot inline (bool).
    '''
    
    # save the figure to file (PDF and PNG)
    fig.tight_layout()
    os.makedirs(root, exist_ok=True)
    fig.savefig(os.path.join(root, filename + '.pdf'), dpi=144, format='pdf')
    fig.savefig(os.path.join(root, filename + '.png'), dpi=144, format='png')
    
    # show if interactive
    if show:
        plt.show()
    
    # release memory
    fig.clf()
    plt.close(fig)