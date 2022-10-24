import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt


# MOANA (MOtif ANAlysis)

def activation_pwm(fmap, X, threshold=0.5, window=20):
    # Set the left and right window sizes
    window_left = int(window/2)
    window_right = window - window_left

    N, L, A = X.shape # assume this ordering (i.e., TensorFlow ordering) of channels in X
    num_filters = fmap.shape[-1]

    W = []
    for filter_index in range(num_filters):

        # Find regions above threshold
        coords = np.where(fmap[:,:,filter_index] > np.max(fmap[:,:,filter_index])*threshold)
        x, y = coords

        # Sort score
        index = np.argsort(fmap[x,y,filter_index])[::-1]
        data_index = x[index].astype(int)
        pos_index = y[index].astype(int)

        # Make a sequence alignment centered about each activation (above threshold)
        seq_align = []
        for i in range(len(pos_index)):

            # Determine position of window about each filter activation
            start_window = pos_index[i] - window_left
            end_window = pos_index[i] + window_right

            # Check to make sure positions are valid
            if (start_window > 0) & (end_window < L):
                seq = X[data_index[i], start_window:end_window, :]
                seq_align.append(seq)

        # Calculate position probability matrix
        if len(seq_align) > 0:
            W.append(np.mean(seq_align, axis=0))
        else:
            W.append(np.zeros((window, A)))
    W = np.array(W)

    return W


def clip_filters(W, threshold=0.5, pad=3):
    W_clipped = []
    for w in W:
        L, A = w.shape
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, L)
            W_clipped.append(w[start:end,:])
        else:
            W_clipped.append(w)

    return W_clipped


def generate_meme(W, output_file='meme.txt', prefix='Filter'):
    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C  %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j, pwm in enumerate(W):
        if np.count_nonzero(pwm) > 0:
            L, A = pwm.shape
            f.write('MOTIF %s%d \n' % (prefix, j))
            f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
            for i in range(L):
                f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
            f.write('\n')
    
    f.close()


def plot_filters(W, fig, num_cols=8, alphabet="ACGT", names=None, fontsize=12):
    """plot first-layer convolutional filters from PWM"""
    
    if alphabet == "ATCG":
        W = W[:,:,[0,2,3,1]]
    
    num_filter, filter_len, A = W.shape
    num_rows = np.ceil(num_filter/num_cols).astype(int)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for n, w in enumerate(W):
        ax = fig.add_subplot(num_rows,num_cols,n+1)

        # Calculate sequence logo heights -- information
        I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
        logo = I*w
        
        # Create DataFrame for logomaker
        counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(filter_len)))
        for a in range(A):
            for l in range(filter_len):
                counts_df.iloc[l,a] = logo[l,a]
        
        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.set_ylim(0, 2) # set y-axis of all sequence logos to run from 0 to 2 bits
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])

        if names:
            plt.ylabel(names[n], fontsize=fontsize)
        
    return fig

