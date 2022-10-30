import torch


class Augment():
    def __call__(self, x):
        raise NotImplementedError()


class RandomDeletion(Augment):
    """augment batch of sequences x by deleteing a stretch of nucleotides
        of length between delete_min and delete_max randomly chosen
        from each sequence at a *different* randomly chosen position (for *each sequence*);
        to preserve sequence length, a stretch of random nucleotides of length equal to 
        the deletion length is split evenly between the beginning and end of the sequence
        
        x: batch of sequences (shape: (N_batch, A, L))
        delete_min (optional): min length of deletion (default: 0)
        delete_max (optional): max length of deletion (default: 30)
    """
    def __init__(self, delete_min=0, delete_max=30):
        self.delete_min = delete_min
        self.delete_max = delete_max

    def __call__(self, x):
        N_batch, A, L = x.shape
        a = torch.eye(A)
        p = torch.tensor([1/A for _ in range(A)])
        insertions = torch.stack([a[p.multinomial(self.delete_max, replacement=True)].transpose(0,1) for _ in range(N_batch)]).to(x.device)
        
        delete_lens = torch.randint(self.delete_min, self.delete_max + 1, (N_batch,))
        delete_inds = torch.randint(L - self.delete_max + 1, (N_batch,)) # deletion must be in boundaries of seq.
        
        x_aug = []
        for seq, insertion, delete_len, delete_ind in zip(x, insertions, delete_lens, delete_inds):
            insert_beginning_len = torch.div(delete_len, 2, rounding_mode='floor').item()
            insert_end_len = delete_len - insert_beginning_len
            
            x_aug.append( torch.cat([insertion[:,:insert_beginning_len],
                                     seq[:,0:delete_ind], 
                                     seq[:,delete_ind+delete_len:],
                                     insertion[:,self.delete_max-insert_end_len:]],
                                    -1) )
        return torch.stack(x_aug)



class RandomTranslocation(Augment):
    """augment batch of sequences x by translocation--that is, shift *each sequence* along 
        the position dim by a different, randomly chosen amount between shift_min 
        and shift_max (see <https://pytorch.org/docs/stable/generated/torch.roll.html>)
        
        x: batch of sequences (shape: (N_batch, A, L))
        shift_min (optional): min number of places by which position can be shifted (default: 0)
        shift_max (optional): max number of places by which position can be shifted (default: 30)
    """
    def __init__(self, shift_min=0, shift_max=30):
        self.shift_min = shift_min
        self.shift_max = shift_max

    def __call__(self, x):
        N_batch = x.shape[0]
        shifts = torch.randint(self.shift_min, self.shift_max + 1, (N_batch,))
        ind_neg = torch.rand(N_batch) < 0.5
        shifts[ind_neg] = -1 * shifts[ind_neg]
        x_rolled = []
        for i, shift in enumerate(shifts):
            x_rolled.append( torch.roll(x[i], shift.item(), -1) )
        x_rolled = torch.stack(x_rolled).to(x.device)
        return x_rolled



class RandomInsertion(Augment):
    """augment batch of sequences x by inserting a stretch of random nucleotides
        into each sequence at a *different* randomly chosen position (for *each sequence*);
        remaining nucleotides up to insert_max are split between beginning and end
        of sequences
        
        x: batch of sequences (shape: (N_batch, A, L))
        insert_min (optional): min length of insertion (default: 0)
        insert_max (optional): max length of insertion (default: 30)
    """
    def __init__(self, insert_min=0, insert_max=30):
        self.insert_min = insert_min
        self.insert_max = insert_max

    def __call__(self, x):
        N_batch, A, L = x.shape
        a = torch.eye(A)
        p = torch.tensor([1/A for _ in range(A)])
        insertions = torch.stack([a[p.multinomial(self.insert_max, replacement=True)].transpose(0,1) for _ in range(N_batch)]).to(x.device)

        insert_lens = torch.randint(self.insert_min, self.insert_max + 1, (N_batch,))
        insert_inds = torch.randint(L, (N_batch,))

        x_aug = []
        for seq, insertion, insert_len, insert_ind in zip(x, insertions, insert_lens, insert_inds):
            insert_beginning_len = torch.div((self.insert_max - insert_len), 2, rounding_mode='floor').item()
            insert_end_len = self.insert_max - insert_len - insert_beginning_len
            x_aug.append( torch.cat([insertion[:,:insert_beginning_len],
                                     seq[:,:insert_ind], 
                                     insertion[:,insert_beginning_len:insert_beginning_len+insert_len], 
                                     seq[:,insert_ind:],
                                     insertion[:,insert_beginning_len+insert_len:self.insert_max]],
                                    -1) )
        return torch.stack(x_aug)



class RandomInversion(Augment):
    """augment batch of sequences x by inverting a randomly chosen stretch
        of nucleotides to its reverse complement in each sequence (with each stretch 
        being independently and randomly chosen for *each sequence*)
        
        x: batch of sequences (shape: (N_batch, A, L))
        invert_min (optional): min length of reverse complement inversion (default: 0)
        invert_max (optional): max length of reverse complement inversion (default: 30)
    """
    def __init__(self, invert_min=0, invert_max=30):
        self.invert_min = invert_min
        self.invert_max = invert_max

    def __call__(self, x):
        N_batch, A, L = x.shape
        inversion_lens = torch.randint(self.invert_min, self.invert_max + 1, (N_batch,))
        inversion_inds = torch.randint(L - self.invert_max + 1, (N_batch,)) # inversion must be in boundaries of seq.
            
        x_aug = []
        for seq, inversion_len, inversion_ind in zip(x, inversion_lens, inversion_inds):
            x_aug.append( torch.cat([seq[:,:inversion_ind], 
                                     torch.flip(seq[:,inversion_ind:inversion_ind+inversion_len], dims=[0,1]), 
                                     seq[:,inversion_ind+inversion_len:]],
                                    -1) )
        return torch.stack(x_aug)

        

class RandomMutation(Augment):
    """augment batch of sequences x by randomly mutating a fraction mutate_frac
        of each sequence's nucleotides, randomly chosen independently for each 
        sequence in the batch
        
        x: batch of sequences (shape: (N_batch, A, L))
        mutate_frac (optional): fraction of each sequence's nucleotides to mutate 
            (default: 0.1)
    """
    def __init__(self, mutate_frac=0.1):
        self.mutate_frac = mutate_frac

    def __call__(self, x):
        N_batch, A, L = x.shape
        num_mutations = round(self.mutate_frac / 0.75 * L) # num. mutations per sequence (accounting for silent mutations)
        mutation_inds = torch.argsort(torch.rand(N_batch,L))[:, :num_mutations] # see <https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146>0

        a = torch.eye(A)
        p = torch.tensor([1/A for _ in range(A)])
        mutations = torch.stack([a[p.multinomial(num_mutations, replacement=True)].transpose(0,1) for _ in range(N_batch)]).to(x.device)
        
        x_aug = torch.clone(x)
        for i in range(N_batch):
            x_aug[i,:,mutation_inds[i]] = mutations[i]
            
        return x_aug



class RandomRC(Augment):
    """augment batch of sequences x by returning the batch with each sequence 
        either kept as the original sequence or \"mutated\" to its reverse complement
        with probability rc_prob 
        
        x: batch of sequences (shape: (N_batch, A, L))
        rc_prob (optional): probability of each sequence to be \"mutated\" to its 
            reverse complement (default: 0.5)
    """
    def __init__(self, rc_prob=0.5):
        self.rc_prob = rc_prob

    def __call__(self, x):
        x_aug = torch.clone(x)
        ind_rc = torch.rand(x_aug.shape[0]) < self.rc_prob
        x_aug[ind_rc] = torch.flip(x_aug[ind_rc], dims=[1,2])
        return x_aug



class RandomNoise(Augment):
    """augment batch of sequences x by returning a noise-added version of x
        with Gaussian noise added to every one-hot coefficient in x
        
        x: batch of sequences (shape: (N_batch, A, L))
        noise_mean (optional): mean of Guassian noise added (default: 0.0)
        noise_std (optional): standard deviation of Gaussian noise added (default: 0.1)
    """
    def __init__(self, noise_mean=0.0, noise_std=0.2):
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __call__(self, x):
        return x + torch.normal(self.noise_mean, self.noise_std, x.shape).to(x.device)












