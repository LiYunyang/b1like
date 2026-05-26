import numpy as np

def insert_lt_spec(spec_array, lt_array):
    nspec, nbins, nsim = spec_array.shape
    nband = int(np.roots([1,1,-2*nspec])[1])

    nnspec = int(((nband+1)*(nband+2))/2)
    new_spec_array = np.zeros([nnspec, nbins, nsim])
    #the autos
    for i in range(nband):
        new_spec_array[i,:,:] = spec_array[i,:,:]
    new_spec_array[nband,:,:] = lt_array[-1,:,:]

    #the crosses
    ctr = nband
    nctr= nband+1
    lctr= 0
    for i in range(0,nband):
        for j in range(i+1, nband):
            print(i,j, ctr, nctr, lctr)
            new_spec_array[nctr,:,:]  = spec_array[ctr,:,:]
            ctr +=1
            nctr +=1
        print('here:',i,j, ctr, nctr, lctr)
        new_spec_array[nctr,:,:] = lt_array[lctr,:,:]
        nctr +=1
        lctr +=1

    return new_spec_array

def insert_mean_lt(spec_array, lt_array):
    nspec, nbins = spec_array.shape
    nband = int(np.roots([1,1,-2*nspec])[1])

    nnspec = int(((nband+1)*(nband+2))/2)
    new_spec_array = np.zeros([nnspec, nbins])
    #the autos
    for i in range(nband):
        new_spec_array[i,:] = spec_array[i,:]
    new_spec_array[nband,:] = lt_array[-1,:]

    #the crosses
    ctr = nband
    nctr= nband+1
    lctr= 0
    for i in range(0,nband):
        for j in range(i+1, nband):
            new_spec_array[nctr,:]  = spec_array[ctr,:]
            ctr +=1
            nctr +=1
        new_spec_array[nctr,:] = lt_array[lctr,:]
        nctr +=1
        lctr +=1

    return new_spec_array

