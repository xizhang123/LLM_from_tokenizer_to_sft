import numpy as np

#贪心自动分组函数
def greedy_auto_group(matrix_group_batch):
    index = np.zeros((2,)+matrix_group_batch.shape,dtype="int")
    row,column = 0,1
    for i in range(matrix_group_batch.shape[1]):
        index[row,   :,i] = np.argsort(-matrix_group_batch[:,i])
        index[column,:,i] = i
        index = index[:,np.argsort(matrix_group_batch[index[row,:,:i+1],index[column,:,:i+1]].sum(axis=1))]
    return index[row],index[column]

#递归自动分组函数
def recursive_auto_group(matrix_group_batch):
    group_n = matrix_group_batch.shape[1]
    if group_n % 2 == 0:
        row_index_h,column_index_h = recursive_auto_group(-matrix_group_batch[:,:group_n//2])
        row_index_l,column_index_l = recursive_auto_group(matrix_group_batch[:,group_n//2:])
        row_index = np.concatenate([row_index_h,row_index_l],axis=1)
        column_index = np.concatenate([column_index_h,column_index_l+group_n//2],axis=1)
        index = np.array([row_index,column_index])
        index = index[:,np.argsort(matrix_group_batch[row_index,column_index].sum(axis=1))]
        row_index,column_index = index
    else:
        row_index,column_index = greedy_auto_group(matrix_group_batch)
    return row_index, column_index

