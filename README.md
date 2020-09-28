Lei4Cov：Literature evidence information for cov

METHOD: Literature Evidence Information for COVID-19 (Lei4Cov) is a NLP vector embedding tool, which is developed for training entity representations of biomedical concepts from literature based on transformer encoder. The transformer encoder is based on multi-head self-attention algorithm which has  shown its power in capturing dependency relations among entities from the same literature.  

INPUT: Because entities within the same literature are related to each other but are not related across different literature, entity sequences are aligned literature by literature. Since some literature contains fewer entities, paddings are added to make all entity sequence with the same length. The input is an N by M data matrix, where N is the maximum length of entity sequence and M is the total number of literature. The initial N by M data matrix is made up some random numbers and then is fed as the input to the Lei4Cov.  The adadelta optimizer is used to minimize softmax loss function until convergence.

OUTPUT: optimized vector embedding N by M data matrix


# data_process
  - data processing
# lei4cov
  - 文献向量训练

