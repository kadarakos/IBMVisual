from grzegorz_ibm import M1

e = [['a', 'b', 'c'], ['c', 'e', 'd']] # English sentences list of lists of tokens
f = [[1, 2, 3], [4,5,6]] # French sentences, list of lists of tokens
model = M1(e, f, online=False, weighted=False) # Initializing the model
model.iterate() # Running the EM algorithm
ttable =  model.translation_table() # Estimating translation probabilities
