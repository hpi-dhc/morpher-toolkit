



'''

TRAINING THE ALGORITHM

'''

#data: pandas frame 

data = load("development.csv") # wie auch immer geartete Paramater
# things happening
## -> load data in table (raw)


#data: pandas frame
data = preprocess(data)
## load data in table (tranformed)
## -> create metadata table (description of each field)


#train, test
train,test = split(data)




#trained algorithm
trained_algorithm = train(alg, train)

metrics = evaluate(trained_algorithm, test)



'''
EVALUATING THE ALGORITHM ON A NEW DATASET

'''

#data: pandas frame 

data = load("validation.csv") # wie auch immer geartete Paramater

#data: pandas frame
data = preprocess(data)

metrics = evaluate(trained_algorithm, data)


'''
The question then becomes: how can I make sure that the data on the other side is equivalent to what I have at hand?!

'''


### How would I use this to run create a model from a Python perspective?

