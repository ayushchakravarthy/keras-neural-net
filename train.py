# import
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

# define the models
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# fit model onto the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=False)

# make class predictions with the model
predictions = model.predict_classes(X)

# summarizes the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
