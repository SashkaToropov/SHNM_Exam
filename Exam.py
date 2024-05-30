from keras.preprocessing.sequence import pad_sequences
import numpy as np

data = [1, 2, 3, 4, 5]

X = []
for i in range(1, len(data)+1):
    X.append(data[:i])

y = data[:]

max_value = max(data)
X_encoded = [np.eye(max_value+1)[x] for seq in X for x in seq]
y_encoded = np.eye(max_value+1)[y]

max_len = max(len(seq) for seq in X)

X_padded = pad_sequences(X_encoded, maxlen=max_len, padding='post')
y_padded = pad_sequences(y_encoded, maxlen=max_len, padding='post')
