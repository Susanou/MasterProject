import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets, layers, models

import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


def create_model():

    
    # CIFAR10 model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              #run_eagerly=True #debug only
              )
    """


    #MINST model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    """

    return model

def create_culled_model(num_classes):
    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes)
        ]
    )
    model.compile(  optimizer='adam',
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model

def check_vote(g_predictions, voted, labels):
    misscount = 0

    for i, l in enumerate(labels):
        if voted[i] != l:
            for pred in g_predictions:
                if np.argmax(pred[i]) == l:
                    misscount += 1 
                    break

    print(f"There  was a total of {misscount} misscounts during vote where one of the learners at least classified it correctly")

def check_learner_acc(learner, images, labels):
    tp = 0

    for i, l in enumerate(learner.predict(images)):
        if labels[i] == np.argmax(l):
            tp += 1
    
    print(f"tp = {tp} => acc = tp/len = {float(tp/len(labels))}")
    print("Results from built-in tf function ", learner.evaluate(images, labels, batch_size=128, verbose=0))

def is_multualy_exclusive(list1, list2):
    mask = np.isin(list1, list2)
    #print(any(list(mask)))
    return not any(list(mask))

def vote(voters, images, labels):

    # Global Predications of all learners

    f = open(filename, "a")

    global_predictions = []
    for i, v in enumerate(voters):
        global_predictions.append(v.predict(images))
        #check_learner_acc(v, images, labels)
        results = v.evaluate(images, labels, batch_size=128, verbose=0)
        print(f"results of voter {i} acc test: loss={results[0]} acc={results[1]}")
        #print(len(images), len(labels))
        print(f"{e},{i},{results[0]},{results[1]}", file = f)

    global_predictions = np.array(global_predictions)
    f.close()

    #print(global_predictions)

    # Voting part loop
    certain_global = []
    count = 0
    if not culling:
        for i in range(len(labels)): 
            tmp = np.zeros(10)

            for cg in global_predictions:
                best = np.argmax(cg[i])
                tmp[best] += cg[i][best]
                #tmp = np.maximum.reduce(global_predictions[:, i])

            certain_global.append([np.argmax(tmp)])    # original
            #certain_global.append(np.argmax(tmp)) # mnist fix

            if np.argmax(tmp) == labels[i]:
                count += 1
        
        check_vote(global_predictions, certain_global, labels)
    else:
        
        a_class = [0,4,6,8,9]
        b_class = [1,2,3,5,7]
        classes = [a_class, b_class]

        #print(global_predictions[0][0], global_predictions[1][0], labels[0])
        
        for i in range(len(labels)):
            tmp = np.zeros(10)
            for j in range(len(voters)):
                if np.argmax(global_predictions[j][i]) != 5:
                    tmp[classes[j][np.argmax(global_predictions[j][i])]] = global_predictions[j][i][np.argmax(global_predictions[j][i])]
            
            certain_global.append(np.argmax(tmp))
            if np.argmax(tmp) == labels[i]:
                count += 1
        check_vote(global_predictions, certain_global, labels)
    return certain_global, count

def new_vote(voters, images, labels, test_image, test_labels):

    #print(images[0])

    f = open(filename, "a")

    global_predictions = []
    for i, v in enumerate(voters):
        global_predictions.append(v.predict(images))
        #check_learner_acc(v, images, labels)
        results = v.evaluate(test_image, test_labels, batch_size=128, verbose=0)
        print(f"results of voter {i} acc test: loss={results[0]} acc={results[1]}")
        #print(len(images), len(labels))
        print(f"{e},{i},{results[0]},{results[1]}", file = f)

    global_predictions = np.array(global_predictions)
    f.close()

    #print(global_predictions)

    # Voting part loop
    image_voted     =       []
    label_voted     =       []
    image_not       =       []
    label_not       =       []

    certain_global = []
    count = 0

    for i in range(len(labels)): 
        tmp = np.zeros(10)

        for cg in global_predictions:
            best = np.argmax(cg[i])
            tmp[best] += 1 #select only the best and check that its equal to 5 ie unanimous vote        

        
        
        if tmp[np.argmax(tmp)] == len(voters):
            image_voted.append(images[i])
            label_voted.append(labels[i])

            


            if np.argmax(tmp) == labels[i]:
                count += 1
        else:
            image_not.append(images[i])
            label_not.append(labels[i])
    
#    check_vote(global_predictions, certain_global, label_voted)

    return [image_voted, label_voted], [image_not, label_not], count

def train_local(train_x, train_y, learners, i):

    try:
        model = learners[i]

        model.fit(train_x, train_y, epochs=10, shuffle=True, verbose=0)

        # Maybe better way but needed to save into a file at one point
        model.save(f'models/new_method/model_{i}.tf')
    except Exception as e:
        if culling:
            model = create_culled_model(6) # we are training them on 5 classes currently. make it modular later
        else:
            model = create_model()
    
        model.fit(train_x, train_y, epochs=10, shuffle=True, verbose=0)

        # Maybe better way but needed to save into a file at one point
        model.save(f'models/new_method/model_{i}.tf')

def dataset_formatting(train_x, train_y, global_size, percent, n_learners=2):
    local_train_x = train_x[global_size:]
    local_train_y = train_y[global_size:]
    global_train_x = train_x[:global_size]
    global_train_y = train_y[:global_size]

    theta = 4
    if n_learners == 2:
        local_ds = (len(local_train_x)*percent)//100
    else:
        local_ds = len(local_train_x)//n_learners
    print("Length of the local dataset", local_ds)

    trainsets = [[local_train_x[i*local_ds:(i+1)*local_ds], local_train_y[i*local_ds:(i+1)*local_ds]] for i in range(n_learners)]

    return trainsets, global_train_x, global_train_y, local_ds

def dataset_formatting_label_culling(train_x, train_y, global_size, fixed, FP):

    global culling
    culling = True

    if fixed:
        a_class = [0,4,6,8,9]
        b_class = [1,2,3,5,7]
    else:
        a_class = np.random.randint(10, size=5)
        b_class = np.random.randint(10, size=5)
    
    local_train_x = train_x[global_size:]
    local_train_y = train_y[global_size:]
    global_train_x = train_x[:global_size]
    global_train_y = train_y[:global_size]

    train_a_x = []
    train_b_x = []
    train_a_y = []
    train_b_y = []

    for i, e in enumerate(local_train_y):
        if e in a_class:
            train_a_x.append(local_train_x[i])
            train_a_y.append(a_class.index(local_train_y[i]))
        elif random.random() < FP:
            train_a_x.append(local_train_x[i])
            train_a_y.append(a_class.index(local_train_y[i]))
        
        if e in b_class:
            train_b_x.append(local_train_x[i])
            train_b_y.append(b_class.index(local_train_y[i]))
        elif random.random() < FP:
            train_b_x.append(local_train_x[i])
            train_b_y.append(b_class.index(local_train_y[i]))

    trainsets = [[np.array(train_a_x), np.array(train_a_y)], [np.array(train_b_x), np.array(train_b_y)]]

    return trainsets, global_train_x, global_train_y

def test_acc(learners, target):
    certain_global, count = vote(learners, x_test, y_test)
    print("OVER TEST DATA = Certain predictions amount", len(certain_global), "with correct in them", count)

    target_predict = target.predict(x_test)
    count = 0
    for i in range(len(y_test)):

        if np.argmax(target_predict[i]) == y_test[i]:
            count += 1

    print("OVER TEST DATA = TARGET Certain predictions amount", len(target_predict), "with correct in them", count)



##############
# Global Vars
##############

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Use GPUs
tf.config.set_soft_device_placement(True)
#tf.config.run_functions_eagerly(True) # debug for model not compiling
#tf.debugging.set_log_device_placement(True) #uncomment if need to check that it is executing off of GPU
tf.get_logger().setLevel('ERROR')

filename = "outputs/plotdata_new_algo200_per_part.csv"

f = open(filename, "a")
f.write("Epoch,Learner,Loss,Accuracy\n")
f.close()

(x_train, y_train), (x_test, y_test)= keras.datasets.cifar10.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

local_size = 40000

assert local_size < len(x_train)

trainsets, global_x, global_y, local_ds  = dataset_formatting(x_train, y_train, local_size, 10, 5)
#trainsets, global_x, global_y = dataset_formatting_label_culling(x_train, y_train, 20000, True, 0.0)

# Set number of itterations either via local_ds or number of epochs to train
epochs = 30
#epochs = len(global_x) // (local_ds) + 1
local_ds = len(global_x) // epochs
repetition = 1


culling = False

learners = []

e = 0 #variable for the epoch number

# Training on local dataset
for i in range(len(trainsets)):
    print(f"Building model {i}")

    #print(trainsets[i][0])

    train_local(trainsets[i][0], trainsets[i][1], learners, i)
    learners.append(load_model(f'models/new_method/model_{i}.tf'))

print(len(trainsets[0][0]), len(trainsets[0][1]))

print("learners: ", len(learners))

# Target training 

model = create_model()

print("Training target")
model.fit(x_train, y_train, epochs=10, shuffle=True, verbose=0)
model.save(f'models/new_method/target.tf')
del model
target = load_model('models/new_method/target.tf')


#check_learner_acc(target, x_test, y_test)
results = target.evaluate(x_test, y_test, batch_size=128, verbose=0)
print(f"results of target acc test: loss={results[0]} acc={results[1]}")

## Train target before the rest so we can compare acc
target_predict = target.predict(global_x)
count = 0
for i in range(len(global_y)):

    if np.argmax(target_predict[i]) == global_y[i]:
        count += 1

print("TARGET Certain predictions amount", len(target_predict), "with correct in them", count)

# first acc test before anything
test_acc(learners, target)

print(f"Data per epoch in itteration {local_ds}")
print(f"Number of epochs {epochs}")
e = 0

# Training loop iterations
while len(global_x) != 0 and e<epochs:
    print(f"Training epoch {e}")

    e += 1

    """
    start_x = (i*local_ds) % len(global_x)
    end_x = ((i+1)*local_ds) % len(global_x) if (((i+1)*local_ds) % len(global_x)) != 0 else len(global_x)
    start_y = (i*local_ds) % len(global_y)
    end_y = ((i+1)*local_ds) % len(global_y) if (((i+1)*local_ds) % len(global_y)) != 0 else len(global_y)
    
    #print(global_x[i*local_ds:(i+1)*local_ds][0])
    
    # OLD VOTING
    certain_global, count = vote(learners,
                               global_x[start_x:end_x], 
                               global_y[start_y:end_y])
    certain_global = np.array(certain_global)
    print("Certain predictions amount", len(certain_global), "with correct in them", count)
    """
    voted, remaining, count = new_vote(learners,
                                global_x, 
                                global_y,
                                x_test, y_test)
    
    global_x = np.array(remaining[0])
    global_y = np.array(remaining[1])

    # fit model to the new labels
    # Training loop
    for j in range(len(learners)):
        print(f"Training learner {j}")

        tmp_img = trainsets[j][0]
        tmp_labels = trainsets[j][1]

        #print(tmp_labels)
        #print(certain_global)

        trainsets[j][0] = np.append(tmp_img, 
                                    voted[0],
                                    axis=0)
        trainsets[j][1] = np.append(tmp_labels, voted[1], axis=0)
        
        #assert len(trainsets[j][0]) == len(trainsets[j][1])

        train_local(trainsets[j][0], trainsets[j][1], [], j) #old code
        #train_local(voted[0], voted[1], learners, j)
        learners[j] = load_model(f'models/new_method/model_{j}.tf')

    #test_acc(learners, target)

e += 1

# Last round of perdictions to check accuracy changes over a test dataset
certain_global, count = new_vote(learners, global_x, global_y, x_test, y_test)

certain_global = np.array(certain_global)
print("Certain predictions amount after new training ", len(certain_global), "with correct in them", count)
e += 1
test_acc(learners, target)