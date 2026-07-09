#----------------
#IMPORTED MODULES
#----------------

#first, I began my code by importing some standard python modules (I made sure they were not external libraries), which would aid me throughout this project.
import csv #this will help me read my csv file which contains the dataset collected by playing the lander game
import math #for the math functions used throughout the code
import random #I used this to create random number and used it for weight initialization

#-----------------
#GLOBAL CONSTANTS:
#-----------------


CSV_PATH = "ce889_dataCollection.csv" #this is the path address that contains the csv dataset, it has been made sure to put the csv file and this py file in the same folder

EPOCHS = 100 #number of epochs shows how many times the code goes over the dataset 

LEARNING_RATE = 0.03 #basically shows how big of a step my code will take when the weights are updated.
#slower learning rate usually means smaller values but it also means more stability and on the other end faster learning rate can mean larger values but can also cause overshoot and unstability
#this learning rate was achieved by first reading parts of "deep learning" by Goodfellow, Bengio, Courville and then doing a research on LeCun's work then seeing the best result on matlab (more explanation on the slides)

HIDDEN_SIZE = 6 #number of neurons in the hidden layer, while more neurons means more learning capacity it also means more potential for overfitting.
# this number was chosen based on three research sources, first one being "deep learning" by Goodfellow, Bengio, Courville, second one being heaton's guid to choosing number of hidden nuerons (2008)
# and the third one being a thesis from DEPARTMENT OF THE AIR FORCE AIR UNIVERSITY  

SEED = 42 #setting a seed makes random numbers repeatable each time the code is run, which is useful for getting the same results each time and understanding the problems, fixing them and improving them
#any integer number could be used here, number 42 was used here due to it being a common number in the examples for seed (thanks Douglas Adams!)

WEIGHTS_FILE = "lander_neuron_weights.txt" #this is where the trained weights will be saved

SCREEN_WIDTH = 800 #this is the width od our game
SCREEN_HEIGHT = 600 #this is the height of our game

#in here I used Pythagorean theorem which indicates a^2 + b^2 = c^2, I used the math function I imported earlier and called the variable DIAG.
#I used this as a normalization scale using height and width, in order to keep all the values in a similar range
#the reason I used Diagnal is because the inputs are x_target and y_target, which could mean any distance, so first I specified what my max_x and max_y could be,
# then in order to not have x and y be normalized on different scales and hurt the "understanding" of my netword, I used a factor that implements both x and y and help the NN learn better.
DIAG = math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)  #the answer to this equation is almost equal to 1000

#this is where we use the seed to generate repeatable numbers on each code run
random.seed(SEED)


# ------------------------------
# SIGMOID ACTIVATION
# ------------------------------

#the sigmoind function fits any number in the range(0,1)
#it is used here as an activation number (it is usually used in neural networks as an activation function)
def sigmoid(x):
    #in here we clip the input to put in a safe range, is x ends up too negative, math.exp(-x) will be too big and too small if x is a large positive number
    if x > 40: return 1.0 #this will basically return 1 as for values above roughly +40 the sigmoid is indistinguishable from 1
    if x < -40: return 0.0 #this will basically return 0 as for values bellow roughly -40 the sigmoid is indistinguishable from 0
    return 1.0 / (1.0 + math.exp(-x)) # Standard sigmoid formula: 1 / (1 + e^(-x))

#if a = sigmoid(x), then sigmoid'(x) = a * (1 - a), there for the derivation is in respect to it's output a.
# passing a direcyly helps with avoiding recomputeing sigmoind(x)
def sigmoid_derivative(a):
    return a * (1.0 - a)


# ------------------------------
# NEURON
# ------------------------------

#this class will represent a neuron and then this class will be called in the mlp class, by this I will try to avoid repetetive codes and defining neurons over and over again.
#a neuron has 1)weights 2)bias 3)activation function (sigmoind or linear) 

class Neuron:
    def __init__(self, n_inputs, activation="sigmoid"): #in this class the sigmoind activation function has been used, n_inputs are for random weights (between -1 and 1)
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]  #random weights brak symmetry and repetetivness, if all the weights were the same, all the neurons would have learnt the same thing
        self.bias = random.uniform(-1, 1) #bias is also initialized randomley between -1 and 1 and it's job is to allow neuron to shift it's activation treshhold left and right
        self.activation = activation #the activation function will be stored later in the forward() part


    def forward(self, inputs): #this part calculates the neuron's output for a lists of inputs
        #in lecture 3, slide 4, we had the most popular mathematical model for neuron, this where this equation comes from:
        #zip pairs each input with it's matching weight then the bias is added
        z = sum(i*w for i, w in zip(inputs, self.weights)) + self.bias

        #now the chosen activation function, which is sigmoind, is added.
        #the reason why sigmoind is chosen is because our problem is not linear, even in a non-mathematical point of view: if it get's too close too close to the rocks, move gently or don't go down too fast
        #these behaviors are not linear, one can not put them on a linear function therefore I used sigmoind function so my network can learn curves and non linear behaviors
        if self.activation == "sigmoid":
            #if activation is indeed sigmoind, then it passes z, which we defined earlier, through it
            return sigmoid(z)
        else:
            return z  #otherwise the code returs Z which will result in a linear outcome
        

    def derivative(self, activated_output):  #this will give the derivative of the activation function which is needed during backpropagation to compute gradients
        if self.activation == "sigmoid":
            return sigmoid_derivative(activated_output) #if the neuron uses sigmoind then the code will pass the already activated ouput which is sigmoind(z)
        else:
            return 1.0  #otherwise it will simply return 1, because f(x) = x and it's derivative is 1


# ------------------------------
# MLP
# ------------------------------
class MLP:
    #this class represents the neuron network, as seen bellow, I will call the class neuron in this class and use it in here.
    #this class has one hidden layer, housing many neurons (in this network the hidden layer has 8 neurons)
    #this class also has an output layer

    def __init__(self, n_inputs, n_hidden, n_outputs, lr=0.03):
        #this creates the hidden layer
        #n_hidden neurons receive n_inputs inputs
        #it is worth pointing out that they all use sigmoind activation function to squish all the numbers between 0 and 1
        self.hidden = [Neuron(n_inputs, activation="sigmoid") for _ in range(n_hidden)]

        #this here creates the outout layer
        #n_ouputs neurons, receive n_hidden inputs
        #in this part linear activation function has been used so the final numbers are not scaled between 0 and 1
        self.output = [Neuron(n_hidden, activation="linear") for _ in range(n_outputs)]
        #an example of this can be seen in Example 2, in which the output of the hidden layer was sigmoind while the output of the outer layer of linear


        #this stores the learning rate for weight updates
        self.lr = lr

    def forward(self, inputs):
        #this function does a forward pass through the whole network: input -> hidden layer -> output layer 

        #each hidden neuron's output should be calculated, and for this the full input vector should be passed to each hidden neuron
        h_out = [h.forward(inputs) for h in self.hidden]

        #then the output of each hidden neuron is used as inputs for each output neuron
        out = [o.forward(h_out) for o in self.output]
        #the two equations for hidden layer outputs and final outs can be found in lecture 3, slide 6 and lecture 4,5, slide 7

        #both of the final outputs and the hidden layer's outputs are returned, as we need the hidden layer's outputs during the training
        return h_out, out

    def train(self, X, Y, epochs=100):
        #I'll call X the list of input vectors (which has to be given to neurons in hidden layer)
        #and I'll call the target output vectors Y
        #as said earlier, epochs are the number of times the code will go through the whole dataset


        for epoch in range(epochs):
            #mse starts at 0 for each epoch (mean squared error) but the sum of errors will be accumulated and divided later
            mse = 0.0

            #what zip does is pair X and Y together for each row
            for inputs, targets in zip(X, Y):
                #forward pass will compute hidden layer outputs and final outputs
                hidden, outputs = self.forward(inputs)

                #--------------------------------
                #BACKPROPAGATION FOR OUTPUT LAYER
                #--------------------------------

                #in lecture 4,5, slide 11, we were thought about error function and delta , which turned to python code looks like the following:
                #delta is error multiplyed by the deriviative of activation output, delta has been calculated for the output layer
                output_deltas = []
                for i, neuron in enumerate(self.output):
                    #error = target - prediction (how different was our target from our prediction)
                    error = targets[i] - outputs[i]
                    #in truth we know that derivative for a linear function would be 1, but for keeping it general, it is seen as common practice to keep the derivative()
                    delta = error * neuron.derivative(outputs[i])
                    #uses append which is commonly used to store new items, here it is used to store delta for this neuron
                    output_deltas.append(delta)
                    #+= adds squared error to the mse
                    mse += error**2

                #--------------------------------
                #BACKPROPAGATION FOR HIDDEN LAYER
                #--------------------------------

                # in here we use full back propagation equation which can be found in lecture 4,5 slide 13. 
                #we need to keep in mind that each hidden neuron direcyly effects all output neuron, and the equation for it's error reflects that
                #the error is the sum of all the output deltas multiplyed by weights from hidden -> output
                hidden_deltas = []
                for j, neuron in enumerate(self.hidden):
                    error = sum(output_deltas[k] * self.output[k].weights[j]
                                for k in range(len(self.output)))
                    
                    #hidden[j] is the activated output from hidden neuron j and by passing that to neuron.derivative, one can achieve the slope at that point
                    delta = error * neuron.derivative(hidden[j])
                    #again append has been used to store the delta
                    hidden_deltas.append(delta)

                #-------------------------------
                #UPDATE WEIGHTS FOR OUTPUT LAYER
                #-------------------------------
                
                #the equations used here are taken from lecture 4,5, slide 12
                #this is the part where weights and bias gets adjusted
                #gradient descent is used here and it is as follows = old_weight + lr * delta * input_value(the hidden neuron's output)
                for i, neuron in enumerate(self.output):
                    for j in range(len(neuron.weights)):
                        neuron.weights[j] += self.lr * output_deltas[i] * hidden[j] #in here neuron.weight[j] corresponds to connection from hidden[j] to this output neuron
                    neuron.bias += self.lr * output_deltas[i] #the bias get updated as well using delta

                #-------------------------------
                #UPDATE WEIGHTS FOR HIDDEN LAYER
                #-------------------------------

                #this is extremely similar to the last part but the inputs that are being talked here are actually the inputs that have been there since the very beginning

                for j, neuron in enumerate(self.hidden):
                    for k in range(len(neuron.weights)):
                        neuron.weights[k] += self.lr * hidden_deltas[j] * inputs[k] #neuron.weight[k] corresponds to connection from neuron[k] to this hidden neuron
                    neuron.bias += self.lr * hidden_deltas[j] #the bias get updated here as well 

            #now that all the samples for this epoch has been covered, I've gone for the average mse
            #in order to avoid too much text after running the code, I have decided to print every 10 epoch
            #so the code below says only print the number of epochs devidable by 10 or the final epoch
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d} | MSE = {mse/len(X):.6f}")

    def predict(self, inputs):
        #this is what essentially is a helper function
        #this was one of the many aspects in this code for me as the part were mostly covered in CE705
        #this function is here to run only a forward pass and get the ouputs and in here only the final outputs are returned
        _, outputs = self.forward(inputs)
        return outputs

    def save(self, path):
        #what this does is save all the trained weights and biases into a text file, weights and bias of each neuron are written and a # is added between layers
        with open(path, "w") as f:
            for layer in [self.hidden, self.output]:
                for neuron in layer:
                    f.write(",".join(map(str, neuron.weights)) + "\n") #writes weights separated by comma
                    f.write(str(neuron.bias) + "\n") #goes to the next line and writes the bias
                f.write("#\n") # by using #, separates layers from eachother 
        print(f"Weights saved to {path}")


# ----------------------------
# DATA LOADING AND CONTROLLERS
# ----------------------------
def read_csv(path):
    """
    let's explain what this function does and why it is truly here:
    this function reads the csv file and creates two very important thingswhich has been explained before:
    X : a list of input vectors 
    Y : a list of target ouput vectors

    the csv rows that we have are:
    x_target
    y_target
    new_vel_y
    new_vel_x

    but what I have done here is ignore the recorded velocities from the file, because even though I played the game many times, for many hours,
    and tried to work with the original collected dataset, whether it was for my lack of coding skills or lack of gaming skills (assuming both),
    the data collected by me, like any human collected data was extremely noisy and imperfect

    
    trying a mathematical approach seemed like the next best thing to do, therefore here I've computed my own target velocities (vx, vy) 
    by using a mathematical controller

    the idea of using mathematical control came after doing some research on projects similar to our lander projects and reading papers based on
    these projects, eventually I chose one paper and one opensource code to base this code on (further explanation can be found on the slides)

    """
    
    
    X, Y = [], [] #X will store inpute pairs while Y will store out pairs (normalised velocities)
    with open(path, newline="") as f: #newline="" is here to avoid issues with extra blank lines
        r = csv.reader(f)

        #loops over each and every row in the CSV file
        for row in r:
            if len(row) < 2: #skips rows that are too short (missing data)
                continue

            # ----------------
            #EXTRACT RAW INPUT 
            # ----------------

            #first we convert everything to float
            #x_target and y_target are positions related to the landing point that our rocket needs to land
            x_target = float(row[0])
            y_target = float(row[1])

            # ----------------
            # NORMALIZE INPUTS 
            # ----------------

            #as explained before, we want x_target and y_target to be in the range of 0 to 1
            # therefore we normalize them using DIAG which I explained eralier in the code
            #we use min max normalization which is [-diag/2, diag/2], where -diag/2 roughly gives us a 0 and diag/2 roughly gives us a 1

            x_in = (x_target + DIAG/2.0) / DIAG
            y_in = (y_target + DIAG/2.0) / DIAG

            # ============================================================
            # MATHEMATICAL CONTROLLER:
            # ============================================================

            # ==========================
            # 1. HORIZONTAL CONTROL
            # ==========================
            #a maximum horizontal speed is chosen based on how far the rocket is horizontally
            #this is where the width and height that we defined earlier becomes extremely handy
            #if the rocket is too far, meaning x_target > 300 , a high speed is allowed in order to reach the landing point quicker
            #if the rocket is closer, then the speed needs to be reduced in order to encourage gentle behavior around the rocks (danger points)
            if abs(x_target) > 300:
                Vx_max = 6.0     # far from the landing point -> move fast
            elif abs(x_target) > 200:
                Vx_max = 5.0     #medium distance, not too far not too close -> medium speed 
            else:
                Vx_max = 3.5     # all other possibilities are considered to be close -> speed needs to be reduced for more gentleness

            # what kx does is controls how strongly the controller reacts to horizontal error
            #the reason 400 is used is because this code is saying when the rocket is about 400 pixels away, the controller should output vx_max as the desired speed
            #in other words 400 is a scaling factor, as 400 is seen as a large distance in the game
            kx = Vx_max / 400.0   

            # this here, again reinforces my want that the farther the rocket is from the landing point, the faster I want it to move
            #it is called proportional error because the target acts proportionally to the error
            vx = kx * x_target

            # ---------------------------------------------
            #during the trial and error runs of the game, one of problems I faces was that I had problems with landing my rocket when the landing point was placed in the far right corner
            #so in the following code I tried to boost the rightward target velocity as best as I could, to ensure that my rocket lands.
            #boosting the right side basically makes the controller tell the network to push harder to the right side
            # ---------------------------------------------
            
            if vx > 0:  #only boost if vx is pointing to the right side
            
                if abs(x_target) > 300:
                    vx *= 2.0     # if it is far away -> very strong boost
                elif abs(x_target) > 200:
                    vx *= 1.7     # mid distance
                else:
                    vx *= 1.4     # near the landing point

            # here I used clamping, which is forcing a value to stay within a min and max limit, the reason I used clamping is to prevent my network to output
            #unstable or unfeasible values, and also to not exceed the speed limits I set up earlier
            if vx >  Vx_max: vx =  Vx_max
            if vx < -Vx_max: vx = -Vx_max

          # ======================================
            # 2.VERTICAL CONTROL
            # ======================================

            #an extremely important part of what we need to do now is to keep the rocket safely above the ground
            #BASE_ALT is a constant that describes the minimum safe altitude
            BASE_ALT = 120  #20% of screen height, which gives the rocket enough room to avoid the obstacles but is not too large that might cause problems
                            # always stay at least 120 pixels above graound
            
            ALT_GAIN = 0.35 #number 35 gives us room to work with, as it is not too big to be uncontrolable, and not too small to be inefficient             
                            # climb more when the rocket is horizontally far away and will speed up as per previos instructions

            #if the rocket is far horizontally, it has to fly higher, to avoid hitting the taller rocks/spikes
            #for example if the landing point is located at |-400| then : desired_altitude = 120 + 0.35*400 = 260
            desired_altitude = BASE_ALT + ALT_GAIN * abs(x_target)

            # if pad is below current position, y_target is positive.
            # if y_target < desired_altitude, then the rocket is TOO LOW → climb or keep position!
            if y_target < desired_altitude:
                # the rocket is too low, so it needs to avoid going down.
                # I set vy to 0.0 meaning almost no downward speed
                vy = 0.0 
            else:
                #in this situation it is safe to descend, but very gently
                #these are the min and max allowed speed 
                Vy_min = 0.3
                Vy_max = 1.0

                #here I used capping again, to limit how large y_target can affect the speed
                y_eff = min(y_target, 300.0)
                vy = Vy_min + (Vy_max - Vy_min) * (y_eff / 300.0) ## if y_eff is 0, vy ≈ 0.3, if y_eff is 300, vy ≈ 1.0

            # Clamping happens here again
            if vy < -0.5: vy = -0.5        # limit upward speed
            if vy >  1.0: vy =  1.0        # limit descent speed



            # ============================================================
            #NORMALISE TARGET VELOCITIES FOR SIGMOID
            # ============================================================
            #it is important to know that our network's outputs are going to be interupted in a range which will correspond to velocities 
            #from from -max to +max speed, so it is important to normalise vx and vy into [0,1] so they match the sigmoid style scale

            max_speed = 8.0   # used in NeuralNetHolder for de-normalisation

            # Normalisation formula:
            # value in [-max_speed, max_speed] → (value + max_speed) / (2 * max_speed)
            # So:
            #   -max_speed → 0
            #   0          → 0.5
            #   +max_speed → 1

            vx_norm = (vx + max_speed) / (2.0 * max_speed)
            vy_norm = (vy + max_speed) / (2.0 * max_speed)

            X.append([x_in, y_in])
            Y.append([vx_norm, vy_norm])

    return X, Y



# ------------------------------
# TRAINING
# ------------------------------
"""
    This function splits the dataset into three parts:
    - Training set   (70%)
    - Validation set (15%)
    - Test set       (15%)

    Why split the data?
    - Training set: used to adjust the weights.
    - Validation set: used to tune hyperparameters or detect overfitting.
    - Test set: used at the very end to get an unbiased performance estimate.
"""
def train_val_test_split(X, Y, tr=0.7, vr=0.15):
    idx = list(range(len(X))) # create a list of all indices [0, 1, 2, ..., len(X)-1]
    random.shuffle(idx) # shuffle the indices so we get random mixing of samples.
    n = len(idx)  # total number of samples


    t_end = int(n * tr) #compute the end index for training set
    v_end = int(n * (tr + vr)) #compute the end index for validation set (start of test set).

    Xtr = [X[i] for i in idx[:t_end]] #now use the shuffled indices to create the subsets.
    Ytr = [Y[i] for i in idx[:t_end]]

    Xval = [X[i] for i in idx[t_end:v_end]]
    Yval = [Y[i] for i in idx[t_end:v_end]]

    Xte = [X[i] for i in idx[v_end:]]
    Yte = [Y[i] for i in idx[v_end:]]

    return Xtr, Ytr, Xval, Yval, Xte, Yte


def main():
    print("Loading dataset...")
    X, Y = read_csv(CSV_PATH)

    Xtr, Ytr, Xval, Yval, Xte, Yte = train_val_test_split(X, Y) #split the data into training, validation, and test sets.

    #print how many samples ended up in each subset.
    print(f"Training samples:   {len(Xtr)}")
    print(f"Validation samples: {len(Xval)}")
    print(f"Test samples:       {len(Xte)}")

    #create the MLP model with:
    # - 2 inputs (x_in and y_in)
    # - HIDDEN_SIZE hidden neurons
    # - 2 outputs (vx_norm and vy_norm)
    # - LEARNING_RATE as the learning rate
    model = MLP(2, HIDDEN_SIZE, 2, LEARNING_RATE)
    model.train(Xtr, Ytr, EPOCHS)
    model.save(WEIGHTS_FILE)

    # Validation MSE
    preds = [model.predict(x) for x in Xval]
    val_mse = sum((p[i]-t[i])**2 for p,t in zip(preds,Yval) for i in range(2)) / len(Yval)
    print(f"Validation MSE = {val_mse:.6f}")

    # Test MSE
    preds = [model.predict(x) for x in Xte]
    test_mse = sum((p[i]-t[i])**2 for p,t in zip(preds,Yte) for i in range(2)) / len(Yte)
    print(f"Test MSE = {test_mse:.6f}")


if __name__ == "__main__":
    main()
    