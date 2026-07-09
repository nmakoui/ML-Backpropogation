from lander_mlp_neuron_based import MLP
import math

class NeuralNetHolder:
    #this class is the bridge that connects the trained neural network and the game. the network is trained offline in lander_mlp_neuron_based,
    #then the weights are saved and by using this code, those weights are uploaded to the actual game so it can actually use them.

    def __init__(self):
        print("Loading neural network for game...")

        #much like the training, the width and height of the game screen is needed, cause we need the DIAG, which is our normalisation scale.
        #it's been mentioned during our lectures times and times again that it is extremely important that the normalisation used during prediction should
        #match the normalisation used during training, otherwise the NN will not behave correctly
        self.SCREEN_W = 800
        self.SCREEN_H = 600
        #Pythagorean theorem is used again to calculate the DIAG
        self.DIAG = math.sqrt(self.SCREEN_W ** 2 + self.SCREEN_H ** 2)

        #here the same exact same MLP architecture is used, which we used during the training
        self.model = MLP(2, 6, 2, 0.03)
        #we load the trained weights from the txt file produced by the lander_mlp_neuron_based file
        self.load_weights("lander_neuron_weights.txt")

    def safe_float(self, v, fallback=0.0):

        #this is a helper function, it converts values to float, one of the main issues I faced was my NN crashing due to unexpected errors
        #so this kinda acts like a safety blanket and even if it fails to convert the values to float, it returns a fallback value
        try:
            return float(v)
        except:
            return fallback

    def load_weights(self, path):
        #this function loads all the trained weights and biases in to the model and the format is exactly same as how the weights were saved in the training file
        #mhaving the same structure makes sure that each and every neuron receives the exact corresponding weights used during the training
        print(f"Loading weights from file: {path}")
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        idx = 0

        #----------------------------
        #LOADING HIDDEN LAYER WEIGHTS
        #----------------------------

        for neuron in self.model.hidden:
            neuron.weights = [float(w) for w in lines[idx].split(",")]  #comma is used to split because weights are separated by commas
            neuron.bias = float(lines[idx + 1])
            idx += 2

        idx += 1  # skip "#"

        #----------------------------
        #LOADING OUTPUT LAYER WEIGHTS
        #----------------------------

        for neuron in self.model.output:
            neuron.weights = [float(w) for w in lines[idx].split(",")]
            neuron.bias = float(lines[idx + 1])
            idx += 2

        print("Weights loaded successfully!")

    def predict(self, input_row):
        """
        this function recieves raw game inputs which are x_target and y_target and normlaises them exactly like the training,
        feeds them to the trained NN and then converts the outputs given by the NN back in to some real velocities that the game can use
    
        """

        # ----------------------------
        # 1. SAFE PARSING OF INPUT ROW
        # ----------------------------
        #inputs can be lists or x,y pairs (strings)
        if isinstance(input_row, str):
            parts = input_row.split(",")
        else:
            parts = list(input_row)

        
        #if there are any values are missing, default to 0
        if len(parts) < 2:
            x_raw = 0.0
            y_raw = 0.0
        else:
            x_raw = self.safe_float(parts[0], 0.0)
            y_raw = self.safe_float(parts[1], 0.0)

        # Clamp absurd values (also known as safety clamp)
        x_raw = max(-2000.0, min(2000.0, x_raw))
        y_raw = max(-2000.0, min(2000.0, y_raw))

        # --------------------------------------------
        # 2. INPUT NORMALIZATION (MUST MATCH TRAINING)
        # --------------------------------------------
        #as mentioned, it is very important to normalise using the exact same method as the one used in training
        x_norm = (x_raw + self.DIAG / 2.0) / self.DIAG
        y_norm = (y_raw + self.DIAG / 2.0) / self.DIAG

        # ------------------
        # 3. NN FORWARD PASS
        # ------------------
        #the model.predict() function returns [vx_norm, vy_norm] in [0,1].
        out = self.model.predict([x_norm, y_norm])

        # ---------------------------------------
        # 4. DE-NORMALIZE BACK TO TRUE VELOCITIES
        # ---------------------------------------
        max_speed = 8.0

        #output from NN is in [0,1]
        vx_norm = max(0.0, min(1.0, out[0]))
        vy_norm = max(0.0, min(1.0, out[1]))

        #convert normalised velocities back to real velocities:
        # 0   → -8
        # 0.5 →  0
        # 1   → +8
        vx = vx_norm * (2.0 * max_speed) - max_speed
        vy = vy_norm * (2.0 * max_speed) - max_speed

        #final safety clamping, it is there to prevent extreme velocity outputs
        if vx >  max_speed: vx =  max_speed
        if vx < -max_speed: vx = -max_speed
        if vy >  max_speed: vy =  max_speed
        if vy < -max_speed: vy = -max_speed

        return [vx, vy]