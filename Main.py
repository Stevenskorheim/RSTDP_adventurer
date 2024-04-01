import random
import itertools

# Initialization variables
INPUT_LAYER_ROWS = 3
INPUT_LAYER_COLS = 3
MIDDLE_LAYER_ROWS = 6
MIDDLE_LAYER_COLS = 6
OUTPUT_LAYER_ROWS = 2
OUTPUT_LAYER_COLS = 1

class Neuron:
    def __init__(self, resting_voltage=0, threshold=1, reset_voltage=-1, voltage_decay=0.01, sodium_decay=0.05):
        self.membrane_voltage = resting_voltage
        self.resting_voltage = resting_voltage
        self.threshold = threshold
        self.reset_voltage = reset_voltage
        self.voltage_decay = voltage_decay
        self.sodium_input = 0
        self.sodium_decay = sodium_decay
        self.outgoing_synapses = []

    def apply_decay(self, input_voltage=0):
        # Add sodium input to the membrane voltage
        self.membrane_voltage += self.sodium_input

        # Decay the sodium input
        if self.sodium_input != 0:
            self.sodium_input *= (1 - self.sodium_decay)
            if abs(self.sodium_input) < 1e-10:
                self.sodium_input = 0

        # Check if the neuron should fire
        if self.membrane_voltage >= self.threshold:
            self.fire()

        # Decay the membrane voltage
        if self.membrane_voltage != 0:
            self.membrane_voltage = self.membrane_voltage * (1 - self.voltage_decay) + self.resting_voltage * self.voltage_decay + input_voltage
            if abs(self.membrane_voltage) < 1e-10:
                self.membrane_voltage = 0

    def add_random_value(self):
        random_value = random.uniform(-2, 2)
        self.membrane_voltage += random_value

    def connect_synapse(self, synapse):
        self.outgoing_synapses.append(synapse)

    def fire(self):
        self.membrane_voltage = self.reset_voltage
        for synapse in self.outgoing_synapses:
            if isinstance(synapse.output_neuron, list):
                for neuron in synapse.output_neuron:
                    neuron.sodium_input += synapse.weight
            else:
                synapse.output_neuron.sodium_input += synapse.weight

class Layer:
    def __init__(self, rows, cols):
        self.neurons = [[Neuron() for _ in range(cols)] for _ in range(rows)]

    def __getitem__(self, index):
        return self.neurons[index]

    def apply_neuron_decay(self, input_voltages=None):
        if input_voltages is None:
            input_voltages = [[0] * len(row) for row in self.neurons]

        for i, row in enumerate(self.neurons):
            for j, neuron in enumerate(row):
                neuron.apply_decay(input_voltages[i][j])

    def add_random_values(self):
        for row in self.neurons:
            for neuron in row:
                neuron.add_random_value()

class Brain:
    def __init__(self):
        self.input_layer = Layer(INPUT_LAYER_ROWS, INPUT_LAYER_COLS)
        self.middle_layer = Layer(MIDDLE_LAYER_ROWS, MIDDLE_LAYER_COLS)
        self.output_layer = Layer(OUTPUT_LAYER_ROWS, OUTPUT_LAYER_COLS)

    def apply_brain_decay(self, input_voltages=None):
        if input_voltages is None:
            input_voltages = {
                "input_layer": [[0] * INPUT_LAYER_COLS for _ in range(INPUT_LAYER_ROWS)],
                "middle_layer": [[0] * MIDDLE_LAYER_COLS for _ in range(MIDDLE_LAYER_ROWS)],
                "output_layer": [[0] * OUTPUT_LAYER_COLS for _ in range(OUTPUT_LAYER_ROWS)]
            }

        self.input_layer.apply_neuron_decay(input_voltages["input_layer"])
        self.middle_layer.apply_neuron_decay(input_voltages["middle_layer"])
        self.output_layer.apply_neuron_decay(input_voltages["output_layer"])

    def add_random_sodium_input(self):
        for row in self.input_layer.neurons:
            for neuron in row:
                random_value = random.uniform(-2, 2)
                neuron.sodium_input += random_value

    def get_membrane_voltages(self):
        input_voltages = [[neuron.membrane_voltage for neuron in row] for row in self.input_layer.neurons]
        middle_voltages = [[neuron.membrane_voltage for neuron in row] for row in self.middle_layer.neurons]
        output_voltages = [[neuron.membrane_voltage for neuron in row] for row in self.output_layer.neurons]

        return {
            "input_layer": input_voltages,
            "middle_layer": middle_voltages,
            "output_layer": output_voltages
        }

class Synapse:
    def __init__(self, input_neuron, output_neuron, weight=0.04, calcium_decay=0.01, dopamine_decay=0.01):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.weight = weight
        self.calcium = 0
        self.dopamine = 0
        self.calcium_decay = calcium_decay
        self.dopamine_decay = dopamine_decay

    def apply_decay(self, calcium_input=0, dopamine_input=0):
        if self.calcium != 0:
            self.calcium = self.calcium * (1 - self.calcium_decay) + calcium_input
            if abs(self.calcium) < 1e-10:
                self.calcium = 0

        if self.dopamine != 0:
            self.dopamine = self.dopamine * (1 - self.dopamine_decay) + dopamine_input
            if abs(self.dopamine) < 1e-10:
                self.dopamine = 0


def main():
    brain = Brain()
    time_steps = 1000

    # Connect synapses between input and middle layers
    for row in brain.input_layer.neurons:
        for neuron in row:
            middle_neurons = random.sample(list(itertools.chain(*brain.middle_layer.neurons)), 3)
            for middle_neuron in middle_neurons:
                synapse = Synapse(neuron, middle_neuron)
                neuron.connect_synapse(synapse)

    # Connect synapses between middle and output layers
    for row in brain.middle_layer.neurons:
        for neuron in row:
            output_neuron = random.choice(brain.output_layer.neurons[0])
            synapse = Synapse(neuron, output_neuron)
            neuron.connect_synapse(synapse)

    with open("history.txt", "w") as file:
        for t in range(time_steps):
            # Apply decay to neurons
            brain.apply_brain_decay()

            # Add random sodium input to input layer neurons at time step 50
            if t == 50:
                brain.add_random_sodium_input()

            # Record membrane voltages every 100 time steps
            if t % 100 == 0:
                membrane_voltages = brain.get_membrane_voltages()
                file.write(f"Time Step: {t}\n")
                for layer, voltages in membrane_voltages.items():
                    file.write(f"{layer}:\n")
                    for row in voltages:
                        file.write(f"{row}\n")
                file.write("\n")

            # Perform other operations for each time step

    # Print connections to a file
    with open("connections.txt", "w") as file:
        file.write("Input Neuron Layer, Input Neuron Location, Output Neuron Layer, Output Neuron Location, Weight\n")
        for i, row in enumerate(brain.input_layer.neurons):
            for j, neuron in enumerate(row):
                for synapse in neuron.outgoing_synapses:
                    output_layer = None
                    output_location = None
                    if synapse.output_neuron in list(itertools.chain(*brain.middle_layer.neurons)):
                        output_layer = "Middle Layer"
                        output_location = list(itertools.chain(*brain.middle_layer.neurons)).index(synapse.output_neuron)
                    elif synapse.output_neuron in brain.output_layer.neurons[0]:
                        output_layer = "Output Layer"
                        output_location = brain.output_layer.neurons[0].index(synapse.output_neuron)
                    file.write(f"Input Layer, ({i}, {j}), {output_layer}, {output_location}, {synapse.weight}\n")

        for i, row in enumerate(brain.middle_layer.neurons):
            for j, neuron in enumerate(row):
                for synapse in neuron.outgoing_synapses:
                    output_layer = None
                    output_location = None
                    if synapse.output_neuron in brain.output_layer.neurons[0]:
                        output_layer = "Output Layer"
                        output_location = brain.output_layer.neurons[0].index(synapse.output_neuron)
                    file.write(f"Middle Layer, ({i}, {j}), {output_layer}, {output_location}, {synapse.weight}\n")

# Run the main function
if __name__ == "__main__":
    main()

