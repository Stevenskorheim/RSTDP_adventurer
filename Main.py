import random
import itertools

# Initialization variables
INPUT_LAYER_ROWS = 3
INPUT_LAYER_COLS = 3
MIDDLE_LAYER_ROWS = 6
MIDDLE_LAYER_COLS = 6
OUTPUT_LAYER_ROWS = 2
OUTPUT_LAYER_COLS = 1
RECORDING_INTERVAL = 1  # Record every 1 time step
EPOCH_LENGTH = 500
NUM_EPOCHS = 10
RELEASE_NOISE = 0.1
SYNAPSE_WEIGHT = 0.05
NEURON_RESET_VOLTAGE = -1
NEURON_VOLTAGE_DECAY = 0.01
NEURON_SODIUM_DECAY = 0.05
NEURON_POTASSIUM_DECAY = 0.03
SYNAPSE_CALCIUM_DECAY = 0.001
SYNAPSE_DOPAMINE_DECAY = 0.01
PLASTICITY_VARIABLE = 0.0001
STIMULATION_TIME = 50
MAX_SYNAPSE_WEIGHT = 0.5
MIN_SYNAPSE_WEIGHT = 0

class Neuron:
    def __init__(self, resting_voltage=0, threshold=1):
        self.membrane_voltage = resting_voltage
        self.resting_voltage = resting_voltage
        self.threshold = threshold
        self.sodium_input = 0
        self.potassium = 0
        self.incoming_synapses = []
        self.outgoing_synapses = []
        self.fire_count = 0

    def apply_decay(self, input_voltage=0):
        # Add sodium input to the membrane voltage
        self.membrane_voltage += self.sodium_input

        # Decay the sodium input
        if self.sodium_input != 0:
            self.sodium_input *= (1 - NEURON_SODIUM_DECAY)
            if abs(self.sodium_input) < 1e-10:
                self.sodium_input = 0

        # Decay the potassium
        if self.potassium != 0:
            self.potassium *= (1 - NEURON_POTASSIUM_DECAY)
            if abs(self.potassium) < 1e-10:
                self.potassium = 0

        # Check if the neuron should fire
        if self.membrane_voltage >= self.threshold:
            self.fire()

        # Decay the membrane voltage
        if self.membrane_voltage != 0:
            self.membrane_voltage = self.membrane_voltage * (1 - NEURON_VOLTAGE_DECAY) + self.resting_voltage * NEURON_VOLTAGE_DECAY + input_voltage
            if abs(self.membrane_voltage) < 1e-10:
                self.membrane_voltage = 0

    def connect_incoming_synapse(self, synapse):
        self.incoming_synapses.append(synapse)

    def connect_outgoing_synapse(self, synapse):
        self.outgoing_synapses.append(synapse)

    def fire(self):
        self.membrane_voltage = NEURON_RESET_VOLTAGE
        self.potassium += 1  # Increase potassium by 1 when the neuron fires
        self.fire_count += 1  # Increment the fire count

        for synapse in self.outgoing_synapses:
            if isinstance(synapse.output_neuron, list):
                for neuron in synapse.output_neuron:
                    noise_factor = random.uniform(1 - RELEASE_NOISE, 1 + RELEASE_NOISE)
                    neuron.sodium_input += synapse.weight * noise_factor
                    neuron.potassium += 1  # Increase potassium by 1 in the connected neurons
                    synapse.calcium += neuron.potassium  # Increase synapse's calcium by the output neuron's potassium
            else:
                noise_factor = random.uniform(1 - RELEASE_NOISE, 1 + RELEASE_NOISE)
                synapse.output_neuron.sodium_input += synapse.weight * noise_factor
                synapse.output_neuron.potassium += 1  # Increase potassium by 1 in the connected neuron
                synapse.calcium += synapse.output_neuron.potassium  # Increase synapse's calcium by the output neuron's potassium

        for synapse in self.incoming_synapses:
            synapse.calcium -= synapse.input_neuron.potassium  # Decrease synapse's calcium by the input neuron's potassium

class Layer:
    def __init__(self, rows, cols):
        self.neurons = [[Neuron() for _ in range(cols)] for _ in range(rows)]

    def apply_neuron_decay(self, input_voltages=None):
        if input_voltages is None:
            input_voltages = [[0] * len(row) for row in self.neurons]

        for i, row in enumerate(self.neurons):
            for j, neuron in enumerate(row):
                neuron.apply_decay(input_voltages[i][j])

    def reset_neurons(self):
        for row in self.neurons:
            for neuron in row:
                neuron.sodium_input = 0
                neuron.membrane_voltage = neuron.resting_voltage
                neuron.fire_count = 0  # Reset the fire count

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

    def stimulate_input_neurons(self, stimulate_odd):
        for i, row in enumerate(self.input_layer.neurons):
            for j, neuron in enumerate(row):
                if stimulate_odd and (i + j) % 2 == 1:
                    neuron.sodium_input += 0.2
                elif not stimulate_odd and (i + j) % 2 == 0:
                    neuron.sodium_input += 0.2

    def get_membrane_voltages(self):
        input_voltages = [[neuron.membrane_voltage for neuron in row] for row in self.input_layer.neurons]
        middle_voltages = [[neuron.membrane_voltage for neuron in row] for row in self.middle_layer.neurons]
        output_voltages = [[neuron.membrane_voltage for neuron in row] for row in self.output_layer.neurons]

        return {
            "input_layer": input_voltages,
            "middle_layer": middle_voltages,
            "output_layer": output_voltages
        }

    def reset_brain(self):
        self.input_layer.reset_neurons()
        self.middle_layer.reset_neurons()
        self.output_layer.reset_neurons()

class Synapse:
    def __init__(self, input_neuron, output_neuron):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.weight = SYNAPSE_WEIGHT
        self.calcium = 0
        self.dopamine = 0

    def apply_decay(self, calcium_input=0, dopamine_input=0):
        if self.calcium != 0:
            self.calcium = self.calcium * (1 - SYNAPSE_CALCIUM_DECAY) + calcium_input
            if abs(self.calcium) < 1e-10:
                self.calcium = 0

        if self.dopamine != 0:
            self.dopamine = self.dopamine * (1 - SYNAPSE_DOPAMINE_DECAY) + dopamine_input
            if abs(self.dopamine) < 1e-10:
                self.dopamine = 0

    def update_weight(self):
        self.weight += self.dopamine * self.calcium * PLASTICITY_VARIABLE
        self.weight = max(min(self.weight, MAX_SYNAPSE_WEIGHT), MIN_SYNAPSE_WEIGHT)
def main():
    brain = Brain()
    total_time_steps = EPOCH_LENGTH * NUM_EPOCHS

    # Connect synapses between input and middle layers
    for row in brain.input_layer.neurons:
        for neuron in row:
            middle_neurons = random.sample(list(itertools.chain(*brain.middle_layer.neurons)), 3)
            for middle_neuron in middle_neurons:
                synapse = Synapse(neuron, middle_neuron)
                neuron.connect_outgoing_synapse(synapse)
                middle_neuron.connect_incoming_synapse(synapse)

    # Connect synapses between middle and output layers
    for row in brain.middle_layer.neurons:
        for neuron in row:
            output_row = random.choice(brain.output_layer.neurons)
            output_neuron = random.choice(output_row)
            synapse = Synapse(neuron, output_neuron)
            neuron.connect_outgoing_synapse(synapse)
            output_neuron.connect_incoming_synapse(synapse)

    with open("history.txt", "w") as file:
        for epoch in range(NUM_EPOCHS):
            # Reset sodium input and membrane voltage at the start of each epoch
            brain.reset_brain()

            # Randomly select odd or even neurons to stimulate
            stimulate_odd = random.choice([True, False])

            for t in range(EPOCH_LENGTH):
                # Apply decay to neurons
                brain.apply_brain_decay()

                # Stimulate input neurons at the specified time step
                if t == STIMULATION_TIME:
                    brain.stimulate_input_neurons(stimulate_odd)

                # Update synapse weights
                for layer in [brain.input_layer, brain.middle_layer]:
                    for row in layer.neurons:
                        for neuron in row:
                            for synapse in neuron.outgoing_synapses:
                                synapse.update_weight()

                # Record membrane voltages based on the recording interval
                if t % RECORDING_INTERVAL == 0:
                    membrane_voltages = brain.get_membrane_voltages()
                    file.write(f"Epoch: {epoch}, Time Step: {t}\n")
                    for layer, voltages in membrane_voltages.items():
                        file.write(f"{layer}:\n")
                        for row in voltages:
                            file.write(f"{row}\n")
                    file.write("\n")

            # Update dopamine levels based on the epoch's stimulation pattern
            if stimulate_odd:
                for layer in [brain.input_layer, brain.middle_layer]:
                    for row in layer.neurons:
                        for neuron in row:
                            for synapse in neuron.outgoing_synapses:
                                if synapse.output_neuron == brain.output_layer.neurons[0][0]:
                                    synapse.dopamine += 0.1 * brain.output_layer.neurons[0][0].fire_count
                                elif synapse.output_neuron == brain.output_layer.neurons[1][0]:
                                    synapse.dopamine -= 0.1 * brain.output_layer.neurons[1][0].fire_count
            else:
                for layer in [brain.input_layer, brain.middle_layer]:
                    for row in layer.neurons:
                        for neuron in row:
                            for synapse in neuron.outgoing_synapses:
                                if synapse.output_neuron == brain.output_layer.neurons[0][0]:
                                    synapse.dopamine -= 0.1 * brain.output_layer.neurons[0][0].fire_count
                                elif synapse.output_neuron == brain.output_layer.neurons[1][0]:
                                    synapse.dopamine += 0.1 * brain.output_layer.neurons[1][0].fire_count

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
                    elif synapse.output_neuron in list(itertools.chain(*brain.output_layer.neurons)):
                        output_layer = "Output Layer"
                        output_location = list(itertools.chain(*brain.output_layer.neurons)).index(synapse.output_neuron)
                    file.write(f"Input Layer, ({i}, {j}), {output_layer}, {output_location}, {synapse.weight}\n")

        for i, row in enumerate(brain.middle_layer.neurons):
            for j, neuron in enumerate(row):
                for synapse in neuron.outgoing_synapses:
                    output_layer = None
                    output_location = None
                    if synapse.output_neuron in list(itertools.chain(*brain.output_layer.neurons)):
                        output_layer = "Output Layer"
                        output_location = list(itertools.chain(*brain.output_layer.neurons)).index(synapse.output_neuron)
                    file.write(f"Middle Layer, ({i}, {j}), {output_layer}, {output_location}, {synapse.weight}\n")

# Run the main function
if __name__ == "__main__":
    main()
