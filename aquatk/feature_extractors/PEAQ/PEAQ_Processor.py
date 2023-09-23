class PeaqModulationProcessor:
    def __init__(self):
        # Initialize variables here
        self.ear_model = None
        self.average_loudness = None
        self.modulation = None

    def set_ear_model(self, ear_model):
        self.ear_model = ear_model

    def get_ear_model(self):
        return self.ear_model

    def process(self, unsmeared_excitation):
        band_count = self.ear_model.get_band_count()
        step_size = self.ear_model.get_step_size()
        sampling_rate = self.ear_model.get_sampling_rate()
        derivative_factor = sampling_rate / step_size

        for k in range(band_count):
            # Calculate loudness and its derivative (Equation 54 in [BS1387])
            loudness = np.power(unsmeared_excitation[k], 0.3)
            loudness_derivative = derivative_factor * abs(loudness - self.previous_loudness[k])

            # Update filtered loudness derivative (Equation 55 in [BS1387])
            self.filtered_loudness_derivative[k] = (
                    self.ear_time_constants[k] * self.filtered_loudness_derivative[k] +
                    (1 - self.ear_time_constants[k]) * loudness_derivative
            )

            # Update filtered loudness (Equation 55 in [BS1387])
            self.filtered_loudness[k] = (
                    self.ear_time_constants[k] * self.filtered_loudness[k] +
                    (1 - self.ear_time_constants[k]) * loudness
            )

            # Calculate modulation (Equation 57 in [BS1387])
            self.modulation[k] = self.filtered_loudness_derivative[k] / (1 + self.filtered_loudness[k] / 0.3)

            # Update previous loudness for the next iteration
            self.previous_loudness[k] = loudness

    def get_average_loudness(self):
        return self.average_loudness

    def get_modulation(self):
        return self.modulation

