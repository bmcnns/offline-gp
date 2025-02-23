class ImitationLearner:
    def __init__(self, regressors, histories, scalers):
        self.regressors = regressors
        self.histories = histories
        self.scalers = scalers

    def predict(self, state):
        prediction = []
        for regressor in self.regressors:
            prediction.append(regressor.predict(state))
        return prediction
