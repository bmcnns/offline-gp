class ImitationLearner:
    def __init__(self, regressors, histories):
        self.regressors = regressors
        self.histories = histories

    def predict(self, state):
        prediction = []
        for regressor in self.regressors:
            prediction.append(regressor.predict(state))
        return prediction