# simple tabular Q-learning classifier (toy)
import random
import pickle
import numpy as np

class QClassifier:
    def __init__(self, states=100, actions=4, alpha=0.1, gamma=0.9, eps=0.1):
        # discrete state buckets; for real use replace with NN
        self.states = states
        self.actions = actions
        self.Q = np.zeros((states, actions))
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def discretize(self, feature_vector):
        # naive hashing to bucket
        s = int(abs(hash(tuple(np.round(feature_vector,3)))) % self.states)
        return s

    def act(self, feature_vector):
        s = self.discretize(feature_vector)
        if random.random() < self.eps:
            return random.randrange(self.actions)
        return int(np.argmax(self.Q[s]))

    def update(self, feature_vector, action, reward, next_vector):
        s = self.discretize(feature_vector)
        s2 = self.discretize(next_vector)
        best_next = np.max(self.Q[s2])
        self.Q[s,action] += self.alpha * (reward + self.gamma*best_next - self.Q[s,action])

    def save(self, path):
        with open(path,'wb') as f:
            pickle.dump(self.Q, f)

    def load(self, path):
        with open(path,'rb') as f:
            self.Q = pickle.load(f)

if __name__ == "__main__":
    print("QClassifier toy ready")
# placeholder: paste full code here
