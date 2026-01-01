# training/train_classifier.py
# Toy script: uses features.csv from tools/parse_logs.py to train q-classifier via synthetic rewards
import numpy as np
import pandas as pd
from agents.classifier.q_learning_classifier import QClassifier
import os

DATA_CSV = os.path.join(os.getcwd(), 'data', 'features.csv')
MODEL_OUT = os.path.join(os.getcwd(), 'agents', 'classifier', 'q_classifier.pkl')

def main():
    df = pd.read_csv(DATA_CSV)
    # build feature vectors (simple): throughput per row
    features = df['throughput_bps'].values
    # normalize
    fnorm = (features - features.min()) / (features.max() - features.min() + 1e-6)
    clf = QClassifier(states=200, actions=4)
    # synthetic training loop: treat high throughput = efficient -> reward mapping
    for ep in range(5):
        for i in range(len(fnorm)-1):
            s = np.array([fnorm[i]])
            s2 = np.array([fnorm[i+1]])
            action = clf.act(s)
            # toy reward: if throughput high and action==0 (noop) -> +1, else small penalty
            reward = 1.0 if (fnorm[i] > 0.7 and action==0) else -0.01
            clf.update(s, action, reward, s2)
    clf.save(MODEL_OUT)
    print("Saved classifier to", MODEL_OUT)

if __name__ == "__main__":
    main()
# placeholder: paste full code here
