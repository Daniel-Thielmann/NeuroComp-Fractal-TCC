from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from hig import hig
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.core.kfold import kfold
from bciflow.datasets.cbcic import cbcic
import numpy as np
import os
import sys
project_directory = os.getcwd()
sys.path.append(project_directory)


dataset = cbcic(subject=1)

pre_folding = {'tf': (chebyshevII, {})}
pos_folding = {'fe': (hig, {}),
               'clf': (lda(), {})}

results = kfold(target=dataset,
                start_window=dataset['events']['cue'][0]+0.5,
                pre_folding=pre_folding,
                pos_folding=pos_folding)

print(results)

true_labels = np.array(results['true_label'])
predict_labels = np.array(
    ['left-hand' if i[0] > i[1] else 'right-hand' for i in np.array(results)[:, -2:]])

accuracy = np.sum(true_labels == predict_labels)/len(true_labels)
print("Acuracia obtida: %.2f" % (accuracy*100))
