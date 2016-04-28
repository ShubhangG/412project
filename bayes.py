from Naive_Bayes_Algorithm import *
import pandas as pd
f = 'data/train_users_2.csv'
df = pd.DataFrame(pd.read_csv(f))
mean, std = Setup_Naive_Bayes(df)
testfile = 'data/test_users.csv'
testdf = pd.DataFrame(pd.read_csv(testfile))
testids = testdf['id']

predicted_labels = Naive_Bayes_Predictor(testdf, mean, std)
outputs = []
counts = defaultdict(int)
for i in range(len(predicted_labels)):
	counts[predicted_labels[i]] += 1

for i in range(len(predicted_labels)):
	curr = (testids[i], predicted_labels[i], counts[predicted_labels[i]])
	outputs.append(curr)

outputs = sorted(outputs, key=itemgetter(2), reverse=True)
for curr in outputs:
	print curr[0] + ',' + curr[1]

