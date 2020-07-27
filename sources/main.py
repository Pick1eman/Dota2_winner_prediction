import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dataset_file = ""
for i in sys.argv[0].split("/")[:-2]:
	dataset_file += i + "/"
dataset_file += "resources"
train_targets_file_path = dataset_file + "/train_targets.csv"
train_features_file_path = dataset_file + "/train_features.csv"
test_features_file_path = dataset_file + "/test_features.csv"
sample_submission_path = dataset_file + "/sample_submission.csv"
output_submission_path = dataset_file + "/output_submission.csv"

x = pd.read_csv(train_features_file_path).drop("match_id_hash", axis = 1)
series_class = pd.read_csv(train_targets_file_path)
y = series_class["radiant_win"]

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.05)
random_forest = RandomForestClassifier(n_estimators=2000)
random_forest.fit(x_train, y_train)

prediction = random_forest.predict(x_valid)
validation_accuracy = accuracy_score(prediction, y_valid)
print(validation_accuracy)

x_test = pd.read_csv(test_features_file_path).drop("match_id_hash", axis=1)
y_predict_proba_test = random_forest.predict_proba(x_test)[:, 1]
print(y_predict_proba_test)

output_submission = pd.read_csv(sample_submission_path)
output_submission["radiant_win_prob"] = y_predict_proba_test
# print(output_submission[:0])
output_submission.to_csv(output_submission_path, index=False, columns=["match_id_hash", "radiant_win_prob"])
