# evaluate.py

from preprocess_DATA641_HW3 import load_and_preprocess
from train_DATA641_HW3 import train_and_evaluate
from utils_DATA641_HW3 import set_seed

# modify file path as needed
file_path = "C:/Users/natha/Documents/IMDB/IMDB Dataset.csv"
# use the load_and_preprocess function from preprocess_DATA641_HW3.py to load and preprocess the data
padded_data, y_train, y_test = load_and_preprocess(file_path)

# run code, using the set_seed function from utils_DATA641_HW3.py before each run to ensure randomness
# the below code contains three test entries
set_seed(42)
acc1, _ = train_and_evaluate('RNN', 'tanh', 'adam', 25, True, 5, 5, 32, padded_data, y_train, y_test)
set_seed(42)
acc2, _ = train_and_evaluate('RNN', 'tanh', 'adam', 50, True, 5, 5, 32, padded_data, y_train, y_test)
set_seed(42)
acc3, _ = train_and_evaluate('RNN', 'tanh', 'adam', 100, True, 5, 5, 32, padded_data, y_train, y_test)
