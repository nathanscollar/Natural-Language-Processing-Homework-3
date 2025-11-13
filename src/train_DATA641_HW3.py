# train.py

import numpy as np
from sklearn.metrics import f1_score
from models_DATA641_HW3 import build_model, get_optimizer

def train_and_evaluate(model_type, activation, optimizer_name, seq_len,
                       clip, clipnorm_val, epochs, batch_size,
                       padded_data, y_train, y_test):

    # pad data to required length
    X_train, X_test = padded_data[seq_len]

    # use functions from models_DATA641_HW3 to get optimizer and build the model
    opt = get_optimizer(optimizer_name, clip=clip, clipnorm_val=clipnorm_val)
    model = build_model(model_type=model_type, activation=activation, seq_len=seq_len)

    # compile the model with binary crossentropy loss
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    print(f"\n🧪 Training {model_type.upper()} | Activation: {activation} | Optimizer: {optimizer_name.upper()} "
          f"| Seq Len: {seq_len} | Gradient Clipping: {clip}")
    
    # fit model with 20% of training data used for validation
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # evaluate and print test accuracy
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {acc:.4f}")

    # calculate F1 score
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")

    # not required, but for my own understanding, the below code tells me what percent
    # of the predictions are from each class (0 or 1)
    unique, counts = np.unique(y_pred, return_counts=True)
    percentages = counts / len(y_pred) * 100
    for label, pct in zip(unique, percentages):
        print(f"Predicted {label}: {pct:.2f}% of the test set")

    return acc, history