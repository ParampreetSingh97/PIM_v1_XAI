import tensorflow as tf
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
    metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), ],
    )
    history = model.fit(x_train, y_train, 
                        verbose = 1, epochs=10, 
                        batch_size = 1, 
                        validation_data=(x_val,y_val))

    return model, history

def evaluate_model(model, x_test, y_test, class_names):
    yhat_probs = model.predict(x_test, verbose=1)
    yhat1 = np.zeros_like(yhat_probs)
    max_prob_indices = np.argmax(yhat_probs, axis=1)
    for i, index in enumerate(max_prob_indices):
        yhat1[i, index] = 1
    report = classification_report(y_test, yhat1, output_dict=True, target_names = class_names )
    return report
