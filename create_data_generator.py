import numpy as np

def data_generator(train_gcn_feats, train_adj_list, omic_data1, omic_data2, omic_data3, y_data, batch_size, shuffle=True, peek=False, verbose=True):
    """
    Generates batches of training data for deep learning models.

    Args:
        train_gcn_feats (np.ndarray): Graph convolutional network (GCN) features for training.
        train_adj_list (np.ndarray): Normalized adjacency matrix list.
        omic_data1 (np.ndarray): First omic data input.
        omic_data2 (np.ndarray): Second omic data input.
        omic_data3 (np.ndarray): Third omic data input.
        y_data (np.ndarray): Target labels (e.g., drug response AUC values).
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the data at the start of each epoch. Defaults to True.
        peek (bool, optional): Whether to return the first batch without shuffling. Defaults to False.
        verbose (bool, optional): Whether to print debugging information. Defaults to True.

    Yields:
        tuple: A tuple ((batch_x_df, batch_x_da, batch_x_od1, batch_x_od2, batch_x_od3), batch_y),
        where batch_x_* are feature arrays and batch_y are target labels.
    """
    num_samples = len(train_gcn_feats) # Total number of samples
    indices = np.arange(num_samples) # Create index array
    
    # Return first batch without shuffling
    if peek:
        end = min(batch_size, num_samples) # Ensure we don't exceed dataset size
        if verbose:
            print(f"Peeking: Generating first batch up to index {end}")
        batch_x_df = train_gcn_feats[:end]
        batch_x_da = train_adj_list[:end]
        batch_x_od1 = omic_data1[:end]
        batch_x_od2 = omic_data2[:end]
        batch_x_od3 = omic_data3[:end]
        batch_y = y_data[:end]
        #peek = False
        #yield ([batch_x_df, batch_x_da, batch_x_od1, batch_x_od2, batch_x_od3], batch_y)
        yield ((batch_x_df, batch_x_da, batch_x_od1, batch_x_od2, batch_x_od3), batch_y)

    # Infinite loop for multiple epochs
    while True:
        # Shuffle data at start of each epoch
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples) # Handle last batch properly
            batch_indices = indices[start:end] # Select batch indices

            # Print batch indices for debugging
            if verbose:
                if shuffle and len(batch_indices) < batch_size:
                    print(f"Batch indices: {np.sort(batch_indices)}")
                else:
                    print(f"Generating batch {start} to {end}")

            # Generate batches
            batch_x_df = train_gcn_feats[batch_indices]
            batch_x_da = train_adj_list[batch_indices]
            batch_x_od1 = omic_data1[batch_indices]
            batch_x_od2 = omic_data2[batch_indices]
            batch_x_od3 = omic_data3[batch_indices]
            batch_y = y_data[batch_indices]

            # Yield batch
            yield ((batch_x_df, batch_x_da, batch_x_od1, batch_x_od2, batch_x_od3), batch_y)

def batch_predict(model, data_gen, steps, batch_size, flatten=True, verbose=False):
    """
    Performs batch-wise predictions using a trained model.

    Args:
        model (tf.keras.Model): Trained Keras model for prediction.
        data_gen (generator): Data generator providing batches of input data.
        steps (int): Number of batches to process.
        batch_size (int): Number of samples per batch.
        flatten (bool, optional): Whether to flatten the predictions and labels. Defaults to True.
        verbose (bool, optional): Whether to print progress information. Defaults to False.

    Returns:
        tuple: (predictions, true_values), where both are NumPy arrays.
    """
    # Pre-allocate arrays to avoid appending in a loop
    predictions = np.empty((steps * batch_size,), dtype=np.float32)
    true_values = np.empty((steps * batch_size,), dtype=np.float32)
    
    index = 0  # To keep track of insertion point in the arrays
    
    for step in range(steps):
        x_batch, y_batch = next(data_gen)  # Get the batch
        batch_pred = model.predict(x_batch, verbose=0)  # Predict on batch

        # Flatten predictions and true values if specified
        if flatten:
            batch_pred = batch_pred.flatten()
            y_batch = y_batch.flatten()

        # Insert batch predictions and true values into pre-allocated arrays
        predictions[index:index + len(batch_pred)] = batch_pred
        true_values[index:index + len(y_batch)] = y_batch
        index += len(batch_pred)
        
        # Optional verbose logging
        if verbose:
            print(f"Batch {step + 1}/{steps}:")
            print(f"Batch Predictions: {len(batch_pred)}")
            print(f"Cumulative Predictions: {index}")
    
    # Slice arrays to correct length in case of incomplete final batch
    return predictions[:index], true_values[:index]

def print_duration(activity: str, start_time: float, end_time: float):
    """
    Prints the duration of a given activity in hours, minutes, and seconds.

    Args:
        activity (str): Description of the activity.
        start_time (float): Start time (timestamp).
        end_time (float): End time (timestamp).
    """
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print(f"Time for {activity}: {hours} hours, {minutes} minutes, and {seconds} seconds\n")