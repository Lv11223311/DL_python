def my_input_fn(features, targets, batch_size = 1, shuffle = True, num_epochs = None):
    """
    Trains a linear regression model of one feature.

    Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: true or False.Whether to shuffle the data.
        num_epochs:NUmber of epochs for which data sould be repeated. None = repeat indefinitely
    
    Return:
    Tuple of (features, labels) for nex data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, ande configure batching/repeating.
    ds = Dataset.from_tensor_slices(features, targets)
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size = 10000)

    # Return the nex batch of data.
    features, labels = ds.make_one_shot_iterator().get_nex()
    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature = 'total_rooms'):
    """
    Trains a linear regression model of one feature.

    Args:
        learning_rate:A'flota', the leatning rate.
        steps: A non-zero'int', the total number of training steps. A training step
            consisits of a forward ande backward pass using a singlg batch.
        batch_size : A non_zero'int', the batch size.
        input_features: A 'string' specifying a column from 'california_housing_dataframe
            to ues as input feature
    """

    perides = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = 'median_house_value'
    targets = california_housing_dataframe[my_label]

    # Creat feature columns.
    feature_columns = [tf.feature_column.numeric_column(mt_feature)]

    # Create input functions.
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size =batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data,targets, num_epochs=1, shuffle= False)

    #Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    my_optimizer= tf.contrib.estima