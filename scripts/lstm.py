from imported_libraries import *

def lstm_combined(df, args):
  print('hi')
  print('df')

def seq_lstm(area, args):
  print('starting sequential lstm')
  #Make train and test sets:

def create_df_list(df, args, seq_output):
  df_list = list()
  for i in range(len(df.index)):
    if i < args.days_of_cv_predict + args.min_entries_df_lstm: #only keep df if it has min number of entries
      continue
    if seq_output == 0:
      df_list.append(df[0:i]) #here could also create week and month predictions
    else:
      df_list.append(df[0:i]) #here could also create week and month predictions
  return df_list

def get_X_Y_scaler(scaling_set_X, scaling_set_Y, seq_output = 0):
  X_scaler = MinMaxScaler(feature_range = (-1,1))
  X_scaler = X_scaler.fit(scaling_set_X)
  Y_scaler = MinMaxScaler(feature_range = (-1,1))
  Y_scaler = Y_scaler.fit(scaling_set_Y)
  #print('scaling set transformed')
  #print(X_scaler.transform(scaling_set_X))
  #print(Y_scaler.transform(scaling_set_Y))
  return X_scaler, Y_scaler

def get_X_Y_test_train_scaled(X_scaler, Y_scaler, train_set, test_set, args, seq_output):
  X_list_train = list()
  Y_list_train = list()
  print('len train set: {}'.format(len(train_set)))
  print('len test set: {}'.format(len(test_set)))
  for i in range(len(train_set)):
    print('i: {}'.format(i))
    this_train_set = train_set[i]
    print(this_train_set)
    this_train_X, this_train_Y = get_X_Y(this_train_set, args, seq_output, X_scaler, Y_scaler)
    X_list_train.append(this_train_X)
    Y_list_train.append(this_train_Y)
    '''
    print(this_train_X)
    print(this_train_Y)
    '''
    '''
    if i == 0 or i== len(train_set) - 1:
      print('train X, y')
      print(this_train_X)
      print(this_train_Y)
    '''
  final_train_X = np.array(X_list_train)
  final_train_Y = np.array(Y_list_train)
  final_train_Y = np.squeeze(final_train_Y)

  X_list_test = list()
  Y_list_test = list()
  for i in range(len(test_set)):
    this_test_set = test_set[i]
    this_test_X, this_test_Y = get_X_Y(this_test_set, args, seq_output, X_scaler, Y_scaler)
    X_list_test.append(this_test_X)
    Y_list_test.append(this_test_Y)
    '''
    if i == 0 or i== len(test_set) - 1:
      print('test X, y')
      print(this_test_X)
      print(this_test_Y)
    '''
  final_test_X = np.array(X_list_test)
  final_test_Y = np.array(Y_list_test)
  final_test_Y = np.squeeze(final_test_Y)
  if seq_output == 0:
    final_train_Y = final_train_Y.reshape(-1,1)
    final_test_Y = final_test_Y.reshape(-1,1)
  print('returning')

  return final_train_X, final_train_Y, final_test_X, final_test_Y

def get_unscaled_train_test_X_Y(X_scaler, Y_scaler, model, final_train_X, final_train_Y, final_test_X, final_test_Y, args):
  y_train_predict = Y_scaler.inverse_transform(model.predict(final_train_X))
  y_test_predict = Y_scaler.inverse_transform(model.predict(final_test_X))
  x_array_train = get_days(X_scaler, final_train_X, args)
  x_array_test = get_days(X_scaler, final_test_X, args)
  return x_array_train, y_train_predict, x_array_test, y_test_predict

def get_Y_data_unscaled(Y_scaler, final_train_Y, final_test_Y):
  y_train_actual = Y_scaler.inverse_transform(final_train_Y)
  y_test_actual = Y_scaler.inverse_transform(final_test_Y)
  return y_train_actual, y_test_actual

def lstm(area, args, seq_output = 0):
  print(area.name)
  #np.random.seed(7) need to think about when to use/not use this
  df = area.cv_days_df_not_scaled
  print(df)
  #Convert Y-M-D to corresponding day in 2020 (e.g. Jan 2 2020 = day 2 of 2020, Dec 31 is day -1 of 2020, format needed for LSTM)
  df[args.name_cv_days] = get_2020_days_array(df, args)
  #Only keep desired variables for training
  df = df[[args.name_cv_days, 'new_cases', 'new_deaths', 'total_cases', 'total_deaths']] #NATASHA USE PARSER VAR
  #Create list of samples (dataframes)
  df_list = create_df_list(df, args, seq_output)
  #Split list of samples into training and testing sets
  train_set, test_set = get_train_test_sets(df_list, args, 1) #1 specifies dataframes for lstm

  #Scale train and test sets and split into X and Y
  scaling_set = train_set[-1].copy()
  scaling_set_X, scaling_set_Y = get_X_Y(scaling_set, args, seq_output) #0 specifies we are predicting one value not sequence
  np.set_printoptions(suppress=True)
  X_scaler, Y_scaler = get_X_Y_scaler(scaling_set_X, scaling_set_Y)
  #Create arrays of training X and Y samples. Create arrays of test X and Y samples.
  final_train_X, final_train_Y, final_test_X, final_test_Y = get_X_Y_test_train_scaled(X_scaler, Y_scaler, train_set, test_set, args, seq_output)

  #Design Network
  hidden_layer_dimensions = 100
  dropout = 0.2
  n_batch = 1
  epochs = 1000
  model = Sequential()

  if seq_output == 0:
    model.add(Masking(mask_value = args.mask_value_lstm, input_shape=(final_train_X.shape[1], final_train_X.shape[2])))
    model.add(LSTM(hidden_layer_dimensions, activation = 'relu', input_shape = (final_train_X.shape[1], final_train_X.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1))
  else:
    model.add(Masking(mask_value = args.mask_value_lstm, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2])))
    model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful=True))
    model.add(Dense(final_train_Y.shape[1]))

  model.compile(loss='mean_squared_error', optimizer = 'adam')
  print(model.summary())
  model.fit(final_train_X, final_train_Y, epochs=epochs, batch_size=n_batch, verbose=1, shuffle=False, validation_data = (final_test_X, final_test_Y))

  forecasts = list()
  dates = list()
  for i,j in zip(final_train_X, final_train_Y):
    i = i.reshape(n_batch, i.shape[0], i.shape[1])
    forecasts.append(Y_scaler.inverse_transform(model.predict(i, batch_size = n_batch)))
    dates_df = pd.DataFrame(np.squeeze(i))
    original_df = pd.DataFrame(X_scaler.inverse_transform(dates_df))
    last_day = original_df.iloc[-1][0] +1
    predicted_days = np.arange(last_day, last_day + args.days_of_cv_predict)
    dates.append(predicted_days)

  forecasts_test = list()
  dates_test = list()
  for i, j in zip(final_test_X, final_train_Y):
    i = i.reshape(n_batch, i.shape[0], i.shape[1])
    forecasts_test.append(Y_scaler.inverse_transform(model.predict(i, batch_size = n_batch)))
    dates_df = pd.DataFrame(np.squeeze(i))
    original_df = pd.DataFrame(X_scaler.inverse_transform(dates_df))
    last_day = original_df.iloc[-1][0] +1
    predicted_days = np.arange(last_day, last_day + args.days_of_cv_predict)
    dates_test.append(predicted_days)


  print('done')
  print(forecasts)
  print(dates)
  for i,j in zip(dates, np.squeeze(forecasts)):
    #plt.plot(i,j, color = 'blue', label = 'Training Set Predictions')
    plt.plot(i,j, color = 'blue')
  for i,j in zip(dates_test, np.squeeze(forecasts_test)):
    #plt.plot(i,j, color = 'green', label = 'Test Set Predictions')
    plt.plot(i,j, color = 'green')
  print(train_set)
  plt.plot(df['cv_days'], df['total_deaths'], color = 'black', label = 'data')
  plt.savefig(args.output_dir + '/lstm.png')
  exit()
  print('out of loop')

  print(forecasts)
  print(final_train_X)
  print(final_train_Y)
  print(len(train_set))
  print(len(test_set))
  exit()
  for i, j in zip(dates, forecasts):
    print('in loop')
    print(i)
    print(j)
    plt.plot(i,j)
  plt.savefig(args.output_dir + '/lstm.png')
  '''
  exit()
  nb_epoch = 2
  for i in range(nb_epoch):
    model.fit(final_train_X, final_train_Y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
    print(model.summary())
    #exit()
    print(i)
    #model.reset_states()
  #history = model.fit(final_train_X, final_train_Y, epochs = 100, verbose = 1, shuffle = False, batch_size = 1)
  print('here')
  '''

  #Plot Data and prediction with real days
  #Obtain unscaled X + Y datasets for test, train, and actual data
  x_array_train, y_train_predict, x_array_test, y_test_predict = get_unscaled_train_test_X_Y(X_scaler, Y_scaler, model, final_train_X, final_train_Y, final_test_X, final_test_Y, args)
  y_train_actual, y_test_actual = get_Y_data_unscaled(Y_scaler, final_train_Y, final_test_Y)
  all_predict = np.concatenate((y_train_predict, y_test_predict), axis = 0)
  all_data = np.concatenate((y_train_actual, y_test_actual), axis = 0)
  rmse = get_rmse(y_test_predict, y_test_actual)
  print(y_test_predict)
  print(y_test_actual)
  print(rmse)
  plt.plot(x_array_train, y_train_predict, color = 'orange', label = 'LSTM: ' + str(round(rmse,2)))
  plt.plot(x_array_test, y_test_predict, color = 'orange')
  plt.plot(x_array_train, y_train_actual, 'bo-', label = 'Train Set', markersize = args.markersize)
  plt.plot(x_array_test, y_test_actual, 'go-', label = 'Test Set', markersize = args.markersize)
  plt.title(area.name)
  plt.ylabel('Total Deaths')
  plt.xlabel('Date')
  plt.xticks(fontsize = args.tick_font_size)
  plt.legend()
  plt.savefig(args.output_dir + '/' + area.name + '_lstm_realdays.png')
  plt.close('all')

  ''' make this a module Natasha
  #LSTM Hyperparamters
  n_neurons = 1
  neuron_array = [1]
  dropout = 0.2
  epochs = 100
  repeats = 1
  #Test epoch size vs RMSE
  for i in range(repeats):
    model = Sequential()
    model.add(Masking(mask_value = -100, input_shape=(final_train_X.shape[1], final_train_X.shape[2])))
    model.add(LSTM(n_neurons, input_shape = (final_train_X.shape[1], final_train_X.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    history = model.fit(final_train_X, final_train_Y, epochs = epochs, validation_data = (final_test_X, final_test_Y), verbose = 1, shuffle = False)
    model.reset_states()
    #Plot history
    if i == 0:
      plt.plot(history.history['loss'], label = 'Train', color = 'blue')
      plt.plot(history.history['val_loss'], label = 'Test', color = 'orange')
    else:
      plt.plot(history.history['loss'], color = 'blue')
      plt.plot(history.history['val_loss'], color = 'orange')
  plt.xlabel('Epoch')
  plt.ylabel('RMSE')
  plt.legend()
  plt.savefig(args.output_dir + '/' + area.name + '_epoch_rmse.png')
  '''


