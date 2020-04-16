from imported_libraries import *

def lstm_combined(df, args):
  print('hi')
  print('df')

def seq_lstm(area, args):
  print('starting sequential lstm')
  #Make train and test sets:

def get_2020_days_array(df, args):
  date_format = '%Y-%m-%d'
  start_date = datetime.strptime(args.days_to_count_from_lstm, date_format) 
  df = df.astype({'date': str})
  date_array = [(datetime.strptime(i, date_format) - start_date).days for i in df['date'] ]
  return date_array

def lstm(area, args):
  #Hard-coded Variables to be fixed
  np.random.seed(7)
  df = area.cv_days_df_not_scaled
  percent_train_df = args.train_set_percentage
  min_entries_df = 5
  train_length = 60
  predict_var = 'total_deaths'

  #Convert Y-M-D to corresponding day in 2020 (e.g. Jan 2 2020 = day 2 of 2020, Dec 31 is day -1 of 2020, format needed for LSTM)
  df['days'] = get_2020_days_array(df, args)
  df = df[['days', 'new_cases', 'new_deaths', 'total_cases', 'total_deaths']]
  print(df)

  #create dataframe lists
  df_list = list()
  for i in range(len(df.index)):
    if i < min_entries_df: #only keep df if it has min number of entries
      continue
    df_list.append(df[0:i+args.days_of_cv_predict]) #here could also create week and month predictions

  #Split into train and test sets
  train_set_length = int(len(df_list)*percent_train_df)
  train_set = df_list[:train_set_length + 1]
  test_set = df_list[train_set_length:]
  #Scale datasets based on last train set 
  scaling_set = train_set[-1]
  scaling_set_X, scaling_set_Y = get_X_Y(scaling_set, predict_var)
  X_scaler = MinMaxScaler(feature_range = (-1,1))
  X_scaler = X_scaler.fit(scaling_set_X)
  Y_scaler = MinMaxScaler(feature_range = (-1,1))
  Y_scaler = Y_scaler.fit(scaling_set_Y)

  X_list_train = list()
  Y_list_train = list()
  for i in range(len(train_set)):
    this_train_set = train_set[i]
    this_train_X, this_train_Y = get_X_Y(this_train_set, predict_var,  X_scaler, Y_scaler, train_length)
    X_list_train.append(this_train_X)
    Y_list_train.append(this_train_Y)
  final_train_X = np.array(X_list_train)
  final_train_Y = np.array(Y_list_train)

  X_list_test = list()
  Y_list_test = list()
  for i in range(len(test_set)):
    this_test_set = test_set[i]
    this_test_X, this_test_Y = get_X_Y(this_test_set, predict_var, X_scaler, Y_scaler, train_length)
    X_list_test.append(this_test_X)
    Y_list_test.append(this_test_Y)
  final_test_X = np.array(X_list_test)
  final_test_Y = np.array(Y_list_test)
  final_train_Y = final_train_Y.reshape(-1,1)
  final_test_Y = final_test_Y.reshape(-1,1)

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

  #Design Network
  model = Sequential()
  model.add(Masking(mask_value = -100, input_shape=(final_train_X.shape[1], final_train_X.shape[2])))
  model.add(LSTM(100, input_shape = (final_train_X.shape[1], final_train_X.shape[2])))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer = 'adam')
  history = model.fit(final_train_X, final_train_Y, epochs = 100, validation_data = (final_test_X, final_test_Y), verbose = 1, shuffle = False)

  #Plot actual and prediction with real days
  plt.close('all')
  y_train_predict = Y_scaler.inverse_transform(model.predict(final_train_X))
  y_train_actual = Y_scaler.inverse_transform(final_train_Y)
  y_test_predict = Y_scaler.inverse_transform(model.predict(final_test_X))
  y_test_actual = Y_scaler.inverse_transform(final_test_Y)
  x_array_train = get_days(X_scaler, final_train_X, args)
  x_array_test = get_days(X_scaler, final_test_X, args)

  all_predict = np.concatenate((y_train_predict, y_test_predict), axis = 0)
  all_data = np.concatenate((y_train_actual, y_test_actual), axis = 0)
  #rmse = math.sqrt(mean_squared_error(all_predict, all_data))

  rmse = get_rmse(y_test_predict, y_test_actual)
  print('here')
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

def get_days(scaler, data, args):
  days = list()
  date_format = '%Y-%m-%d'
  start_date = datetime.strptime(args.days_to_count_from_lstm, date_format) 
  for i in range(len(data)):
    X_train = scaler.inverse_transform(data[i])
    X_train = X_train[:,0]
    X_train = round(X_train[-1],0) + 1
    print(X_train)

    date = start_date + timedelta(X_train -1)
    days.append(date)

    #df = df.astype({'date': str})
    #df['days'] = [(datetime.strptime(i, date_format) - start_date).days for i in df['date'] ]
  print(days)
  return days


def get_X_Y(df, predict_var, X_scaler=0, Y_scaler=0, length = 0):
  X = df.iloc[:-1,:] #X is everything except the last row

  if X_scaler != 0 and Y_scaler != 0:
    #Y is the predict var in the last row
    Y = df[predict_var].iloc[-1]
    Y = Y.reshape(1,-1)
    #Standardize X and Y
    scaled_X = X_scaler.transform(X)
    scaled_Y = Y_scaler.transform(Y)
    #Pad X matrices to have same length
    n_rows_X = scaled_X.shape[0]
    n_rows_to_add = length - n_rows_X
    pad_rows = np.empty((n_rows_to_add, scaled_X.shape[1]), float)
    pad_rows[:] = -100
    padded_scaled_X = np.concatenate((pad_rows, scaled_X))
    return padded_scaled_X, scaled_Y
  else:
    #Allow Y scaler to scale all Y train values, scaling only one does not work
    Y = df[[predict_var]]
    return X, Y 





















