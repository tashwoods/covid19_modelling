from imported_libraries import *

def lstm_combined(area_obj_list, args):
  print('hi')

def lstm(area, args):
  print('---------------------LSTM----------')
  #HARDCODED THINGS TO FIX LATER
  np.random.seed(7)
  df = area.cv_days_df_not_scaled
  number_of_test_dfs = -10 #number of days to keep for testing
  min_entries_df = 5
  train_length = 60
  predict_var = 'total_deaths'

  print('original df')
  print(df)
  date_format = '%Y-%m-%d'
  start_date = datetime.strptime(args.days_to_count_from_lstm, date_format) 
  df = df.astype({'date': str})
  df['days'] = [(datetime.strptime(i, date_format) - start_date).days for i in df['date'] ]
  df = df[['days', 'new_cases', 'new_deaths', 'total_cases', 'total_deaths']]
  print('dataframe typessss')
  print(df.dtypes)

  #create dataframe lists
  df_list = list()
  for i in range(len(df.index)):
    if i < min_entries_df: #only keep df if it has min number of entries
      continue
    df_list.append(df[0:i]) #here could also create week and month predictions
    this_df = df[0:i]

  #Split into train and test sets
  train_set = df_list[:number_of_test_dfs]
  test_set = df_list[number_of_test_dfs:]
  #Determine how to scale based on last train set 
  #print('Finding scalers----')
  scaling_set = train_set[-1]
  scaling_set_X, scaling_set_Y = get_X_Y(scaling_set, predict_var)
  X_scaler = MinMaxScaler(feature_range = (-1,1))
  X_scaler = X_scaler.fit(scaling_set_X)
  Y_scaler = MinMaxScaler(feature_range = (-1,1))
  Y_scaler = Y_scaler.fit(scaling_set_Y)
  #print('train set Y')
  #print(scaling_set_Y)
  #print('Y_scaler train set')
  #print(Y_scaler.transform(scaling_set_Y))
  #Process training sets
  print('processing training sets')

  X_list_train = list()
  Y_list_train = list()
  #for i in range(0,2):
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
    print('testing sets formatting----------')
    this_test_set = test_set[i]
    this_test_X, this_test_Y = get_X_Y(this_test_set, predict_var, X_scaler, Y_scaler, train_length)
    X_list_test.append(this_test_X)
    Y_list_test.append(this_test_Y)
  final_test_X = np.array(X_list_test)
  final_test_Y = np.array(Y_list_test)
  


  print('final train X shape')
  print(final_train_X.shape)
  print('final train Y shape')
  print(final_train_Y.shape)
  print('final test X shape')
  print(final_test_X.shape)
  print('final test Y shape')
  print(final_test_Y.shape)
  #final_train_X = final_train_X.reshape(final_train_X.shape[0], final_train_X.shape[1], final_train_X.shape[2])
  final_train_Y = final_train_Y.reshape(-1,1)
  #final_test_X = final_test_X.reshape(final_test_X.shape[0], final_test_X.shape[1], final_test_X.shape[2])
  final_test_Y = final_test_Y.reshape(-1,1)

  #Design Network
  model = Sequential()
  model.add(Masking(mask_value = -100, input_shape=(final_train_X.shape[1], final_train_X.shape[2])))
  model.add(LSTM(100, input_shape = (final_train_X.shape[1], final_train_X.shape[2])))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer = 'adam')

  '''
  model.add(LSTM(units=30, return_sequences = True, input_shape = (final_train_X.shape[1],final_train_X.shape[2])))
  model.add(LSTM(units=30, return_sequences = True, input_shape = (final_train_X.shape[1],final_train_X.shape[2])))
  model.add(LSTM(units=30, return_sequences = True))
  model.add(LSTM(units=30))
  model.add(Dense(units=1))
  model.add(Dropout(0.2))
  '''
  model.summary()
  history = model.fit(final_train_X, final_train_Y, epochs = 100, validation_data = (final_test_X, final_test_Y), verbose = 1, shuffle = False)
  # plot history
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.savefig('history.pdf')

  #Plot actual and prediction
  y_predict = Y_scaler.inverse_transform(model.predict(final_test_X))
  y_actual = Y_scaler.inverse_transform(final_test_Y)
  x_array = np.linspace(0,len(y_predict)-1, len(y_predict))
  print(x_array)
  plt.close('all')
  plt.plot(x_array, y_predict, label = 'prediction')
  plt.plot(x_array, y_actual, label = 'actual')
  plt.legend()
  plt.savefig('comparelstm.pdf')

  exit()



  #model = Sequential()
  #model.add(LSTM(5, batch_input_shape = (1,final_train_X.shape[1], final_train_X.shape[2])))
  #model.add(Dense(1))
  #model.compile(loss='mae', optimizer = 'adam')
  #Fit Network
  #history = model.fit(final_train_X, final_train_Y, epochs = 3, batch_size = 1, validation_data = (final_test_X, final_test_Y), verbose = 2, shuffle = False)

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





















