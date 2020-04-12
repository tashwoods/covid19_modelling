from imported_libraries import *

def lstm(area, args):
  print('---------------------LSTM----------')
  #HARDCODED THINGS TO FIX LATER
  np.random.seed(7)
  df = area.cv_days_df_not_scaled
  number_of_test_dfs = -3 #number of days to keep for testing
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
  #for i in range(len(train_set)):
  X_list_train = list()
  Y_list_train = list()
  for i in range(0,2):
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
  model.add(LSTM(units=30, return_sequences = True, input_shape = (final_train_X.shape[1],final_train_X.shape[2])))
  model.add(LSTM(units=30, return_sequences = True))
  model.add(LSTM(units=30))
  model.add(Dense(units=1))
  model.summary()
  model.compile(optimizer='adam', loss = 'mean_squared_error')
  model.fit(final_train_X, final_train_Y, batch_size=1)
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
    pad_rows[:] = np.NaN
    padded_scaled_X = np.concatenate((pad_rows, scaled_X))
    return padded_scaled_X, scaled_Y
  else:
    #Allow Y scaler to scale all Y train values, scaling only one does not work
    Y = df[[predict_var]]
    return X, Y 
























'''
def new_lstm(area, args):
  print('---------------------LSTM----------')
  #HARDCODED THINGS TO FIX LATER
  np.random.seed(7)
  df = area.cv_days_df_not_scaled
  number_of_test_dfs = -3 #number of days to keep for testing
  min_entries_df = 4 
  train_length = 10
  predict_var = 'total_deaths'

  print('original df')
  print(df)
  date_format = '%Y-%m-%d'
  start_date = datetime.strptime(args.days_to_count_from_lstm, date_format) 
  df = df.astype({'date': str})
  df['days'] = [(datetime.strptime(i, date_format) - start_date).days for i in df['date'] ]
  df = df[['days', 'new_cases', 'new_deaths', 'total_cases', 'total_deaths']]
  df = df.values

  df = df.astype('float32')
  df = np.reshape(df, (-1,1))
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = df
  print('the dataset')
  print(dataset)
  dataset = scaler.fit_transform(dataset) #WRONG
  train_size = int(len(dataset) * 0.80)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

  def create_dataset(dataset, look_back=1):
    X, Y = [], []
    print('in create dataset')
    print(dataset)
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        print('a')
        print(a)
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

  look_back = 30
  X_train, Y_train = create_dataset(train, look_back)
  X_test, Y_test = create_dataset(test, look_back)

  # reshape input to be [samples, time steps, features]
  X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
  X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

  model = Sequential()
  model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
  #model.add(Dropout(0.2))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')

  history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test), verbose = 1, shuffle = False)
#                      callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

  model.summary()

  train_predict = model.predict(X_train)
  test_predict = model.predict(X_test)
  # invert predictions
  train_predict = scaler.inverse_transform(train_predict)
  Y_train = scaler.inverse_transform([Y_train])
  test_predict = scaler.inverse_transform(test_predict)
  Y_test = scaler.inverse_transform([Y_test])
  print('llllll')
  print(train_predict)
  print('xtrain')
  print(X_train.shape)
  print(X_train)
  print('xtest')
  print(X_test.shape)
  print(X_test)

  aa=[x for x in range(100)]
  array_train = [x for x in range(X_train.shape[1])]
  array_test = [x for x in range(X_test.shape[1])]
  plt.figure(figsize=(8,4))
  plt.plot(array_train, Y_train, label = 'train predict')
  plt.plot(array_test, Y_test, label = 'test precition')
  plt.savefig('test.pdf')
  #plt.plot(aa, Y_test[0][:200], marker='.', label="actual")
  #plt.plot(aa, test_predict[:,0][:200], 'r', label="prediction")
  # plt.tick_params(left=False, labelleft=True) #remove ticks
  #plt.tight_layout()
  #sns.despine(top=True)
  #plt.subplots_adjust(left=0.07)
  #plt.ylabel('Global_active_power', size=15)
  #plt.xlabel('Time step', size=15)
  plt.legend(fontsize=15)
  plt.show();
'''
