from imported_libraries import *

def create_df_list(df, args, seq_output):
  df_list = list()
  for i in range(len(df.index)):
    if i < args.days_of_cv_predict + args.min_entries_df_lstm: #only keep df if it has min number of entries
      continue
    if seq_output == 0:
      df_list.append(df[0:i].copy()) #here could also create week and month predictions
    else:
      df_list.append(df[0:i].copy()) #here could also create week and month predictions
  return df_list

def get_X_Y_scaler(scaling_set_X, scaling_set_Y, seq_output = 0):
  X_scaler = MinMaxScaler(feature_range = (-1,1))
  X_scaler = X_scaler.fit(scaling_set_X)
  Y_scaler = MinMaxScaler(feature_range = (-1,1))
  Y_scaler = Y_scaler.fit(scaling_set_Y)
  return X_scaler, Y_scaler

def get_X_Y_test_train_scaled(X_scaler, Y_scaler, train_set, test_set, args, seq_output, scaled = 1):
  X_list_train = list()
  Y_list_train = list()
  print('len train set: {}'.format(len(train_set)))
  print('len test set: {}'.format(len(test_set)))
  for i in range(len(train_set)):
    this_train_set = train_set[i]
    this_train_X, this_train_Y = get_X_Y(this_train_set, args, seq_output, X_scaler, Y_scaler)
    X_list_train.append(this_train_X)
    Y_list_train.append(this_train_Y)
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
  final_test_X = np.array(X_list_test)
  final_test_Y = np.array(Y_list_test)
  final_test_Y = np.squeeze(final_test_Y)
  if seq_output == 0:
    final_train_Y = final_train_Y.reshape(-1,1)
    final_test_Y = final_test_Y.reshape(-1,1)

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

def convert_to_2020_date(array, args):
  date_format = '%Y-%m-%d'
  start_date = datetime.strptime(args.days_to_count_from_lstm, date_format)
  new_dates = list()
  for day in array:
    real_date = start_date + timedelta(days=int(day))
    new_dates.append(real_date.date())
  return new_dates

def get_train_test_X_Y(df, args, seq_output, X_scaler = -1, Y_scaler = -1):
  #Create list of samples (dataframes)
  df_list = create_df_list(df, args, seq_output)
  #Split list of samples into training and testing sets
  train_set, test_set = get_train_test_sets(df_list, args, 1) #1 specifies dataframes for lstm

  #Scale train and test sets and split into X and Y
  scaling_set = train_set[-1].copy()
  scaling_set_X, scaling_set_Y = get_X_Y(scaling_set, args, seq_output) #0 specifies we are predicting one value not sequence
  if X_scaler == -1 and Y_scaler == -1:
    X_scaler, Y_scaler = get_X_Y_scaler(scaling_set_X, scaling_set_Y)
  

  #Create arrays of training X and Y samples. Create arrays of test X and Y samples.
  final_train_X, final_train_Y, final_test_X, final_test_Y = get_X_Y_test_train_scaled(X_scaler, Y_scaler, train_set, test_set, args, seq_output)
  return final_train_X, final_train_Y, final_test_X, final_test_Y, X_scaler, Y_scaler

def get_reshaped_X(input_X, n_batch, X_scaler):
  i = input_X.reshape(n_batch, input_X.shape[0], input_X.shape[1])
  output_df = pd.DataFrame(np.squeeze(input_X))
  original_df = pd.DataFrame(X_scaler.inverse_transform(output_df))
  return original_df

def lstm_combined(area_obj_list, args):
  #Combine all test dataframes to determine correct X_scaler and Y_scalers
  seq_output = 1 #makes lstm predict sequences
  list_df_train = list()
  for i in range(len(area_obj_list)):
    area = area_obj_list[i]
    df = area.cv_days_df_not_scaled.copy()
    df.loc[:,'area_number'] = i #Natasha make this longitude and latitude eventually
    df = df[[args.name_cv_days, args.name_2020_days, args.predict_var, 'new_cases', 'new_deaths', 'total_cases', 'area_number']] #NATASHA USE PARSER VAR
    list_df_train.append(df.iloc[:round(args.train_set_percentage*len(df.index))].copy())
  all_train_dfs = pd.concat(list_df_train)
  final_train_X, final_train_Y, final_test_X, final_test_Y, X_scaler, Y_scaler = get_train_test_X_Y(df, args, seq_output) #0 to say not combined lstm 

  #create scaled train and test samples
  max_samples = 10
  train_samples = list()
  for i in range(max_samples):
    print('-----------------------------')
    print(i)
    df_list = list()
    for area in list_df_train:
      print(len(area.index))
      if i < len(area.index):
        df_list.append(area.loc[:i].copy())
      else:
        df_list.append(area.copy())
    this_day = pd.concat(df_list) 
    print(this_day)

      

    

      
  return

def build_model(args, epochs, n_batch, hidden_layer_dimensions, dropout, final_train_X, final_train_Y):
  patience = 50
  validation_split = 0.1
  model = Sequential()
  model.add(Masking(mask_value = args.mask_value_lstm, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2])))
  model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True, return_sequences = True))
  model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True))
  model.add(Dropout(dropout))
  model.add(Dense(final_train_Y.shape[1]))
  es = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=patience)
  model.compile(loss='mean_squared_error', optimizer = 'adam')
  history = model.fit(final_train_X, final_train_Y, epochs=epochs, batch_size=n_batch, verbose=1, shuffle=False, validation_split=validation_split, callbacks=[es])
  return model, history


def lstm(area, args, seq_output = 0):
  np.set_printoptions(suppress=True)
  df = area.cv_days_df_not_scaled.copy()
  df = df[[args.name_cv_days, args.name_2020_days, 'new_cases', 'new_deaths', 'total_cases', 'total_deaths']] #NATASHA USE PARSER VAR
  final_train_X, final_train_Y, final_test_X, final_test_Y, X_scaler, Y_scaler = get_train_test_X_Y(df, args, seq_output) #0 to say not combined lstm 

  do_hyperparam_opt = 0
  if do_hyperparam_opt:
    layers = 3
    dropout_array = [0, 0.01, 0.05]
    hidden_layer_dimensions_array = [10, 100, 200]
    col_array = plt.cm.jet(np.linspace(0,1,len(dropout_array)))
    lstm_hyperparam_opt(args, n_batch, final_train_X, final_train_Y, layers, hidden_layer_dimensions, dropout, dropout_array, hidden_layer_dimensions_array, epochs, col_array)

  #Design Network
  hidden_layer_dimensions = 100
  dropout = 0.01
  n_batch = 1
  epochs = 2

  if seq_output == 0: # to predict one day out
    model = Sequential()
    model.add(Masking(mask_value = args.mask_value_lstm, input_shape=(final_train_X.shape[1], final_train_X.shape[2])))
    model.add(LSTM(hidden_layer_dimensions, activation = 'relu', input_shape = (final_train_X.shape[1], final_train_X.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1))
  else: # to predict cv_days_predict number of days out
    model_error, _ = build_model(args, epochs, n_batch, hidden_layer_dimensions, 0.1, final_train_X, final_train_Y)
    model, history = build_model(args, epochs, n_batch, hidden_layer_dimensions, dropout, final_train_X, final_train_Y)
    plt.plot(history.history['loss'], color = 'blue', linestyle = 'solid',label = 'Train Set')
    plt.plot(history.history['val_loss'], color = 'green', linestyle = 'solid', linewidth = args.linewidth, label = 'Validation Set')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.savefig(args.output_dir + '/' + area.name + 'lstm_loss_final.png')
    plt.close('all')

  #Plot predictions for test and train sets
  forecasts = list()
  dates = list()
  for i,j in zip(final_train_X, final_train_Y):
    original_df = get_reshaped_X(i, n_batch, X_scaler)
    i = i.reshape(n_batch, i.shape[0], i.shape[1])
    thisforecast = Y_scaler.inverse_transform(model.predict(i, batch_size = n_batch))
    forecasts.append(Y_scaler.inverse_transform(model.predict(i, batch_size = n_batch)))
    last_day = round(original_df.iloc[-1][0] +1) #not sure why rounding is necessary here, but without arrays come out with different shapes
    predicted_days = np.arange(last_day, last_day + args.days_of_cv_predict)
    dates.append(predicted_days)

  forecasts_test = list()
  dates_test = list()
  errors_test = list()
  final_predict = list()
  final_dates = list()
  rms = 0
  for n in range(len(final_test_X)):
    i = final_test_X[n]
    j = final_train_Y[n]
    original_df = get_reshaped_X(i, n_batch, X_scaler)
    i = i.reshape(n_batch, i.shape[0], i.shape[1])
    thisforecast = Y_scaler.inverse_transform(model.predict(i, batch_size = n_batch))
    forecasts_test.append(thisforecast)
    last_day = round(original_df.iloc[-1][0] +1)
    predicted_days = np.arange(last_day, last_day + args.days_of_cv_predict)
    dates_test.append(predicted_days)
    if n == len(final_test_X) - 1: #Retrieve error from last prediction
      error_forecast = Y_scaler.inverse_transform(model_error.predict(i, batch_size = n_batch))
      errors_test = np.squeeze(abs(thisforecast - error_forecast)/2)
      final_predict = np.squeeze(thisforecast)
      final_dates = np.squeeze(predicted_days)
      rms = round(sqrt(mean_squared_error(np.squeeze(thisforecast),np.squeeze(Y_scaler.inverse_transform(j.reshape(-1,1))))),2)
  plt.fill_between(convert_to_2020_date(final_dates,args), final_predict - errors_test, final_predict + errors_test, color = 'green', alpha = 0.1)

  forecasts = np.squeeze(forecasts)
  forecasts_test = np.squeeze(forecasts_test)
  print('about to plot train predictions')
  for n in range(len(dates)): #plot training predictions
    i = dates[n]
    j = forecasts[n]
    newdates = convert_to_2020_date(i,args)

    if n == 0:
      plt.plot(newdates,j, color = 'blue', label = 'Train Set', linewidth = args.linewidth, markersize = 0)
    else:
      plt.plot(newdates,j, color = 'blue', linewidth = args.linewidth, markersize = 0)
  for n in range(len(dates_test)): #plot test predictions
    i = dates_test[n]
    j = forecasts_test[n]
    newdates = convert_to_2020_date(i,args)
    if n == 0:
      plt.plot(newdates,j, color = 'green', label = 'Test Set (RMSE: ' + str(rms) + ')', linewidth = args.linewidth, markersize = 0)
    else:
      plt.plot(newdates,j, color = 'green', linewidth = args.linewidth, markersize = 0)
  plt.xlabel(args.date_name)
  plt.ylabel(args.predict_var)
  plt.xticks(fontsize = args.tick_font_size)
  plt.plot(convert_to_2020_date(df['cv_days'],args), df[args.predict_var], color = 'black', label = 'Data')
  plt.legend()
  plt.title(area.name)
  plt.savefig(args.output_dir + '/' + area.name + '_lstm.png')
  plt.close('all')



def lstm_hyperparam_opt(args, n_batch, final_train_X, final_train_Y, layers, hidden_layer_dimensions, dropout, dropout_array, hidden_layer_dimensions_array, epochs, col_array):
  #Check number of layers--------------------
  for i in range(layers):
    print(i)
    model = Sequential()
    model.add(Masking(mask_value = args.mask_value_lstm, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2])))
    if i == 0:
      model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True))
    if i > 0:
      model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True, return_sequences = True))
    if i > 1:
      model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True, return_sequences = True))
    if i > 2:
      model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True, return_sequences = True))
    if i > 0:
      model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True))
    model.add(Dropout(dropout))
    model.add(Dense(final_train_Y.shape[1]))

    es = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=100)
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    print(model.summary())
    history = model.fit(final_train_X, final_train_Y, epochs=epochs, batch_size=n_batch, verbose=1, shuffle=False, validation_split=0.1, callbacks=[es])
    plt.plot(history.history['val_loss'], color = col_array[i], linestyle = 'solid', linewidth = args.linewidth, label = str(i+1) + ' Layers')
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('RMSE')
  plt.savefig(args.output_dir + '/lstm_loss_layers.png')
  plt.close('all')

  #Check number of dropout--------------------
  for i in range(len(dropout_array)):
    dropout = dropout_array[i]
    model = Sequential()
    model.add(Masking(mask_value = args.mask_value_lstm, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2])))
    model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True, return_sequences = True))
    model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True))
    model.add(Dropout(dropout))
    model.add(Dense(final_train_Y.shape[1]))

    es = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=100)
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    print(model.summary())
    history = model.fit(final_train_X, final_train_Y, epochs=epochs, batch_size=n_batch, verbose=1, shuffle=False, validation_split=0.1, callbacks=[es])
    plt.plot(history.history['val_loss'], color = col_array[i], linestyle = 'solid', linewidth = args.linewidth, label = str(dropout) + ' Dropout')
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('RMSE')
  plt.savefig(args.output_dir + '/lstm_loss_dropout.png')
  plt.close('all')

  #Check number of number of neurons--------------------
  for i in range(len(hidden_layer_dimensions_array)):
    hidden_layer_dimensions = hidden_layer_dimensions_array[i]
    model = Sequential()
    model.add(Masking(mask_value = args.mask_value_lstm, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2])))
    model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True, return_sequences = True))
    model.add(LSTM(hidden_layer_dimensions, batch_input_shape=(n_batch,final_train_X.shape[1], final_train_X.shape[2]), stateful = True))
    model.add(Dropout(dropout))
    model.add(Dense(final_train_Y.shape[1]))

    es = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=100)
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    print(model.summary())
    history = model.fit(final_train_X, final_train_Y, epochs=epochs, batch_size=n_batch, verbose=1, shuffle=False, validation_split=0.1, callbacks=[es])
    plt.plot(history.history['val_loss'], color = col_array[i], linestyle = 'solid', linewidth = args.linewidth, label = str(hidden_layer_dimensions) + ' Neurons')
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('RMSE')
  plt.savefig(args.output_dir + '/lstm_loss_neurons.png')
  plt.close('all')

  return

