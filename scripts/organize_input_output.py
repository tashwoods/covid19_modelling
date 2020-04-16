from imported_libraries import *

def daterange(start_date, end_date):
  for n in range(int((end_date - start_date).days)):
    yield start_date + timedelta(n)

def get_days(scaler, data, args):
  days = list()
  date_format = '%Y-%m-%d'
  start_date = datetime.strptime(args.days_to_count_from_lstm, date_format) 
  for i in range(len(data)):
    X_train = scaler.inverse_transform(data[i])
    X_train = X_train[:,0]
    X_train = round(X_train[-1],0) + 1
    date = start_date + timedelta(X_train -1)
    days.append(date)

  return days

def get_X_Y(df, args, seq_output = 0, X_scaler=0, Y_scaler=0):
  print('getting X and Y')
  print(seq_output)
  if seq_output == 0: #predicting one day
    X = df.iloc[:-1,:] #X is everything except the last row
    if X_scaler != 0 and Y_scaler != 0:
      #Y is the predict var in the last row
      Y = df[args.predict_var].iloc[-1]
      Y = Y.reshape(1,-1)
      #Standardize X and Y
      scaled_X = X_scaler.transform(X)
      scaled_Y = Y_scaler.transform(Y)
      #Pad X matrices to have same length
      n_rows_X = scaled_X.shape[0]
      n_rows_to_add = args.lstm_seq_length - n_rows_X
      pad_rows = np.empty((n_rows_to_add, scaled_X.shape[1]), float)
      pad_rows[:] = args.mask_value_lstm
      padded_scaled_X = np.concatenate((pad_rows, scaled_X))
      return padded_scaled_X, scaled_Y
    else:
      #Allow Y scaler to scale all Y train values, scaling only one does not work
      Y = df[[args.predict_var]]
      return X, Y 
  else: #Predict Sequence
    print('seq get X Y')
    print('full df')
    print(df)
    train = df.iloc[:-args.days_of_cv_predict,:] #exclude last n entries of df to use for prediction
    test = df.iloc[-args.days_of_cv_predict:,:]
    print('train')
    print(train)
    print('test')
    print(test)
    test = test[args.predict_var].to_numpy().reshape(1,-1)

    if X_scaler != 0 and Y_scaler != 0:
      Y = df[args.predict_var]
      #Standardize X and Y
      scaled_X = X_scaler.transform(train)
      scaled_Y = Y_scaler.transform(test)
      #Pad X matrices to have same length
      n_rows_X = scaled_X.shape[0]
      n_rows_to_add = args.lstm_seq_length - n_rows_X
      pad_rows = np.empty((n_rows_to_add, scaled_X.shape[1]), float)
      pad_rows[:] = args.mask_value_lstm
      padded_scaled_X = np.concatenate((pad_rows, scaled_X))
      return padded_scaled_X, scaled_Y
    else:
      return train, test

def get_2020_days_array(df, args):
  date_format = '%Y-%m-%d'
  start_date = datetime.strptime(args.days_to_count_from_lstm, date_format) 
  df = df.astype({'date': str})
  date_array = [(datetime.strptime(i, date_format) - start_date).days for i in df['date'] ]
  return date_array







def simple_get_first_cv_day(country, population, args):
  cv_thres_per_mil = population*(1/args.cv_day_thres) #will give first day that one in cv_day_thres people in country affected via predict var
  cv_thres_not_scaled = args.cv_day_thres_notscaled

  truncated_list_per_mil = country[country[args.predict_var] > cv_thres_per_mil]
  truncated_list_not_scaled = country[country[args.predict_var] > cv_thres_not_scaled]

  first_index_per_mil = -1
  first_index_not_scaled = -1
  if len(truncated_list_per_mil) > 0:
    first_cv_day = truncated_list_per_mil.iloc[0].name
    first_index_per_mil = country.index.get_loc(first_cv_day)
  if len(truncated_list_not_scaled) > 0:
    first_cv_day = truncated_list_not_scaled.iloc[0].name
    first_index_not_scaled = country.index.get_loc(first_cv_day)
  return first_index_per_mil, first_index_not_scaled

def get_first_cv_day(country_object, scale):
  args = country_object.input_args
  country = country_object.df
  population = country_object.population
  if scale == 'scaled':
    cv_thres = population*(1/args.cv_day_thres) #will give first day that one in cv_day_thres people in country affected via predict var
  elif scale == 'notscaled':
    cv_thres = args.cv_day_thres_notscaled
  else:
    print('Set threshold for number of {} that starts COVID Outbreak Day'.format(args.predict_var))
    exit()
  truncated_list = country[country[args.predict_var] > cv_thres]
  if len(truncated_list) > 0:
    first_cv_day = truncated_list.iloc[0].name
    first_index = country.index.get_loc(first_cv_day)
    return(first_index)
  else:
    print('trunchated list empty')
    return(-1)

def get_train_test_sets(df, args, lstm = 0):
  if lstm == 0:
    train_set_length = int(len(df.index)*args.train_set_percentage)
    train_set = df[:train_set_length]
    test_set = df[train_set_length:]
  else: #input is a list of dataframes for lstm
    train_set_length = int(len(df)*args.train_set_percentage)
    train_set = df[:train_set_length + 1]
    test_set = df[train_set_length:]
  return train_set, test_set



def get_population(area, pop_df, region_name, pop_name, state = -1):
  area_df = pop_df.loc[pop_df[region_name]== area]
  if state != -1:
    area_df = area_df.loc[area_df['STATE'] == state]
  #if area == 'King County': #For Seattle need to choose King county in WA not TX
  #  area_df = area_df.loc[area_df['STNAME']== 'Washington']
  population = int(area_df[pop_name])
  return population

def translate_country_name(country):
  if country == 'Korea, South':
    return 'South Korea'
  if country == 'US':
    return 'United States'
  else:
    return country

def get_cv_days_df(area_df, population, args):
  start_date_per_mil, start_date_not_scaled = simple_get_first_cv_day(area_df, population, args)
  area_df_per_mil = area_df[start_date_per_mil:].reset_index()
  area_df_not_scaled = area_df[start_date_not_scaled:].reset_index()
  cv_days_df_per_mil = pd.DataFrame()
  cv_days_df_not_scaled = pd.DataFrame()
  if start_date_per_mil != -1:
    cv_days_df_per_mil = area_df_per_mil
    cv_days_df_per_mil[args.name_cv_days] = area_df_per_mil.index.values
  if start_date_not_scaled != -1:
    cv_days_df_not_scaled = area_df_not_scaled
    cv_days_df_not_scaled[args.name_cv_days] = area_df_not_scaled.index.values

  return cv_days_df_per_mil, cv_days_df_not_scaled

def add_attributes(dataset): #Natasha: make this less hardcoded and more dynamic
  #dataset['Open_Close'] = dataset['Open']/dataset['Close']
  #dataset['Low_High'] = dataset['Low']/dataset['High']
  dataset['Close_Open_Change'] = (dataset['Close'] - dataset['Open'])/dataset['Open']
  dataset['High_Open_Change'] = (dataset['High'] - dataset['Open'])/dataset['Open']
  dataset['Low_Open_Change'] = (dataset['Open'] - dataset['Low'])/dataset['Open']
  return dataset

def averaged_dataframe(dataset, days):
  dfs = list()
  indices = np.arange(0, len(dataset.index), days)
  for i in range(len(indices) - 1):
    this_dataframe = dataset.iloc[indices[i]:indices[i+1]]
    dfs.append(this_dataframe.mean(axis=0))
  combined_dataframe = pd.concat(dfs,axis=1).T
  return combined_dataframe

def averaged_dataframe_array(dataset, days):
  split_data = [dataset[i:i+days] for i in range(0,dataset.shape[0],days)]
  return split_data

def make_test_train_datasets(file_name, args):
  #Check metadata of given stock
  formatted_data_unscaled = get_data(file_name, args.date_name)
  if len(args.drop_columns) > 0:
    formatted_data_unscaled = formatted_data_unscaled.drop(args.drop_columns, axis = 1)
  if args.combined_features == 1:
    formatted_data_unscaled = add_attributes(formatted_data_unscaled)

  #Predict_var at end of dataframe to make using StandardScaler easier later
  columns = list(formatted_data_unscaled.columns)
  columns.remove(args.predict_var)
  columns.append(args.predict_var)
  formatted_data_unscaled = formatted_data_unscaled[columns]

  #Get indicies for test and train sets
  first_test_date = get_day_of_year(args.year_test_set + args.month_test_set + args.day_test_set)
  string_last_test_date = str(int(args.year_test_set) + 1 ) + '0101'
  last_test_date = get_day_of_year(str(int(args.year_test_set) + 1 ) + '0101') #Natasha this is hard-coded to only test over a year span
  first_test_index = (formatted_data_unscaled[args.date_name] >= first_test_date).idxmax()
  last_test_index = (formatted_data_unscaled[args.date_name] >= last_test_date).idxmax()

  #Extract train and test set
  train_set_unscaled = formatted_data_unscaled[:first_test_index]
  test_set_unscaled = formatted_data_unscaled[first_test_index:]

  #Average test set over user defined days_in_week if test_set_averaged true
  if args.test_set_averaged:
    test_set_unscaled = averaged_dataframe(test_set_unscaled, args.days_in_week)

  #Order test and train set by date
  train_set_unscaled = train_set_unscaled.sort_values(by = args.date_name)
  test_set_unscaled = test_set_unscaled.sort_values(by = args.date_name)
  
  return test_set_unscaled, train_set_unscaled, formatted_data_unscaled

def get_data(file_name, date_name):
  #reformat date from Y-M-D to Y.day/365
  formatted_data = pd.read_csv(file_name) 
  formatted_data[date_name] = formatted_data[date_name].str.replace('-','').astype(int)
  formatted_data[date_name] = formatted_data[date_name].apply(get_day_of_year)
  return formatted_data

def get_day_of_year(date):
  date = pd.to_datetime(date, format='%Y%m%d')
  first_day = pd.Timestamp(year=date.year, month=1, day=1)
  days_in_year = get_number_of_days_in_year(date.year)
  day_of_year = date.year+(((date - first_day).days)/(days_in_year)) 
  return day_of_year

def get_number_of_days_in_year(year):
  first_day = pd.Timestamp(year,1,1)
  last_day = pd.Timestamp(year,12,31)
  #Add 1 to days_in_year (e.g. 20181231 -->2018.99 and 20190101 --> 2019.00)
  number_of_days = (last_day - first_day).days + 1
  return number_of_days

def make_output_dir(output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  else:
    make_dir_again = input("Outputdir: {} already exists. Delete it (y/n)?".format(output_dir))
    if make_dir_again == 'y':
      shutil.rmtree(output_dir)
      os.makedirs(output_dir)
    else:
      print('Delete/rename {} or run something else ;). Exiting.'.format(output_dir))
      exit()
    return

def make_nested_dir(output_dir, nested_dir):
 Path(output_dir + '/' + nested_dir).mkdir(parents=True, exist_ok=True) 


