from imported_libraries import *



def logistic_model(x,a,b,c):
  return c/(1+np.exp(-(x-b)/a))

def fit_logistic_all(area_object_list, scale = 'log', lives_saved = 0, scaled  = 0):
  args = area_object_list[0].input_args
  C = args.cv_day_thres
  n_prediction_days_bar = 15
  linewidth = args.linewidth
  markersize = args.markersize
  col_array = plt.cm.jet(np.linspace(0,1,round(len(area_object_list)/2)+5))
  append_string = 'scaled'
  if scaled == 0:
    append_string = 'unscaled'

  #Calculate South Korea fit for comparison to other countries first
  for area in area_object_list: 
    all_data = area.cv_days_df_per_mil
    name = area.name
    frac = 1
    if scaled == 0:
      frac = area.population/C
    if name == 'South Korea':
      train_set, test_set = get_train_test_sets(all_data, args)
      x = train_set[args.name_cv_days]
      y = train_set[args.n_deaths_per_mil]

      popt, pcov = curve_fit(logistic_model, x, y, p0=[5,20, 0.002*C], bounds=[[0,5,0.00001*C],[20,50,C]])
      south_korea_a, south_korea_b, south_korea_c = popt[0], popt[1], popt[2]

      x_array, _ = get_x_array_for_prediction(all_data, args)
      y_predict = logistic_model(x_array, south_korea_a, south_korea_b, south_korea_c)

      if(lives_saved == 0):
        rmse = get_rmse(frac*logistic_model(test_set[args.name_cv_days], south_korea_a, south_korea_b, south_korea_c), frac*test_set[args.n_deaths_per_mil])
        plt.plot(x_array,frac*y_predict, label = name + ': ' + str(round(frac/south_korea_a,2)) + ' RMSE: ' + str(rmse) , color = col_array[0], linewidth = linewidth)
        plt.scatter(x,frac*y, s = markersize, color = col_array[0])

  #Calculate Fit for other areas and compare to South Korea
  for i in range(len(area_object_list)):
    area = area_object_list[i]
    all_data = area.cv_days_df_per_mil
    frac = 1
    if scaled == 0:
      frac = area.population/C
    if area.fips not in args.nyc_fips_to_skip and area.name != 'South Korea':
      name = area.name
      train_set, test_set = get_train_test_sets(area.cv_days_df_per_mil, args)
      x = train_set[args.name_cv_days]
      y = train_set[args.n_deaths_per_mil]

      #Fit logistic curve to data
      popt, pcov = curve_fit(logistic_model, x, y, p0=[5,20, 0.002*C], bounds=[[0,5,0.00001*C],[20,50,C]])
      a, b, c = popt[0], popt[1], popt[2]
      #Create arrays for plotting future if current behaviors continue
      x_array, _ = get_x_array_for_prediction(all_data, args)
      y_predict = logistic_model(x_array, a, b, c)
      #Calculate offset needed for south korea type future for every country and plot
      lastday, lastentry = x.iloc[-1], y.iloc[-1]
      lastentry_sk = logistic_model(lastday, south_korea_a, south_korea_b, south_korea_c)
      offset = lastentry - lastentry_sk
      x_sk_array = np.linspace(lastday, lastday + args.days_of_cv_predict, args.days_of_cv_predict + 1)
      y_sk_predict = logistic_model(x_sk_array, south_korea_a, south_korea_b, south_korea_c) + offset

      if lives_saved == 0:
        rmse = get_rmse(frac*logistic_model(test_set[args.name_cv_days], a, b, c), frac*test_set[args.n_deaths_per_mil])
        plt.plot(x_array,frac*y_predict, label = name + ': '+ str(round(frac/a,2)) + ' RMSE: ' + str(rmse), color = col_array[i], linewidth = linewidth, linestyle = 'solid')
        plt.plot(x_sk_array, frac*y_sk_predict, color = col_array[i], linestyle = 'dashed', linewidth = linewidth)
        plt.scatter(x,frac*y, s = markersize, color = col_array[i])
      else:
        plt.close('all')
        #Create shorted arrays for lives saved plot
        x_sk_array_fit = x_sk_array[:n_prediction_days_bar]
        y_sk_predict = y_sk_predict[:n_prediction_days_bar]
        #Transform x_array to be 'Days from Now'
        x_sk_array_days_from_now = [x - x_sk_array[0] for x in x_sk_array]
        x_sk_array_days_from_now = x_sk_array_days_from_now[:n_prediction_days_bar]
        #Calculate and plot lives saved
        lives_saved_array = frac*(logistic_model(x_sk_array_fit, a, b, c) - y_sk_predict)
        plt.bar(x_sk_array_days_from_now, lives_saved_array)
        plt.xlabel('Days From Now')
        if scaled == 0:
          plt.ylabel('Lives Saved')
        else:
          plt.ylabel('Lives Saved per ' + get_nice_var(args.cv_day_thres))
        plt.title(name)
        plt.savefig(args.output_dir + '/all_logisitic_lives_saved_' + name + '_' + append_string + '.png')
        plt.close('all')

  plt.yscale(scale)
  plt.ylim(bottom = 1)
  plt.legend(loc = 'lower right', prop={'size':6})
  plt.xlabel('Days since 1 Death per ' + get_nice_var_name(C))
  if scaled == 0:
    plt.ylabel(get_nice_var_name(args.name_total_deaths))
  else:
    plt.ylabel('Deaths per ' + get_nice_var_name(C))

  if lives_saved == 0:
    plt.savefig(args.output_dir + '/allsigmoidfit_' + scale + '_' + append_string + '.png')
  plt.close('all')

  return

def fit_logistic(x, y, C, args, name):
  popt, pcov = curve_fit(logistic_model, x, y, p0=[2,30, 0.02*C], bounds=[[0,5,0.000001*C],[20,500,0.5*C]])
  a = popt[0]
  b = popt[1]
  c = popt[2]
  errors = [np.sqrt(pcov[i][i]) for i in [0,1,2]]
  sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))

  print('total deaths: {} +/- {}'.format(c, errors[2]))
  print('infec speed: {} +/- {}'.format(a, errors[0]))
  print('turning pt: {} +/- {}'.format(b, errors[1]))
  print(100*(c/C))
  print(sol)

  x_array = np.linspace(0, sol+30, sol+1+30)
  y_predict = logistic_model(x_array, popt[0], popt[1], popt[2])

  plt.scatter(x,y, label ='NYT Data')
  plt.plot(x_array,y_predict, label = 'Sigmoid Fit')

  plt.legend()
  plt.title(name)
  plt.savefig(args.output_dir + '/' + name + '_sigmoidfit.pdf')
  plt.close('all')

def make_gif(area_obj_list, dataframe_name, var, start_date, end_date, args):
  print('here-----------')
  df_list = list()
  maxes = list()
  #Create area_object dataframes list per day for GIFs
  area_obj = area_obj_list[0]
  #Format start and end date so they can be iterated through
  start_date_array = start_date.split('-')
  end_date_array = end_date.split('-')
  start_date = date(int(start_date_array[0]), int(start_date_array[1]), int(start_date_array[2]))
  end_date = date(int(end_date_array[0]), int(end_date_array[1]), int(end_date_array[2]))
  #Iterate through dates, create dataframe per date with all need area info for jpg to make gif
  for day in daterange(start_date, end_date):
    fips = list()
    var_list = list()
    cv_days = list()
    #Loop through areas
    for i in range(len(area_obj_list)):
      cv_days.append(day)
      fips.append(area_obj_list[i].fips)
      county = getattr(area_obj_list[i], dataframe_name)

      if county.empty: #if dataframe empty 
        var_list.append(float('nan'))
        continue
      selected_row = county.loc[county[args.date_name]==str(day)]
      if selected_row.empty: #if row empty
        var_list.append(float('nan'))
      else:
        var_list.append(float(selected_row[var].values))

    this_df = pd.DataFrame({args.date_name:cv_days, args.name_fips: fips, var: var_list})
    df_list.append(this_df)

  #Calculate max variable value for heatmap plot
  this_variable_max_array = list()
  for i in range(len(df_list)):
    df = df_list[i]
    this_variable_max_array.append(np.nanpercentile(df[var], args.gif_percentile))

  vmax = np.nanmax(this_variable_max_array)

  #Create jpegs for each day to be combined later into gifs
  filenames = []
  for i in range(len(df_list)):
    df = df_list[i]
    vmin = 0
    counties = geopandas.read_file(args.county_map_file)
    counties[args.name_fips] = counties['id'].astype(int)
    counties_df = pd.DataFrame(counties)

    merged_inner = pd.merge(counties_df, df, how = 'outer', on = args.name_fips)
    merged_inner[var] = merged_inner[var].fillna(-100) #in case some counties have not reported fill empty values with -1 for heatmap
    merged_inner = merged_inner[merged_inner['STATE']!= '15'] #Exlude Hawaii
    merged_inner = merged_inner[merged_inner['STATE']!= '02'] #Exclude Alaska
    merged_inner = merged_inner[merged_inner['STATE'] != '72'] #Exclude Puerto Rico
    combined_geo = geopandas.GeoDataFrame(merged_inner)

    fig, ax = plt.subplots(1,1)
    my_cmap = plt.cm.get_cmap('jet')
    my_cmap.set_under('grey')
    my_cmap.set_over('magenta')
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.rc('font', size = 40)

    ax = combined_geo.plot(column = var, figsize=(60,40), cmap = my_cmap, vmin=vmin, vmax=vmax)
    plt.title(str(df.iloc[0]['date']))
    plt.axis('off')

    # add colorbar
    fig = ax.get_figure()
    cax = fig.add_axes([0.1, 0.2, 0.8, 0.01])
    fig.colorbar(sm, orientation = 'horizontal', aspect = 80, label = get_nice_var_name(var, args), cax = cax, extend='both')

    filename = args.output_dir + '/counties_' + var + '_' + str(i) + '.jpg'
    filenames.append(filename)
    #plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
  make_gif_command = 'convert -delay ' + args.gif_delay + ' '
  for ifile in filenames:
    make_gif_command += ifile + ' '
  make_gif_command += args.output_dir + '/' + var + '.gif'
  os.system(make_gif_command)

def make_gif_cv_days(area_obj_list, dataframe_name, var, ndays, args, thisvmax = 0):
  df_list = list()
  maxes = list()
  #create area_object dataframes list per day for GIFs
  for day in range(ndays):
    fips = list()
    var_list = list()
    cv_days = list()
    for i in range(len(area_obj_list)): #loop thru US counties
      cv_days.append(day)
      fips.append(area_obj_list[i].fips)
      county = getattr(area_obj_list[i], dataframe_name)
      if county.empty: #if dataframe empty 
        var_list.append(float('nan'))
        continue
      if (len(county.index) - 1) < day: #if there is not data for given county for selected day
        var_list.append(float('nan'))
      else:
        var_list.append(county[var].iloc[day])

    this_df = pd.DataFrame({'cv_day':cv_days, 'fips': fips, var: var_list})
    df_list.append(this_df)

  #Calculate max variable value for heatmap plot
  this_variable_max_array = list()
  for i in range(ndays):
    df = df_list[i]
    pd.set_option('display.max_rows', None)
    this_variable_max_array.append(np.nanpercentile(df[var], 10))
  vmax = max(this_variable_max_array)
  vmax = thisvmax

  #Create jpegs for each day to be combined later into gifs
  filenames = []
  for i in range(ndays):
    df = df_list[i]
    vmin = 0

    counties = geopandas.read_file(args.county_map_file)
    counties['fips'] = counties['id'].astype(int)
    counties_df = pd.DataFrame(counties)

    merged_inner = pd.merge(counties_df, df, how = 'outer', on = 'fips')
    merged_inner[var] = merged_inner[var].fillna(-100) #in case some counties have not reported fill empty values with -1 for heatmap
    merged_inner = merged_inner[merged_inner['STATE']!= '15'] #Exlude Hawaii
    merged_inner = merged_inner[merged_inner['STATE']!= '02'] #Exclude Alaska
    merged_inner = merged_inner[merged_inner['STATE'] != '72'] #Exclude Puerto Rico
    combined_geo = geopandas.GeoDataFrame(merged_inner)

    fig, ax = plt.subplots(1,1)
    my_cmap = plt.cm.get_cmap('jet')
    my_cmap.set_under('grey')
    my_cmap.set_over('magenta')
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.rc('font', size = 40)

    ax = combined_geo.plot(column = var, figsize=(60,40), cmap = my_cmap, vmin=vmin, vmax=vmax)
    plt.title('CV Outbreak Day ' + str(i))
    plt.axis('off')

    # add colorbar
    fig = ax.get_figure()
    cax = fig.add_axes([0.1, 0.2, 0.8, 0.01])
    fig.colorbar(sm, orientation = 'horizontal', aspect = 80, label = get_nice_var_name(var, args), cax = cax, extend='both')

    filename = args.output_dir + '/counties_' + var + '_' + str(i) + '.jpg'
    filenames.append(filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
  make_gif_command = 'convert -delay ' + args.gif_delay + ' '
  for ifile in filenames:
    make_gif_command += ifile + ' '
  make_gif_command += args.output_dir + '/' + var + '_cv_days.gif'
  os.system(make_gif_command)


def get_nice_var_name(var, args=0):
  if var == 'total_deaths':
    return 'Total Deaths'
  if var == 'total_cases':
    return 'Total Cases'
  if var == 'new_cases':
    return 'New Cases'
  if var == 'new_deaths':
    return 'New Deaths'
  if var == 1000000:
    return '1M'
  if var == 'deaths_per_mil':
    group_size = '{:,}'.format(args.cv_day_thres)
    return 'Total Deaths per ' + group_size

def get_shifted_prediction(area_df, var, slope, intercept, best_growth_rate, args):
  #Obtain last x,y value from data
  last_x = len(area_df.index.values) - 1
  last_y = np.log10(area_df[var].iloc[-1])
  #Calculate shifted intercept for fit to data, to make last point be in linear prediction 
  shifted_intercept = last_y - (slope * last_x)
  #Calculate shifted intercept for best_growth_rate to make last point be in linear prediction
  best_shifted_intercept = last_y - (best_growth_rate * last_x)
  #Return arrays for x, y_fitted_shifted, y_best_shifted
  last_predict_day = last_x + args.days_of_cv_predict
  x = np.linspace(last_x,last_predict_day, last_predict_day - last_x + 1)
  #Since inputs for y were log(y), convert back to y
  y = 10**((slope * x) + shifted_intercept)
  y_best = 10**((best_growth_rate *x) + best_shifted_intercept)
  return x, y, y_best, shifted_intercept, best_shifted_intercept

def get_lives_saved_bar_chart(x_predict, y_predict, y_best, name, args, savename,scale):
  plt.close('all') 
  #Calculate lives saved as the difference between the prediction and best case scenario
  lives_saved = y_predict - y_best
  #Make x-axis the days from the first day (e.g. Days from Now)
  x_predict = [x - x_predict[0] for x in x_predict]
  plt.xlim(0,args.days_of_cv_predict + 1)
  plt.bar(x_predict, lives_saved)
  plt.xlabel('Days From Now')
  plt.ylabel('Total Lives Saved')
  plt.yscale(scale)
  total_saved = round(lives_saved[-1],3)
  if savename == 'Individual':
    plt.title(name + ': Total Lives One Person Could Save')
  else:
    plt.title(name + ': Total Lives Saved Days from Now')
  plt.text(0.02, 0.9, 'Current Projected Deaths: ' + str(round(y_predict[-1],1)), transform = plt.gca().transAxes)
  plt.text(0.02, 0.8, 'Best Case Projected Deaths: ' + str(round(y_best[-1],1)), transform = plt.gca().transAxes)
  plt.text(0.02, 0.7, 'Lives That Could Be Saved: ' + str(total_saved), transform = plt.gca().transAxes)
  plt.savefig(args.output_dir + '/' + name + 'lives_saved' + savename + '.png')

def plot(area_objects_list, args, plot_type, variables, scale_array):
  #Internal Variables
  covid_days = 'cv_days' #name of covid days variable in dataframes
  col = plt.cm.jet(np.linspace(0,1,round(len(area_objects_list)/2)+5))
  #Plot time series for variables
  for var in variables:
    #Iterate thru specified y-scales
    for scale in scale_array:
      #Iterate thru objects in area_objects_list (e.g. countries, states, counties)
      for i in range(len(area_objects_list)):
        area = area_objects_list[i]
        if area.fips in args.nyc_fips_to_skip: #Skip NYC duplicates, they're identical due to NYT dataset structure
          continue
        #Plot Raw data vs Date
        if plot_type == 'raw_dates': 
          area_df = area.df
          plt.plot(area_df[args.date_name], area_df[var], label = area.name, linewidth = args.linewidth, color = col[i])
          plt.xlabel(args.n_date_name)
          plt.ylabel(get_nice_var_name(var, args))
        #Plot Raw data vs outbreak days
        elif plot_type == 'raw_covid_days' or plot_type == 'lives_saved_raw_covid_days':
          all_data = area.cv_days_df_not_scaled
          train_set, test_set = get_train_test_sets(all_data, args)
          #Calculate max number of days to plot based on days_of_cv_predict
          x, x_max = get_x_array_for_prediction(all_data, args)
          plt.xlim(0,x_max)
          #Fit log(var) and date to line
          model, log_intercept, log_slope, prediction, growth_rate = get_log_fit(train_set, covid_days, var, x)
         #Calculate fitted prediction with constraint that last point matches the last dataset value
          x_predict, y_predict, y_best, shifted_intercept, shifted_intercept_best = get_shifted_prediction(train_set, var, log_slope, log_intercept, args.min_growth_rate, area.input_args)
          y_all = 10**((log_slope * all_data[covid_days]) + shifted_intercept)
          y_test = 10**((log_slope * test_set[covid_days]) + shifted_intercept)
          y_test_best = 10**((args.min_growth_rate * test_set[covid_days]) + shifted_intercept_best)
          test_rmse = get_rmse(y_test, test_set[var])

          #Plot Data and Prediction
          if plot_type == 'raw_covid_days':
            plt.scatter(all_data[covid_days], all_data[var], label = area.name + ':' + str(growth_rate) + ' RMSE: ' + str(test_rmse), s = args.markersize, color = col[i])
            plt.plot(all_data[covid_days], y_all, linestyle = 'solid', color = col[i])
            plt.plot(test_set[covid_days],y_test_best, color = col[i], linestyle = 'dashed')
            plt.xlabel('Days since ' + str(args.cv_day_thres_notscaled) + ' Deaths')
            plt.ylabel(get_nice_var_name(var, args))

          elif plot_type == 'lives_saved_raw_covid_days':
            get_lives_saved_bar_chart(x_predict, y_predict, y_best, area.name, area.input_args, 'All', scale)
            #Calculate Individual impact
            current_indiv_slope = (log_slope/area.population)
            improved_slope = log_slope - (current_indiv_slope - args.min_indiv_growth_rate)
            x_predict, y_predict_indiv, y_best_indiv, shifted_intercept, shifted_intercept_best = get_shifted_prediction(train_set, var, log_slope, log_intercept, improved_slope, area.input_args)
            get_lives_saved_bar_chart(x_predict, y_predict_indiv, y_best_indiv, area.name, area.input_args, 'Individual', scale)

        #Plot var/1M vs outbreak days
        elif plot_type == 'per_mil_covid_days':
          all_data = area.cv_days_df_per_mil
          train_set, test_set = get_train_test_sets(all_data, args)
          #Get max number of cv outbreak days to determine plot limits
          x, x_max = get_x_array_for_prediction(all_data, args)
          #Fit log(var)
          model, log_intercept, log_slope, prediction, growth_rate = get_log_fit(train_set, covid_days, var, x)
          #Calculate fitted prediction with constraint that last point matches the last dataset value
          x_predict, y_predict, y_best, shifted_intercept, shifted_intercept_best = get_shifted_prediction(train_set, var, log_slope, log_intercept, args.min_growth_rate, area.input_args)
          y_all = 10**((log_slope * all_data[covid_days]) + shifted_intercept)
          y_test = 10**((log_slope * test_set[covid_days]) + shifted_intercept)
          y_test_best = 10**((args.min_growth_rate * test_set[covid_days]) + shifted_intercept_best)
          test_rmse = get_rmse(y_test, test_set[var])
          #Plot Data and Prediction
          plt.scatter(all_data[covid_days], all_data[var], label = area.name + ':' + str(growth_rate) + ' RMSE: ' + str(test_rmse), s = args.markersize, color = col[i])
          plt.plot(all_data[covid_days], y_all, linestyle = 'solid', color = col[i])
          plt.plot(test_set[covid_days],y_test_best, color = col[i], linestyle = 'dashed')

          #plt.plot(all_df.index.values, all_df[var], label = area.name + ':' + str(growth_rate) + ' RMSE: ' + str(test_rmse), linewidth = args.linewidth)
          #plt.plot(x_predict, y_predict)
          plt.xlabel('Days since 1 ' + get_nice_var_name(var, args))
          plt.ylabel(get_nice_var_name(var, args))

      #Plot Nominal Growth Rates on COVID Days Plots
      #Set ymax from data not growth rate (bc that makes the data too zoomed out)
      _, ymax = plt.gca().get_ylim()
      if plot_type == 'raw_covid_days':
        plt.ylim(args.cv_day_thres_notscaled,ymax)
        for i in args.growth_rates:
          y = args.cv_day_thres_notscaled*(i)**x
          plt.plot(x, y, '--r', label = str(i) + 'x Daily Growth')
      elif plot_type == 'per_mil_covid_days':
        plt.ylim(1, ymax)
        for i in args.growth_rates:
          y = ((i)**x)
          plt.plot(x, y, '--r', label = str(i) + 'x Daily Growth')

      plt.yscale(scale)
      plt.legend(prop = {'size': 6})
      plt.xticks(fontsize = args.tick_font_size)
      plt.savefig(args.output_dir + '/' + var + '_' + plot_type + '_' + scale + '.png')
      plt.close('all')

def get_log_fit(train_set, covid_days, var, x):
  model = np.poly1d(np.polyfit(train_set[covid_days], np.log10(train_set[var]),1))
  log_intercept = model[0]
  log_slope = model[1]
  #Calculate fitted prediction
  prediction = 10**(model(x))
  growth_rate = round(10**log_slope,2)
  return model, log_intercept, log_slope, prediction, growth_rate

def get_rmse(model, data):
  return round(math.sqrt(mean_squared_error(model, data)))

def get_x_array_for_prediction(dataframe, args):
  x_max = len(dataframe.index.values) + args.days_of_cv_predict
  return np.linspace(0,x_max, x_max), x_max
