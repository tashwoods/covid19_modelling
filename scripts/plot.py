from imported_libraries import *

def daterange(start_date, end_date):
  for n in range(int((end_date - start_date).days)):
    yield start_date + timedelta(n)

def logistic_model(x,a,b,c):
  return c/(1+np.exp(-(x-b)/a))

def fit_logistic_all(area_object_list):
  args = area_object_list[0].input_args

  for area in area_object_list: 
    if area.name == 'South Korea':
      print('should be south korea')
      df = area.cv_days_df_not_scaled
      print(area.name)
      print(df)
      x = df['cv_days']
      y = df['deaths_per_mil']
      name = area.name
      #C = area.population
      C = 1000000

      popt, pcov = curve_fit(logistic_model, x, y, p0=[5,20, 0.002*C], bounds=[[0,5,0.00001*C],[20,50,C]])
      south_korea_a = popt[0]
      south_korea_b = popt[1]
      south_korea_c = popt[2]

      x_array = np.linspace(0,100,100)
      y_predict = logistic_model(x_array, popt[0], popt[1], popt[2])

      plt.plot(x_array,y_predict, label = name + 'fit', color = 'y')
      #plt.scatter(x,y, s = 5, color = 'y')

  col_array = plt.cm.jet(np.linspace(0,1,round(len(area_object_list)/2)+5))
  for i in range(len(area_object_list)):
    print(i)
    area = area_object_list[i]
    print('should be plotting the {}'.format(area.name))
    if area.fips not in [36005, 36061, 36081, 36085] and area.name != 'South Korea':
      df = area.cv_days_df_not_scaled
      print(df)
      x = df['cv_days']
      lastday = x.iloc[-1]
      y = df['deaths_per_mil']
      lastentry = y.iloc[-1]
      name = area.name
      #C = area.population
      C = 1000000

      popt, pcov = curve_fit(logistic_model, x, y, p0=[5,20, 0.002*C], bounds=[[0,5,0.00001*C],[20,50,C]])
      a = popt[0]
      b = popt[1]
      c = popt[2]
      x_array = np.linspace(0,100,100)
      x_sk_array = np.linspace(lastday, 100, 100 - lastday)
      y_predict = logistic_model(x_array, popt[0], popt[1], popt[2])
      y_sk_predict = logistic_model(x_sk_array, south_korea_a, south_korea_b, south_korea_c) + lastentry

      print('plotted')
      plt.plot(x_array,y_predict, label = name, color = col_array[i])
      plt.plot(x_sk_array, y_sk_predict, color = col_array[i], linestyle = 'dashed', label = 'fit' + name)
      #plt.scatter(x,y, s = 5, color = col)

  plt.yscale('log')
  plt.legend(loc = 'lower right', prop={'size':6})
  plt.xlabel('Days since ' + str(args.cv_day_thres_notscaled) + ' Death')
  plt.ylabel('Deaths per 1,000,000')
  plt.savefig(args.output_dir + '/' + name + '_skallsigmoidfit.pdf')
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
  df_list = list()
  maxes = list()
  #create area_object dataframes list per day for GIFs
  area_obj = area_obj_list[0]

  start_date_array = start_date.split('-')
  end_date_array = end_date.split('-')
  start_date = date(int(start_date_array[0]), int(start_date_array[1]), int(start_date_array[2]))
  end_date = date(int(end_date_array[0]), int(end_date_array[1]), int(end_date_array[2]))
  for day in daterange(start_date, end_date):
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
      #if (len(county.index) - 1) < day: #if there is not data for given county for selected day
      #  var_list.append(float('nan'))
      selected_row = county.loc[county['date']==str(day)]
      if selected_row.empty:
        var_list.append(float('nan'))
      else:
        var_list.append(float(selected_row[var].values))

    this_df = pd.DataFrame({'date':cv_days, 'fips': fips, var: var_list})
    df_list.append(this_df)

  #Calculate max variable value for heatmap plot
  this_variable_max_array = list()
  for i in range(len(df_list)):
    df = df_list[i]
    pd.set_option('display.max_rows', None)
    this_variable_max_array.append(np.nanpercentile(df[var], 99.5))
    if i == len(df_list) -1:
      print(df[var])
    print('99.5')
    print(np.nanpercentile(df[var], 99.7))
    print('99.9')
    print(np.nanpercentile(df[var], 99.9))
  vmax = max(this_variable_max_array)

  #Create jpegs for each day to be combined later into gifs
  filenames = []
  for i in range(len(df_list)):
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
    my_cmap.set_over('black')
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.rc('font', size = 40)

    ax = combined_geo.plot(column = var, figsize=(60,40), cmap = my_cmap, vmin=vmin, vmax=vmax)
    plt.title(str(df.iloc[0]['date']))
    plt.axis('off')

    # add colorbar
    fig = ax.get_figure()
    cax = fig.add_axes([0.1, 0.2, 0.8, 0.01])
    fig.colorbar(sm, orientation = 'horizontal', aspect = 80, label = get_nice_var_name(var), cax = cax, extend='both')

    filename = args.output_dir + '/counties_' + var + '_' + str(i) + '.jpg'
    filenames.append(filename)
    #plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
  make_gif_command = 'convert -delay 100 '
  for ifile in filenames:
    make_gif_command += ifile + ' '
  make_gif_command += args.output_dir + '/' + var + '.gif'
  os.system(make_gif_command)

def make_gif_cv_days(area_obj_list, dataframe_name, var, ndays, args):
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
    this_variable_max_array.append(np.nanpercentile(df[var], 95))
  vmax = max(this_variable_max_array)

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
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.rc('font', size = 40)

    ax = combined_geo.plot(column = var, figsize=(60,40), cmap = my_cmap, vmin=vmin, vmax=vmax)
    plt.title('CV Outbreak Day ' + str(i))
    plt.axis('off')

    # add colorbar
    fig = ax.get_figure()
    cax = fig.add_axes([0.1, 0.2, 0.8, 0.01])
    fig.colorbar(sm, orientation = 'horizontal', aspect = 80, label = get_nice_var_name(var), cax = cax)

    filename = args.output_dir + '/counties_' + var + '_' + str(i) + '.jpg'
    filenames.append(filename)
    #plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
  make_gif_command = 'convert -delay 100 '
  for ifile in filenames:
    make_gif_command += ifile + ' '
  make_gif_command += args.output_dir + '/' + var + '.gif'
  os.system(make_gif_command)


def get_nice_var_name(var):
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
    return 'Deaths (per 1,000,000)'

def get_shifted_prediction(area_df, var, slope, intercept, best_growth_rate, args):
  last_x = len(area_df.index.values) - 1
  last_y = np.log10(area_df[var].iloc[-1])
  shifted_intercept = last_y - (slope * last_x)
  best_shifted_intercept = last_y - (best_growth_rate * last_x)
  last_predict_day = last_x + args.days_of_cv_predict
  x = np.linspace(last_x,last_predict_day, last_predict_day - last_x + 1)
  y = 10**((slope * x) + shifted_intercept)
  y_best = 10**((best_growth_rate *x) + best_shifted_intercept)
  return x, y, y_best

def get_lives_saved_bar_chart(x_predict, y_predict, y_best, name, args, savename):
  plt.close('all')

  lives_saved = y_predict - y_best
  print('in get lives saved')
  print('prediction')
  print(y_predict)
  print('best')
  print(y_best)
  print('saved')
  print(lives_saved)
  print('x')
  print(x_predict)
  plt.yscale('linear')
  x_predict = [x - x_predict[0] for x in x_predict]
  plt.xlim(0,args.days_of_cv_predict + 1)
  plt.bar(x_predict, lives_saved)
  plt.xlabel('Days From Now')
  plt.ylabel('Total Lives Saved')
  total_saved = round(lives_saved[-1],1)
  if savename == 'Individual':
    plt.title(name + ': Total Lives One Person Could Save')
  else:
    plt.title(name + ': Total Lives Saved Days from Now')
  plt.text(0.02, 0.9, 'Current Projected Deaths: ' + str(round(y_predict[-1],1)), transform = plt.gca().transAxes)
  plt.text(0.02, 0.8, 'Best Case Projected Deaths: ' + str(round(y_best[-1],1)), transform = plt.gca().transAxes)
  plt.text(0.02, 0.7, 'Lives That Could Be Saved: ' + str(total_saved), transform = plt.gca().transAxes)
  plt.savefig(args.output_dir + '/' + name + 'livessaved' + savename + '.png')
  plt.close('all')

def plot(area_objects_list, args, plot_type, animate):
  #Make Plot Line Colors Pretty 
  col = plt.cm.jet(np.linspace(0,1,round(len(area_objects_list)/2)+5))
  if plot_type == 'unmodified_covid_days' or plot_type == 'per_mil_covid_days':
    line_cycler = cycler('color', col,) * cycler('linestyle', ['-', ':'])
  else:
    line_cycler = cycler('color', col,)
  plt.rc('axes', prop_cycle = line_cycler)
  #Plot time series for time_series_variables
  for var in args.time_series_variables:
    for i in range(len(area_objects_list)):
      area = area_objects_list[i]
      nyc_fips_to_skip = [36047, 36061, 36081, 36085]
      if area.fips in nyc_fips_to_skip: #do not plot NYC counties individually since NYT dataset combined them
        continue
      area_df = area.df
      if plot_type == 'unmodified':
        plt.plot(area_df[args.date_name], area_df[var], label = area.name, linewidth = args.linewidth)
        plt.xlabel(args.n_date_name)
        plt.ylabel(get_nice_var_name(var))
      elif plot_type == 'per_mil':
        plt.plot(area_df[args.date_name], (area_df[var].div(area.population)*args.cv_day_thres), label = area.name, linewidth = args.linewidth)
        plt.xlabel(args.n_date_name)
        plt.ylabel(get_nice_var_name(var) + ' per ' + get_nice_var_name(args.cv_day_thres))
      elif plot_type == 'bar_unmodified_covid_days':
        area_df = area.cv_days_df_not_scaled
        x_max = len(area_df) + args.days_of_cv_predict
        print(area.name)
        print('xmax: {}'.format(x_max))
        x = np.linspace(0,x_max, x_max)
        plt.xlim(0,30)
        plt.ylim(1,10000)
        if len(area_df.index) != 0:
          model = np.poly1d(np.polyfit(area_df.index.values, np.log10(area_df[var]),1))
          slope = round(10**model[1],2) #this is written for log scale, should make it more general
          intercept = model[0]
          prediction = 10**(model(x))
          x_predict, y_predict, y_best = get_shifted_prediction(area_df, var, model[1], model[0], args.min_growth_rate, area.input_args)
          print('prediction')
          print(y_predict)
          print('best')
          print(y_best)
          get_lives_saved_bar_chart(x_predict, y_predict, y_best, area.name, area.input_args, 'All')
          #individual impact
          current_indiv_slope = (model[1]/area.population)
          improved_slope = model[1] - (current_indiv_slope - args.min_indiv_growth_rate)
          x_predict, y_predict_indiv, y_best_indiv = get_shifted_prediction(area_df, var, model[1], model[0], improved_slope, area.input_args)
          #get_lives_saved_bar_chart(x_predict, y_predict_indiv, y_best_indiv, area.name, area.input_args, 'Individual')
          plt.plot(area_df.index.values, area_df[var], label = area.name + ':' + str(slope), linewidth = args.linewidth)
          plt.plot(x_predict,y_predict)
          plt.plot(x_predict, y_best, label = area.name + ' best')
        plt.xlabel('Days since ' + str(args.cv_day_thres_notscaled) + ' Deaths')
        plt.ylabel(get_nice_var_name(var))


      elif plot_type == 'unmodified_covid_days':
        x_max = len(area_df.index.values) + args.days_of_cv_predict
        print('xmax: {}'.format(x_max))
        x = np.linspace(0,x_max, x_max)
        plt.xlim(0,30)
        plt.ylim(1,10000)
        start_date = get_first_cv_day(area, 'notscaled')
        if start_date != -1: #if there has been at least one CV day plot that area
          area_df = area_df[start_date:].reset_index()
          model = np.poly1d(np.polyfit(area_df.index.values, np.log10(area_df[var]),1))
          slope = round(10**model[1],2) #this is written for log scale, should make it more general
          intercept = model[0]
          prediction = 10**(model(x))
          x_predict, y_predict, y_best = get_shifted_prediction(area_df, var, model[1], model[0], args.min_growth_rate, area.input_args)
          get_lives_saved_bar_chart(x_predict, y_predict, y_best, area.name, area.input_args, 'All')
          #individual impact
          current_indiv_slope = (model[1]/area.population)
          improved_slope = model[1] - (current_indiv_slope - args.min_indiv_growth_rate)
          x_predict, y_predict_indiv, y_best_indiv = get_shifted_prediction(area_df, var, model[1], model[0], improved_slope, area.input_args)
          get_lives_saved_bar_chart(x_predict, y_predict_indiv, y_best_indiv, area.name, area.input_args, 'Individual')
          plt.plot(area_df.index.values, area_df[var], label = area.name + ':' + str(slope), linewidth = args.linewidth)
          plt.plot(x_predict,y_predict)
        plt.xlabel('Days since ' + str(args.cv_day_thres_notscaled) + ' Deaths')
        plt.ylabel(get_nice_var_name(var))
      elif plot_type == 'per_mil_covid_days':
        start_date = get_first_cv_day(area, 'scaled')
        x_max = len(area_df.index.values) + args.days_of_cv_predict
        x = np.linspace(0,x_max, x_max)
        plt.xlim(0,30)
        plt.ylim(1,5000)
        if start_date != -1:
          cv_days_df = pd.DataFrame(list(zip(area_df.index.values, area_df[var].div(area.population)*args.cv_day_thres)), columns = ['cv_days', 'deaths_per_mil'])
          area_df = area_df[start_date:].reset_index()
          if(len(area_df.index.values) < 2):
            continue
          model = np.poly1d(np.polyfit(area_df.index.values, np.log10(area_df[var].div(area.population)*args.cv_day_thres),1))
          slope = round(10**model[1],2)
          intercept = model[0]
          x_predict, y_predict, y_best = get_shifted_prediction(area_df, var, model[1], model[0], args.min_growth_rate, area.input_args)
          plt.plot(area_df.index.values, area_df[var].div(area.population)*args.cv_day_thres, label = area.name + ':' + str(slope), linewidth = args.linewidth)
          #if area.name != 'China' and area.name != 'Hubei':
          plt.plot(x_predict, (y_predict/area.population)*args.cv_day_thres)
        plt.xlabel('Days since 1death/' + get_nice_var_name(args.cv_day_thres) + ' people')
        plt.ylabel(get_nice_var_name(var) + ' per ' + get_nice_var_name(args.cv_day_thres))

          

    #Plot Nominal Growth Rates on COVID Days Plots
    if plot_type == 'unmodified_covid_days':
      for i in args.growth_rates:
        y = args.cv_day_thres_notscaled*(i)**x
        plt.plot(x, y, '--r', label = str(i) + 'x Daily Growth')
    if plot_type == 'per_mil_covid_days':
      for i in args.growth_rates:
        y = ((i)**x)
        plt.plot(x, y, '--r', label = str(i) + 'x Daily Growth')

    plt.yscale(args.plot_y_scale)
    plt.legend(prop = {'size': 6})
    plt.xticks(fontsize = args.tick_font_size)
    plt.savefig(args.output_dir + '/' + var + '_' + plot_type + '.png')
    plt.close('all')
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


