from imported_libraries import *

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

def plot(area_objects_list, args, plot_type):
  #Make Plot Line Colors Pretty 
  col = plt.cm.jet(np.linspace(0,1,round(len(area_objects_list)/2)+2))
  line_cycler = cycler('color', col,) * cycler('linestyle', ['-', ':'])
  plt.rc('axes', prop_cycle = line_cycler)
  #Plot time series for time_series_variables
  for var in args.time_series_variables:
    for i in range(len(area_objects_list)):
      area = area_objects_list[i]
      area_df = area.df
      if plot_type == 'unmodified':
        plt.plot(area_df[args.date_name], area_df[var], label = area.name)
        plt.xlabel(args.n_date_name)
        plt.ylabel(get_nice_var_name(var))
      elif plot_type == 'per_mil':
        plt.plot(area_df[args.date_name], (area_df[var].div(area.population)*args.cv_day_thres), label = area.name)
        plt.xlabel(args.n_date_name)
        plt.ylabel(get_nice_var_name(var) + ' per ' + get_nice_var_name(args.cv_day_thres))
      elif plot_type == 'unmodified_covid_days':
        x_max = len(area_df.index.values)+10
        x = np.linspace(0,x_max, x_max)
        plt.xlim(0,40)
        plt.ylim(10,100000)
        start_date = get_first_cv_day(area, 'notscaled')
        if start_date != -1:
          print(area.name)
          area_df = area_df[start_date:].reset_index()
          print(np.polyfit(area_df.index.values, np.log10(area_df[var]),1))
          model = np.poly1d(np.polyfit(area_df.index.values[:-3], np.log10(area_df[var][:-3]),1))
          slope = round(10**model[1],2)
          intercept = model[0]
          prediction = 10**(model(x))
          prediction20 = 10**(model(30))
          print('day 30 guess:{} '.format(prediction20))
          #for x,y in zip(x,prediction):
          #  print('day: {} predict: {}'.format(x,y))
          if var == 'total_deaths':
            plt.plot(area_df.index.values, area_df[var], label = area.name + ':' + str(slope))
            if area.name != 'China' and area.name != 'Hubei':
              plt.plot(x,prediction)#,label = area.name + ' Fit')
        plt.xlabel('Days since 10 deaths')
        plt.ylabel(get_nice_var_name(var))

      elif plot_type == 'per_mil_covid_days':
        start_date = get_first_cv_day(area, 'scaled')
        x_max = len(area_df.index.values)+10
        x = np.linspace(0,x_max, x_max)
        if start_date != -1:
          area_df = area_df[start_date:].reset_index()
          model = np.poly1d(np.polyfit(area_df.index.values, np.log10(area_df[var].div(area.population)*args.cv_day_thres),1))
          slope = round(10**model[1],2)
          intercept = model[0]
          prediction = 10**(model(x))
          prediction20 = 10**(model(20))
          print('prediction day 20:{}'.format(prediction20))

          if var == 'total_deaths':
            print(area.name)
            print('slope:{} inter:{}'.format(slope, intercept))
            plt.plot(area_df.index.values, area_df[var].div(area.population)*args.cv_day_thres, label = area.name + ':' + str(slope))
            if area.name != 'China' and area.name != 'Hubei':
              plt.plot(x,prediction)#,label = area.name + ' Fit')

        plt.xlabel('Days since 1death/' + get_nice_var_name(args.cv_day_thres) + ' people')
        plt.ylabel(get_nice_var_name(var) + ' per ' + get_nice_var_name(args.cv_day_thres))
        plt.xlim(1,30)
        plt.ylim(1,1000)


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


