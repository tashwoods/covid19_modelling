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

def get_shifted_prediction(area_df, var, slope, intercept, args):
  last_x = len(area_df.index.values) - 1
  last_y = np.log10(area_df[var].iloc[-1])
  shifted_intercept = last_y - (slope * last_x)
  best_shifted_intercept = last_y - (args.min_growth_rate * last_x)
  last_predict_day = last_x + args.days_of_cv_predict
  x = np.linspace(last_x,last_predict_day, last_predict_day - last_x + 1)
  y = 10**((slope * x) + shifted_intercept)
  y_best = 10**((args.min_growth_rate *x) + best_shifted_intercept)
  return x, y, y_best

def get_lives_saved_bar_chart(x_predict, y_predict, y_best, name, args):
  plt.close('all')
  lives_saved = y_predict - y_best
  plt.yscale('linear')
  x_predict = [x - x_predict[0] for x in x_predict]
  plt.xlim(0,args.days_of_cv_predict + 1)
  plt.bar(x_predict, lives_saved)
  plt.xlabel('Days From Now')
  plt.ylabel('Total Lives Saved')
  total_saved = round(lives_saved[-1],0)
  plt.title(name + ': Total Lives Saved Days from Now')
  plt.text(0.5, 2800, 'Current Projected Deaths: ' + str(round(y_predict[-1],0)))
  plt.text(0.5, 2600, 'Best Case Projected Deaths: ' + str(round(y_best[-1],0)))
  plt.text(0.5, 2400, 'Lives That Could Be Saved: ' + str(total_saved))
  plt.savefig(args.output_dir + '/' + name + 'livessaved.pdf')
  plt.close('all')

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
        plt.plot(area_df[args.date_name], area_df[var], label = area.name, linewidth = args.linewidth)
        plt.xlabel(args.n_date_name)
        plt.ylabel(get_nice_var_name(var))
      elif plot_type == 'per_mil':
        plt.plot(area_df[args.date_name], (area_df[var].div(area.population)*args.cv_day_thres), label = area.name, linewidth = args.linewidth)
        plt.xlabel(args.n_date_name)
        plt.ylabel(get_nice_var_name(var) + ' per ' + get_nice_var_name(args.cv_day_thres))
      elif plot_type == 'unmodified_covid_days':
        x_max = len(area_df.index.values) + args.days_of_cv_predict
        x = np.linspace(0,x_max, x_max)
        plt.xlim(0,30)
        plt.ylim(10,100000)
        start_date = get_first_cv_day(area, 'notscaled')
        if start_date != -1: #if there has been at least one CV day plot that area
          area_df = area_df[start_date:].reset_index()
          model = np.poly1d(np.polyfit(area_df.index.values, np.log10(area_df[var]),1))
          slope = round(10**model[1],2) #this is written for log scale, should make it more general
          intercept = model[0]
          prediction = 10**(model(x))
          x_predict, y_predict, y_best = get_shifted_prediction(area_df, var, model[1], model[0], area.input_args)
          #get_lives_saved_bar_chart(x_predict, y_predict, y_best, area.name, area.input_args)
          if var == 'total_deaths':
            plt.plot(area_df.index.values, area_df[var], label = area.name + ':' + str(slope), linewidth = args.linewidth)
            if area.name != 'China' and area.name != 'Hubei': #China and Hubei levelled, don't care to plot their trends
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
          area_df = area_df[start_date:].reset_index()
          model = np.poly1d(np.polyfit(area_df.index.values, np.log10(area_df[var].div(area.population)*args.cv_day_thres),1))
          slope = round(10**model[1],2)
          intercept = model[0]
          x_predict, y_predict, y_best = get_shifted_prediction(area_df, var, model[1], model[0], area.input_args)
          if var == 'total_deaths':
            plt.plot(area_df.index.values, area_df[var].div(area.population)*args.cv_day_thres, label = area.name + ':' + str(slope), linewidth = args.linewidth)
            if area.name != 'China' and area.name != 'Hubei':
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


