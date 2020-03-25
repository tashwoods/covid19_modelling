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

def plot(area_objects_list, args, plot_type):
  #Make Plot Line Colors Pretty 
  col = plt.cm.twilight_shifted(np.linspace(0,1,round(len(area_objects_list)/2)+2))
  line_cycler = cycler('color', col,) * cycler('linestyle', ['-', ':'])
  plt.rc('axes', prop_cycle = line_cycler)
  for var in args.time_series_variables:
    for i in range(len(area_objects_list)):
      area = area_objects_list[i]
      area_df = area.df
      if plot_type == 'unmodified':
        plt.title(get_nice_var_name(var) + ' vs. ' + args.n_date_name)
        plt.plot(area_df[args.date_name], area_df[var], label = area.name)
        plt.xlabel(args.n_date_name)
        plt.ylabel(get_nice_var_name(var))
      elif plot_type == 'per_capita':
        plt.title(get_nice_var_name(var) + ' per Capita vs. ' + args.n_date_name)
        plt.plot(area_df[args.date_name], 100*(area_df[var]/area.population), label = area.name)
        plt.xlabel(args.n_date_name)
        plt.ylabel(get_nice_var_name(var) + ' per Capita')
      elif plot_type == 'unmodified_covid_days':
        plt.title(get_nice_var_name(var) + ' vs. ' + covid_outbreak_days_name)
        start_date = get_first_cv_day(area)
        if start_date != -1:
          area_df = area_df[start_date:].reset_index()
          plt.plot(area_df.index.values, area_df[var], label = area.name)
        plt.xlabel(covid_outbreak_days_name)
        plt.ylabel(get_nice_var_name(var))
      elif plot_type == 'per_capita_covid_days':
        plt.title(get_nice_var_name(var) + ' vs. ' + covid_outbreak_days_name)
        start_date = get_first_cv_day(area)
        if start_date != -1:
          area_df = area_df[start_date:].reset_index()
          plt.plot(area_df.index.values, 100*(area_df[var]/area.population), label = area.name)
        plt.xlabel(covid_outbreak_days_name)
        plt.ylabel(get_nice_var_name(var) + ' per Capita')

    plt.yscale(args.plot_y_scale)
    plt.legend()
    plt.xticks(fontsize = args.tick_font_size)
    plt.savefig(args.output_dir + '/' + var + '_' + plot_type + '.png')
    plt.close('all')

def get_first_cv_day(country_object):
  country = country_object.df
  print(country)
  population = country_object.population
  cv_thres = population * (1/args.cv_day_thres) #will give first day that one in cv_day_thres people in country affected via predict var
  #cv_thres = 10
  print(country_object.name)
  print(cv_thres)
  truncated_list = country[country[args.predict_var] > cv_thres]
  if len(truncated_list) > 0:
    first_cv_day = truncated_list.iloc[0].name
    first_index = country.index.get_loc(first_cv_day)
    print(first_index)
    return(first_index)
  else:
    print('trunchated list empty')
    return(-1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for process.py')
  #Input/Output Information
  parser.add_argument('-f', '--input_file', type = str, dest = 'input_file', default = '../data/full_data.csv',  help = 'text file with input file names of countries')
  parser.add_argument('-state_file', '--state_file', type = str, dest = 'state_file', default = '../data/us_states_covid19_daily.csv', help = 'file with us state level data')
  parser.add_argument('-c', '--countries_file', type = str, dest = 'countries_file', default = 'countries.txt', help = 'Name of text file with names of countries to process')
  parser.add_argument('-o', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'name of output_dir')
  #Variable name information
  parser.add_argument('-plot_time_series', '--plot_time_series', type = int, dest = 'plot_time_series', default = 1, help = 'set to one to plot time series, zero to not')
  parser.add_argument('-time_series_variables', '--time_series_variables', type = list, dest = 'time_series_variables', default = ['new_cases', 'new_deaths', 'total_cases', 'total_deaths'], help = 'list of variables to plot in time series')
  parser.add_argument('-country_var_name', '--country_var_name', type = str, dest = 'country_var_name', default = 'location', help = 'country variable name in input dataset')
  parser.add_argument('-date_name', '--date_name', type = str, dest = 'date_name', default = 'date', help = 'name of date columns in input file')
  parser.add_argument('-n_date_name', '-n_date_name', type = str, dest = 'n_date_name', default = 'Date', help = 'name of date variable for plots')
  parser.add_argument('-name_total_cases', '--name_total_cases', type = str, dest = 'name_total_cases', default = 'total_cases', help = 'name of total case variable in country level file')
  parser.add_argument('-name_total_deaths', '--name_total_deaths', type = str, dest = 'name_total_deaths', default = 'total_deaths', help = 'name of total deaths variable in country level file')
  parser.add_argument('-name_new_cases', '--name_new_cases', type = str, dest = 'name_new_cases', default = 'new_cases', help = 'name of variable in country level file for new cases per day')
  parser.add_argument('-name_new_deaths', '--name_new_deaths', type = str, dest = 'name_new_deaths', default = 'new_deaths', help = 'name of variable in country level file for new deaths per day')
  parser.add_argument('-predict_var', '--predict_var', type = str, dest = 'predict_var', default = 'total_deaths', help = 'number of variable used to determine where to start counting cv days, should be whichever variable in your dataset you believe is the most accurate')
  #Analysis Variables
  parser.add_argument('-cv_day_thres', '--cv_day_thres', type = int, dest = 'cv_day_thres', default = 10000000, help = 'total number of cases to consider it the first day of cv19')
  #Aesthetics : )
  parser.add_argument('-tick_font_size', '--tick_font_size', type = int, dest = 'tick_font_size', default = 8, help = 'size of tick labels in plots')
  parser.add_argument('-plot_y_scale', '--plot_y_scale', type = str, dest = 'plot_y_scale', default = 'log', help = 'scale for y axis, set to linear, log, etc')

  args = parser.parse_args()

  covid_outbreak_days_name = 'COVID-19 Outbreak Days'

  #Make output data directory and get input data
  variables = [args.date_name, args.name_total_cases, args.name_total_deaths, args.name_new_cases, args.name_new_deaths, args.country_var_name]
  make_output_dir(args.output_dir)
  #Obtain country level dataframe
  full_dataframe = pd.read_csv(args.input_file, parse_dates = [args.date_name])
  #Obtain state level dataframe
  states = ['WA', 'CA', 'AL', 'CO', 'NY', 'FL']
  populations = [7800000, 39940000, 4910000, 5800000, 20000000, 22000000]
  state_var = ['new_deaths', 'new_cases', 'total_new_tests', 'total_deaths']
  state_dataframe = pd.read_csv(args.state_file, parse_dates = [args.date_name])
  #Process state-level dataframe to match formatting of country level dataframe
  state_dataframe.fillna(0, inplace=True)
  state_dataframe = state_dataframe.rename(columns = {'state': 'location', 'positive': 'new_cases', 'death': 'new_deaths', 'total': 'total_new_tests'})

  #setup census object to obtain US state populations #Figure out later maybe Natasha##
  #c = Census("42ec78e46c9d0bf074a6332b41bec791e3a95d14", year= 2020)
  #pop = c.acs5.get('POP', geo='state:{Alabama}')

  #Format data
  area_objects_list = list()

  #Process Hubei Data
  hubei_pop = 11000000
  hubei_df = pd.read_csv('../data/covid_19_data_hubei.csv', parse_dates = ['ObservationDate'])
  hubei_df = hubei_df.rename(columns = {'ObservationDate': 'date', 'Province/State': 'location', 'Confirmed': 'total_cases', 'Deaths': 'total_deaths'})
  hubei_df[args.name_new_cases] = hubei_df[args.name_total_cases].diff().fillna(0)
  hubei_df[args.name_new_deaths] = hubei_df[args.name_total_deaths].diff().fillna(0)
  area_object = area_corona_class('Hubei', hubei_df, hubei_pop,args)
  area_objects_list.append(area_object)

  #Process make country dataframe and objects
  countries_file = open(args.countries_file, "r")
  for country in countries_file:
    if len(country.strip()) > 0:
      country = country.rstrip()
      this_country_df = full_dataframe[full_dataframe[args.country_var_name].str.match(country)]
      population = CountryInfo(country).population()
      area_object = area_corona_class(country, this_country_df, population,args)
      area_objects_list.append(area_object)

      if country == 'Chinaz':
        this_country_df = full_dataframe[full_dataframe[args.country_var_name].str.match(country)]
        for var in args.time_series_variables:
          scale_china = 3000
          this_country_df[var] = scale_china * this_country_df[var]
        population = CountryInfo(country).population()
        name = country + 'x' + str(scale_china)
        area_object = area_corona_class(name, this_country_df, population,args)
        area_objects_list.append(area_object)

  #Split state dataframe by state, and compute and append cumulative variables
  for state, population in zip(states,populations) :
    if len(state.strip()) > 0:
      state = state.rstrip()
      this_state_df = state_dataframe[state_dataframe[args.country_var_name].str.match(state)]
      this_state_df = this_state_df.sort_values(by=[args.date_name])
      this_state_df[args.name_total_cases] = this_state_df['new_cases'].cumsum()
      this_state_df[args.name_total_deaths] = this_state_df['new_deaths'].cumsum()
      this_state_df['total_tests'] = this_state_df['total_new_tests'].cumsum()
      area_object = area_corona_class(state, this_state_df, population,args)
      area_objects_list.append(area_object)

  if(args.plot_time_series == 1):
    plot(area_objects_list, args, 'unmodified')
    plot(area_objects_list, args, 'per_capita')
    plot(area_objects_list, args, 'unmodified_covid_days')
    plot(area_objects_list, args, 'per_capita_covid_days')
