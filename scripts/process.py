from imported_libraries import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for process.py')
  #Corona Data 
  parser.add_argument('-country_data_file', '--country_data_file', type = str, dest = 'country_data_file', default = '../data/full_data.csv',  help = 'text file with input file names of countries') 
  parser.add_argument('-state_data_file', '--state_data_file', type = str, dest = 'state_data_file', default = '../data/us-states.csv', help = 'file with us state level data')
  parser.add_argument('-county_data_file', '--county_data_file', type = str, dest = 'county_data_file', default = '../data/us-counties.csv', help = 'file with county level data')
  parser.add_argument('-selected_countries_file', '--selected_countries_file', type = str, dest = 'selected_countries_file', default = 'selected_areas/countries.txt', help = 'Name of text file with names of countries to process')
  parser.add_argument('-selected_states_file', '--selected_states_file', type = str, dest = 'selected_states_file', default = 'selected_areas/states.txt', help = 'Name of text file with name of US states to process')
  parser.add_argument('-selected_counties', '--selected_counties_file', type = str, dest = 'selected_counties_file', default = 'selected_areas/counties.txt', help = 'text file with US counties to use')
  parser.add_argument('-time_series_variables', '--time_series_variables', type = list, dest = 'time_series_variables', default = ['total_deaths', 'total_cases'], help = 'list of variables to plot in time series')
  parser.add_argument('-country_var_name', '--country_var_name', type = str, dest = 'country_var_name', default = 'location', help = 'country variable name in input dataset')
  parser.add_argument('-date_name', '--date_name', type = str, dest = 'date_name', default = 'date', help = 'name of date columns in input file')
  parser.add_argument('-n_date_name', '-n_date_name', type = str, dest = 'n_date_name', default = 'Date', help = 'name of date variable for plots')
  parser.add_argument('-name_total_cases', '--name_total_cases', type = str, dest = 'name_total_cases', default = 'total_cases', help = 'name of total case variable in country level file')
  parser.add_argument('-name_total_deaths', '--name_total_deaths', type = str, dest = 'name_total_deaths', default = 'total_deaths', help = 'name of total deaths variable in country level file')
  parser.add_argument('-name_new_cases', '--name_new_cases', type = str, dest = 'name_new_cases', default = 'new_cases', help = 'name of variable in country level file for new cases per day')
  parser.add_argument('-name_new_deaths', '--name_new_deaths', type = str, dest = 'name_new_deaths', default = 'new_deaths', help = 'name of variable in country level file for new deaths per day')
  parser.add_argument('-predict_var', '--predict_var', type = str, dest = 'predict_var', default = 'total_deaths', help = 'number of variable used to determine where to start counting cv days, should be whichever variable in your dataset you believe is the most accurate')
  #US State Population Info
  parser.add_argument('-us_state_population_file', '--us_state_population_file', type = str, dest = 'us_state_population_file', default = '../data/nst-est2019-alldata.csv', help = 'file with us population data')
  parser.add_argument('-us_county_pop_file', '--us_county_pop_file', type = str, dest = 'us_county_pop_file', default = '../data/county_info_not_tracked/cc-est2018-alldata.csv', help = 'file with US county population info')
  parser.add_argument('-state_pop_region_name', '--state_pop_region_name', type = str, dest = 'state_pop_region_name', default = 'NAME', help = 'name of state variable in population file')
  parser.add_argument('-state_pop_var_name', '--state_pop_var_name', type = str, dest = 'state_pop_var_name', default = 'POPESTIMATE2019', help = 'name of population variable to use (e.g. POPESTIMATE2016-2019')
  parser.add_argument('-county_pop_region_name', '--county_pop_region_name', type = str, dest = 'county_pop_region_name', default = 'CTYNAME', help = 'name of county variable in county_pop_file')
  parser.add_argument('-county_pop_var_name', '--county_pop_var_name', type = str, dest = 'county_pop_var_name', default = 'TOT_POP', help = 'name of total population variable in county_pop_file')
  parser.add_argument('-county_pop_year', '--county_pop_year', type = int, dest = 'county_pop_year', default = 11, help = 'year to use')
  parser.add_argument('-county_pop_age_group', '--county_pop_age_group', type = int, dest = 'county_pop_age_group', default = 0, help = 'age group to use for county population')
  parser.add_argument('-county_var_name', '--county_var_name', type = str, dest = 'county_var_name', default = 'county', help = 'name of county variable in county data file')

  #Specify output and its final location
  parser.add_argument('-output_dir', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'name of output_dir')
  parser.add_argument('-plot_time_series', '--plot_time_series', type = int, dest = 'plot_time_series', default = 1, help = 'set to one to plot time series, zero to not')
  parser.add_argument('-growth_rates', '--growth_rates', type = list, dest = 'growth_rates', default = [1.35], help = 'list of growth rates to plot')
  #Analysis Variables
  parser.add_argument('-cv_day_thres', '--cv_day_thres', type = int, dest = 'cv_day_thres', default = 1000000, help = 'total number of cases to consider it the first day of cv19')
  parser.add_argument('-cv_day_thres_notscaled', '--cv_day_thres_notscaled', type = int, dest = 'cv_day_thres_notscaled', default = 5, help = 'minimum number of deaths to start counting CV days from for unscaled data')

  #Aesthetics : )
  parser.add_argument('-tick_font_size', '--tick_font_size', type = int, dest = 'tick_font_size', default = 8, help = 'size of tick labels in plots')
  parser.add_argument('-plot_y_scale', '--plot_y_scale', type = str, dest = 'plot_y_scale', default = 'log', help = 'scale for y axis, set to linear, log, etc')
  parser.add_argument('-linewidth', '--linewidth', type = int, dest = 'linewidth', default = 1, help = 'width of lines for plots')
  parser.add_argument('-days_of_cv_predict', '--days_of_cv_predict', type = int, dest = 'days_of_cv_predict', default = 30, help = 'number of days past last date in dataset to predict cv trends')
  parser.add_argument('-min_growth_rate', '--min_growth_rate', type = float, dest = 'min_growth_rate', default = 0.03927, help = 'min growth rate to compare to') #0.0357 absolute best
  parser.add_argument('-min_indiv_growth_rate', '--min_indiv_growth_rate', type = float, dest = 'min_indiv_growth_rate', default = 7.6595717E-10, help = 'minimum individual contribution to growth rate')
  args = parser.parse_args()

  #Internally Defined Variables
  covid_outbreak_days_name = 'COVID-19 Outbreak Days'
  #Make output folders
  make_output_dir(args.output_dir)

  #Collect Output Data
  area_objects_list = list()
  #Obtain country level dataframe
  selected_countries = open(args.selected_countries_file, 'r')
  country_dataframe = pd.read_csv(args.country_data_file, parse_dates = [args.date_name])
  for country in selected_countries:
    if len(country.strip()) > 0:
      country = country.rstrip()
      this_country_df = country_dataframe[country_dataframe[args.country_var_name].str.match(country)]
      population = CountryInfo(country).population()
      area_object = area_corona_class(country, this_country_df, population,args)
      area_objects_list.append(area_object)
    if country == 'Chinaz': #Scale China's dataset to see if their reporting seems accurate
      this_country_df = full_dataframe[full_dataframe[args.country_var_name].str.match(country)]
      for var in args.time_series_variables:
        scale_china = 3000
        this_country_df[var] = scale_china * this_country_df[var]
      population = CountryInfo(country).population()
      name = country + 'x' + str(scale_china)
      area_object = area_corona_class(name, this_country_df, population,args)
      area_objects_list.append(area_object)

  #Obtain state level dataframe
  state_population_df = pd.read_csv(args.us_state_population_file)
  selected_states = open(args.selected_states_file, "r")
  state_dataframe = pd.read_csv(args.state_data_file, parse_dates = [args.date_name])
  #state_dataframe.fillna(0, inplace=True)
  state_dataframe = state_dataframe.rename(columns = {'state': 'location', 'cases': 'total_cases', 'deaths': 'total_deaths'})
  #Split state dataframe by state, and compute and append cumulative variables
  for state in selected_states :
    if len(state.strip()) > 0:
      state = state.rstrip()
      population = get_population(state, state_population_df, args.state_pop_region_name, args.state_pop_var_name)
      this_state_df = state_dataframe[state_dataframe[args.country_var_name].str.match(state)]
      this_state_df = this_state_df.sort_values(by=[args.date_name])
      #this_state_df[args.name_total_cases] = this_state_df['new_cases'].cumsum()
      this_state_df['new_cases'] = this_state_df['total_cases'].diff()
      this_state_df['new_deaths'] = this_state_df['total_deaths'].diff()
      this_state_df.fillna(0, inplace = True)
      print(this_state_df)
      area_object = area_corona_class(state, this_state_df, population,args)
      area_objects_list.append(area_object)

  #Obtain US County level dataframe
  print('HERE')
  print(args.us_county_pop_file)
  county_population_df = pd.read_csv(args.us_county_pop_file, encoding='latin-1')
  county_population_df = county_population_df.loc[county_population_df['YEAR'] == args.county_pop_year]
  county_population_df = county_population_df.loc[county_population_df['AGEGRP'] == args.county_pop_age_group]
  print(county_population_df)
  selected_counties = open(args.selected_counties_file, "r")
  counties_dataframe = pd.read_csv(args.county_data_file, parse_dates = [args.date_name])
  print(counties_dataframe)
  #state_dataframe.fillna(0, inplace=True)
  counties_dataframe = counties_dataframe.rename(columns = {'state': 'location', 'cases': 'total_cases', 'deaths': 'total_deaths'})
  #Split state dataframe by state, and compute and append cumulative variables
  for county in selected_counties:
    if len(county.strip()) > 0:
      county = county.rstrip()
      population = get_population(county, county_population_df, args.county_pop_region_name, args.county_pop_var_name)
      county = county.rsplit(' ', 1)[0]
      print(county)
      this_county_df = counties_dataframe[counties_dataframe[args.county_var_name].str.match(county)]
      this_county_df = this_county_df.sort_values(by=[args.date_name])
      #this_state_df[args.name_total_cases] = this_state_df['new_cases'].cumsum()
      this_county_df['new_cases'] = this_county_df['total_cases'].diff()
      this_county_df['new_deaths'] = this_county_df['total_deaths'].diff()
      this_county_df.fillna(0, inplace = True)
      print(this_county_df)
      area_object = area_corona_class(county, this_county_df, population,args)
      area_objects_list.append(area_object)

  #Process Hubei Data
  hubei_pop = 11000000
  hubei_df = pd.read_csv('../data/covid_19_data_hubei.csv', parse_dates = ['ObservationDate'])
  hubei_df = hubei_df.rename(columns = {'ObservationDate': 'date', 'Province/State': 'location', 'Confirmed': 'total_cases', 'Deaths': 'total_deaths'})
  hubei_df[args.name_new_cases] = hubei_df[args.name_total_cases].diff().fillna(0)
  hubei_df[args.name_new_deaths] = hubei_df[args.name_total_deaths].diff().fillna(0)
  area_object = area_corona_class('Hubei', hubei_df, hubei_pop,args)
  area_objects_list.append(area_object)

  if(args.plot_time_series == 1):
    #plot(area_objects_list, args, 'unmodified')
    #plot(area_objects_list, args, 'per_mil')
    plot(area_objects_list, args, 'unmodified_covid_days')
    plot(area_objects_list, args, 'per_mil_covid_days')
