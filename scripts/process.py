from imported_libraries import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for process.py')
  #Input Data ---------------------------------------------------------------
  parser.add_argument('-country_data_file', '--country_data_file', type = str, dest = 'country_data_file', default = '../data/corona_data/full_data.csv',  help = 'text file with input file names of countries') 
  parser.add_argument('-state_data_file', '--state_data_file', type = str, dest = 'state_data_file', default = '../data/corona_data/us-states.csv', help = 'file with us state level data')
  parser.add_argument('-county_data_file', '--county_data_file', type = str, dest = 'county_data_file', default = '../data/corona_data/us-counties.csv', help = 'file with county level data')
  parser.add_argument('-hubei_data_file', '--hubei_data_file', type = str, dest = 'hubei_data_file', default = '../data/corona_data/covid_19_data_hubei.csv', help = 'file with hubei corona data')
  #Population Info
  parser.add_argument('-us_state_population_file', '--us_state_population_file', type = str, dest = 'us_state_population_file', default = '../data/population_data/nst-est2019-alldata.csv', help = 'file with us population data')
  parser.add_argument('-us_county_pop_file', '--us_county_pop_file', type = str, dest = 'us_county_pop_file', default = '../data/population_data/cc-est2018-alldata.csv', help = 'file with US county population info')
  parser.add_argument('-state_pop_region_name', '--state_pop_region_name', type = str, dest = 'state_pop_region_name', default = 'NAME', help = 'name of state variable in population file')
  parser.add_argument('-state_pop_var_name', '--state_pop_var_name', type = str, dest = 'state_pop_var_name', default = 'POPESTIMATE2019', help = 'name of population variable to use (e.g. POPESTIMATE2016-2019')
  parser.add_argument('-county_pop_region_name', '--county_pop_region_name', type = str, dest = 'county_pop_region_name', default = 'CTYNAME', help = 'name of county variable in county_pop_file')
  parser.add_argument('-county_pop_var_name', '--county_pop_var_name', type = str, dest = 'county_pop_var_name', default = 'TOT_POP', help = 'name of total population variable in county_pop_file')
  parser.add_argument('-county_pop_year', '--county_pop_year', type = int, dest = 'county_pop_year', default = 11, help = 'year to use')
  parser.add_argument('-county_pop_age_group', '--county_pop_age_group', type = int, dest = 'county_pop_age_group', default = 0, help = 'age group to use for county population')
  parser.add_argument('-county_var_name', '--county_var_name', type = str, dest = 'county_var_name', default = 'county', help = 'name of county variable in county data file')
  #Land Area Info
  parser.add_argument('-county_area_file', '--county_area_file', type = str, dest = 'county_area_file', default = '../data/land_data/us_county_land_area.csv', help = 'file with land area of US counties')
  parser.add_argument('-state_area_file', '--state_area_file', type = str, dest = 'state_area_file', default = '../data/land_data/us_state_land_area.csv', help = 'file with land area of US states')
  parser.add_argument('-country_area_file', '--country_area_file', type = str, dest = 'country_area_file', default = '../data/land_data/countries_land_area.csv', help = 'file with area of countries')
  #Geopandas file for US counties
  parser.add_argument('-county_map_file', '--county_map_file', type = str, dest = 'county_map_file', default = '../data/map_data/geojson-counties-fips.json', help = 'file that has map data for US counties')
  #Files User Selected---------------------------------------------------------------
  parser.add_argument('-selected_countries_file', '--selected_countries_file', type = str, dest = 'selected_countries_file', default = 'selected_areas/countries.txt', help = 'Name of text file with names of countries to process')
  parser.add_argument('-selected_states_file', '--selected_states_file', type = str, dest = 'selected_states_file', default = 'selected_areas/states.txt', help = 'Name of text file with name of US states to process')
  parser.add_argument('-selected_counties', '--selected_counties_file', type = str, dest = 'selected_counties_file', default = 'selected_areas/mycounties.txt', help = 'text file with US counties to use')
  #parser.add_argument('-selected_counties', '--selected_counties_file', type = str, dest = 'selected_counties_file', default = 'selected_areas/no_hawaii_alaska_us_counties.txt', help = 'text file with US counties to use')
  #Variables Info---------------------------------------------------------------
  parser.add_argument('-time_series_variables', '--time_series_variables', type = list, dest = 'time_series_variables', default = ['total_deaths'], help = 'list of variables to plot in time series')
  parser.add_argument('-area_var_name', '--area_var_name', type = str, dest = 'area_var_name', default = 'location', help = 'area variable name in input dataset')
  parser.add_argument('-date_name', '--date_name', type = str, dest = 'date_name', default = 'date', help = 'name of date columns in input file')
  parser.add_argument('-n_date_name', '-n_date_name', type = str, dest = 'n_date_name', default = 'Date', help = 'name of date variable for plots')
  parser.add_argument('-name_total_cases', '--name_total_cases', type = str, dest = 'name_total_cases', default = 'total_cases', help = 'name of total case variable in country level file')
  parser.add_argument('-name_total_deaths', '--name_total_deaths', type = str, dest = 'name_total_deaths', default = 'total_deaths', help = 'name of total deaths variable in country level file')
  parser.add_argument('-name_new_cases', '--name_new_cases', type = str, dest = 'name_new_cases', default = 'new_cases', help = 'name of variable in country level file for new cases per day')
  parser.add_argument('-name_new_deaths', '--name_new_deaths', type = str, dest = 'name_new_deaths', default = 'new_deaths', help = 'name of variable in country level file for new deaths per day')
  parser.add_argument('-name_cv_days', '--name_cv_days', type = str, dest = 'name_cv_days', default = 'cv_days', help = 'name of cv outbreak days')
  parser.add_argument('-min_entries_df_lstm', '--min_entries_df_lstm', type = int, dest = 'min_entries_df_lstm', default = 5, help = 'number of entries required in dataframe for it to be considered a sample for the lstm')
  parser.add_argument('-predict_var', '--predict_var', type = str, dest = 'predict_var', default = 'total_deaths', help = 'number of variable used to determine where to start counting cv days, should be whichever variable in your dataset you believe is the most accurate')
  parser.add_argument('-name_fips', '--name_fips', type = str, dest = 'name_fips', default = 'fips', help = 'name of fips variable in input data')
  #Specify output and analysis variables---------------------------------------------------------------
  parser.add_argument('-output_dir', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'name of output_dir')
  parser.add_argument('-plot_time_series', '--plot_time_series', type = int, dest = 'plot_time_series', default = 0, help = 'set to one to plot time series, zero to not')
  parser.add_argument('-fit_logistic', '--fit_logistic', type = int, dest = 'fit_logistic', default = 0, help = 'set to one to fit area trends to logistics')
  parser.add_argument('-make_gif', '--make_gif', type = int, dest = 'make_gif', default = 0, help = 'set to one to make gifs of variables for US counties')
  parser.add_argument('-add_land_area', '--add_land_area', type = int, dest = 'add_land_area', default = 0, help = 'set to one to add land area and population density')
  parser.add_argument('-do_lstm', '--do_lstm', type = int, dest = 'do_lstm', default = 0, help = 'set to one to fit time series distributions using lstm')
  parser.add_argument('-growth_rates', '--growth_rates', type = list, dest = 'growth_rates', default = [1.35], help = 'list of growth rates to plot')
  #Analysis Variables
  parser.add_argument('-train_set_percent', '--train_set_percent', type = int, dest = 'train_set_percentage', default = 0.8, help = 'percentage of data to use for train set')
  parser.add_argument('-cv_day_thres', '--cv_day_thres', type = int, dest = 'cv_day_thres', default = 1000000, help = 'total number of cases to consider it the first day of cv19')
  parser.add_argument('-lstm_seq_length', '--lstm_seq_length', type = int, dest = 'lstm_seq_length', default = 100, help = 'length of lstm sequences. Sequences shorter than this will be padded to specified length')
  parser.add_argument('-mask_value_lstm', '--mask_value_lstm', type = int, dest = 'mask_value_lstm', default = -100, help = 'value to apply to nan/bad values in lstm sequence so they are ignored/masked in training. For some reason leaving them as nan does not end well')
  parser.add_argument('-cv_day_thres_notscaled', '--cv_day_thres_notscaled', type = int, dest = 'cv_day_thres_notscaled', default = 10, help = 'minimum number of deaths to start counting CV days from for unscaled data')
  parser.add_argument('-n_deaths_per_mil', '--n_deaths_per_mil', type = str, dest = 'n_deaths_per_mil', default = 'deaths_per_mil', help = 'name of deaths_per_mil variable')
  parser.add_argument('-population_density_name', '--population_density_name', type = str, dest = 'population_density_name', default = 'population_density', help = 'name of variable for population density in dataframes')
  parser.add_argument('-nyc_fips', '--nyc_fips', type = list, dest = 'nyc_fips', default = [36005, 36047, 36061, 36081, 36085], help = 'list of NYC fips')
  parser.add_argument('-nyc_fips_to_skip', '--nyc_fips_to_skip', type = list, dest = 'nyc_fips_to_skip', default = [36047, 36061, 36081, 36085], help = 'list of NYC fips to skip')
  #Aesthetics : )
  parser.add_argument('-tick_font_size', '--tick_font_size', type = int, dest = 'tick_font_size', default = 4.5, help = 'size of tick labels in plots')
  parser.add_argument('-plot_y_scale', '--plot_y_scale', type = str, dest = 'plot_y_scale', default = 'log', help = 'scale for y axis, set to linear, log, etc')
  parser.add_argument('-linewidth', '--linewidth', type = int, dest = 'linewidth', default = 1, help = 'width of lines for plots')
  parser.add_argument('-markersize', '--markersize', type = int, dest = 'markersize', default = 3, help = 'size of markers to use in scatter plots')
  parser.add_argument('-days_of_cv_predict', '--days_of_cv_predict', type = int, dest = 'days_of_cv_predict', default = 2, help = 'number of days past last date in dataset to predict cv trends')
  parser.add_argument('-min_growth_rate', '--min_growth_rate', type = float, dest = 'min_growth_rate', default = 0.0293838, help = 'min growth rate to compare to') #0.0357 absolute best
  parser.add_argument('-dc_land_area', '--dc_land_area', type = float, dest = 'dc_land_area', default = 68.34, help = 'land area of DC')
  parser.add_argument('-min_indiv_growth_rate', '--min_indiv_growth_rate', type = float, dest = 'min_indiv_growth_rate', default = 7.6595717E-10, help = 'minimum individual contribution to growth rate')
  parser.add_argument('-days_to_count_from_lstm', '--days_to_count_from_lstm', type = str, dest = 'days_to_count_from_lstm', default = '2020-01-01', help = 'days to count from for lstm')
  parser.add_argument('-gif_delay', '--gif_delay', type = str, dest = 'gif_delay', default = '20', help = 'delay between jpg for gif')
  parser.add_argument('-gif_percentile', '--gif_percentile', type = float, dest = 'gif_percentile', default = 99.7, help = 'percentile to use to determine max value used in gif heatmaps (otherwise outliers crowd plots)')
  args = parser.parse_args()

  #Internally Defined Variables-----------------------------------------------------------------------------
  covid_outbreak_days_name = 'COVID-19 Outbreak Days'
  china_scale = 3000
  #Make output folders and objects---------------------------------------------------------------------------
  make_output_dir(args.output_dir)
  area_obj_list = list() #where all area objects stored
  counties_obj_list = list() #where all US county objects stored
  area = 0 #set land area to 0 as default

  #Create dataframe of county, state, country land areas to add to area objects ------------------------------
  area_name = 'Geographic area'
  land_area_name = 'Area in square miles - Land area'
  new_land_area_name = 'land_area' 
  country_area_name = 'Country Name'
  year_country_land_area = '2016' #Survey year used for country areas, years > 2016 had NaN, so using 2016
  mile_to_km_squared = 0.386102 #Conversion factor for mi2 to km2 because units matter :)

  #County Level Land Area Dataframes----
  county_area_df = pd.read_csv(args.county_area_file, encoding='latin-1')
  #Extract County Name from area_name variable (which includes country and state, which we do not want here)
  county_area_df['State'] = county_area_df[area_name].str.split(' - ').str[1]
  county_area_df[args.area_var_name] = county_area_df[area_name].str.rsplit(' - ',1).str[-1]
  #Rename Land Area Variable, because the one in the file is verbose
  county_area_df[new_land_area_name] = county_area_df[land_area_name] 
  #Save only Area Name and Land Area to Area Dataframe
  county_area_df = county_area_df[[args.area_var_name, new_land_area_name, 'State']]
  #Remove Entries that are not counties (e.g. Baja Municipio)
  county_area_df = county_area_df[county_area_df[args.area_var_name].str.contains('County')]


  #State Level Land Area Dataframes----
  state_area_df = pd.read_csv(args.state_area_file, encoding='latin-1')
  #Extract State Name from area_name variable (exclude country name)
  state_area_df[args.area_var_name] = state_area_df[area_name].str.split().str[3]
  #Rename Land Area Variable, because the one in the file is verbose
  state_area_df[new_land_area_name] = state_area_df[land_area_name]
  #Save only Area Name and Land Area to Area Dataframe
  state_area_df = state_area_df[[args.area_var_name, new_land_area_name]]
  #Remove Entries that are not states (e.g. counties)
  state_area_df = state_area_df[~state_area_df[args.area_var_name].str.contains('United States', na=False)]

  #Country Level Land Area Dataframes----
  country_area_df = pd.read_csv(args.country_area_file, encoding='latin-1')
  #Rename area name for consistency with other land area dataframes
  country_area_df[args.area_var_name] = country_area_df[country_area_name]
  #Convert land area to miles squared as that is what is saved for US States and Counties
  country_area_df[new_land_area_name] = country_area_df[year_country_land_area]*mile_to_km_squared
  #Save only Area Name and Land Area to Area Dataframe
  country_area_df = country_area_df[[args.area_var_name, new_land_area_name]]

  #Add all Land Area Dataframes Together
  land_area_df = pd.concat([county_area_df, state_area_df, country_area_df])

  #Create Area Dataframes to use for analysis ---------------------------------------------------------------------------
  #Country level dataframes----
  selected_countries = open(args.selected_countries_file, 'r')
  country_dataframe = pd.read_csv(args.country_data_file, parse_dates = [args.date_name])
  for country in selected_countries:
    if len(country.strip()) > 0:
      country = country.rstrip()
      population = CountryInfo(country).population()
      #Extract entries for selected country
      this_country_df = country_dataframe.loc[country_dataframe[args.area_var_name] == country]
      #Get land area for country and add population density to dataframe
      if(args.add_land_area):
        area = land_area_df[land_area_df[args.area_var_name] == country]
        area = area.iloc[0][new_land_area_name]
        this_country_df[args.population_density_name] = population/area
      #Calculate and add deaths_per_mil and population density variables to dataframe
      this_country_df[args.n_deaths_per_mil] = args.cv_day_thres*this_country_df[args.name_total_deaths].div(population)
      #Calculate CV-day dataframe per million people and for entire population
      cv_days_df_per_mil, cv_days_df_not_scaled = get_cv_days_df(this_country_df, population, args)
      #Create area object for selected country and add to area_obj_list
      area_object = area_corona_class(country, this_country_df, population, area, args, cv_days_df_per_mil, cv_days_df_not_scaled)
      area_obj_list.append(area_object)

  #State level dataframes---
  #Obtain State Population Dataframe to add to State Area Objects later
  state_population_df = pd.read_csv(args.us_state_population_file)
  selected_states = open(args.selected_states_file, "r")
  state_dataframe = pd.read_csv(args.state_data_file, parse_dates = [args.date_name])
  #Rename dataframe columns to match other area dataframes
  state_dataframe = state_dataframe.rename(columns = {'state': args.area_var_name, 'cases': args.name_total_cases, 'deaths': args.name_total_deaths})
  #Split state dataframe by state, compute and append cumulative variables
  for state in selected_states :
    if len(state.strip()) > 0:
      state = state.rstrip()
      #Get dataframe for selected state for US States Dataframe
      this_state_df = state_dataframe[state_dataframe[args.area_var_name].str.match(state)]
      #Get land area for country and add population density to dataframe
      population = get_population(state, state_population_df, args.state_pop_region_name, args.state_pop_var_name)
      if(args.add_land_area):
        area = land_area_df[land_area_df[args.area_var_name] == state]
        area = area.iloc[0][new_land_area_name]
        this_state_df[args.population_density_name] = population/area
      #Sort Values by Date in case they are mixed
      this_state_df = this_state_df.sort_values(by=[args.date_name])
      #Calculate new cases and deaths based on total cases and deaths
      this_state_df[args.name_new_cases] = this_state_df[args.name_total_cases].diff().fillna(0)
      this_state_df[args.name_new_deaths] = this_state_df[args.name_total_deaths].diff().fillna(0)
      #this_state_df.fillna(0, inplace = True)
      #Calculate and add deaths_per_mil variable to dataframe
      this_state_df[args.n_deaths_per_mil] = args.cv_day_thres*this_state_df[args.name_total_deaths].div(population)
      #Calculate CV-day dataframe per million people and for entire population
      cv_days_df_per_mil, cv_days_df_not_scaled = get_cv_days_df(this_state_df, population, args)
      #Create area object for selected country and add to area_obj_list
      area_object = area_corona_class(state, this_state_df, population, area, args, cv_days_df_per_mil, cv_days_df_not_scaled)
      area_obj_list.append(area_object)

  selected_counties = pd.read_csv(args.selected_counties_file)
  #Only keep County, State, and TotalPop variables from user specified selected_counties_files
  selected_counties = selected_counties[['County', 'State', 'TotalPop']]
  counties_dataframe = pd.read_csv(args.county_data_file, parse_dates = [args.date_name])
  #Rename dataframe columns for consistency with other dataframes
  counties_dataframe = counties_dataframe.rename(columns = {'state': args.area_var_name, 'cases': args.name_total_cases, 'deaths': args.name_total_deaths})

  NYC_dataframe = counties_dataframe.loc[counties_dataframe[args.county_var_name] == 'New York City']
  NYC_counties = ['New York', 'Kings', 'Queens', 'Bronx', 'Richmond'] 
  nyc_dictionary = {'New York': 36061, 'Kings': 36047, 'Queens': 36081, 'Bronx': 36005, 'Richmond': 36085}
    
  #Split county dataframe by state, and compute and append cumulative variables
  for ind in selected_counties.index:
    county = selected_counties['County'][ind]
    state = selected_counties['State'][ind]
    if len(county.strip()) > 0:
      #Get dataframe for selected county for US Counties Dataframe, have to expand out New York City NYT county
      if county == 'New York City':
        for i in nyc_dictionary:
        #in NYC_counties and state == 'New York':
          this_county_df = NYC_dataframe
          this_county_df[args.name_fips] = nyc_dictionary[i]
          this_county_df[args.county_var_name] = i
          this_county_df[args.area_var_name] = 'New York'
      else:
        this_county_df = counties_dataframe.loc[counties_dataframe[args.county_var_name] == county]
        #Require that county state matches expectation (turns out there are counties with the same name in different states)
        this_county_df = this_county_df.loc[this_county_df[args.area_var_name] == state]
      #Get land area, population, and population density for selected state
      population = selected_counties['TotalPop'][ind]
      if(args.add_land_area):
        land_area_df = land_area_df.dropna()
        area = land_area_df[land_area_df[args.area_var_name].str.contains(county)]
        area = area[area['State'] == state]
        area = area.iloc[0][new_land_area_name]
        this_county_df[args.population_density_name] = population/area

      #Calculate Deaths per million people and add to dataframe
      this_county_df[args.n_deaths_per_mil] = args.cv_day_thres*this_county_df[args.name_total_deaths].div(population)

      if this_county_df.empty != True:
        #Sort entries by date
        this_county_df = this_county_df.sort_values(by=[args.date_name])
        this_county_df = this_county_df.dropna()
        #Calculate and add new cases and deaths to dataframe
        this_county_df[args.name_new_cases] = this_county_df[args.name_total_cases].diff().fillna(0)
        this_county_df[args.name_new_deaths] = this_county_df[args.name_total_deaths].diff().fillna(0)
        #this_county_df.fillna(0, inplace = True)
        #Calculate CV-day dataframe per million people and for entire population
        cv_days_df_per_mil, cv_days_df_not_scaled = get_cv_days_df(this_county_df, population, args)
        fip = this_county_df.iloc[0][args.name_fips]
        #Create area object for selected country
        area_object = area_corona_class(county, this_county_df, population, area, args, cv_days_df_per_mil, cv_days_df_not_scaled, fip)
        #Add area object to area/counties_obj_list
        area_obj_list.append(area_object)
        counties_obj_list.append(area_object)
        
        if county == 'New York City' or county == 'Los Angeles' or county == 'Santa Clara' or county == 'King':
          print(county)

  #Create Hubei Dataframe
  hubei_pop = 11000000
  hubei_area = 71776 #in square miles
  hubei_df = pd.read_csv(args.hubei_data_file, parse_dates = ['ObservationDate'])
  #Select entires from hubei_df from Hubei
  hubei_df = hubei_df.loc[hubei_df['Province/State'] == 'Hubei']
  #Rename dataframe columns for consistency with other dataframes
  hubei_df = hubei_df.rename(columns = {'ObservationDate': args.date_name, 'Province/State': args.area_var_name, 'Confirmed': args.name_total_cases, 'Deaths': args.name_total_deaths})
  #Calculate new cases and deaths and add to dataframe
  hubei_df[args.name_new_cases] = hubei_df[args.name_total_cases].diff().fillna(0)
  hubei_df[args.name_new_deaths] = hubei_df[args.name_total_deaths].diff().fillna(0)
  #Calculate Deaths per million people and add to dataframe
  hubei_df[args.n_deaths_per_mil] = args.cv_day_thres*hubei_df[args.name_total_deaths].div(hubei_pop)
  #Calculate CV-day dataframe per million people and for entire population
  cv_days_df_per_mil, cv_days_df_not_scaled = get_cv_days_df(hubei_df, hubei_pop, args)
  #Create Hubei area object and add to area_obj_list
  area_object = area_corona_class('Hubei', hubei_df, hubei_pop, hubei_area, args, cv_days_df_per_mil, cv_days_df_not_scaled)
  area_obj_list.append(area_object)

  #Plot Time Series of variables
  if(args.plot_time_series == 1):
    plot(area_obj_list, args, 'raw_dates', ['total_deaths', 'deaths_per_mil'], ['log','linear'])
    plot(area_obj_list, args, 'raw_covid_days', ['total_deaths'], ['log', 'linear'])
    plot(area_obj_list, args, 'per_mil_covid_days', ['deaths_per_mil'], ['log', 'linear'])
    #plot(area_obj_list, args, 'lives_saved_raw_covid_days', ['total_deaths'], ['linear'])


#Make GIFs of time series variables for US counties
ndays = 25 #how many CV Outbreak days to plot
if(args.make_gif == 1):
  make_gif(counties_obj_list, 'df', 'total_deaths', '2020-01-21', '2020-04-05', args)
  make_gif(counties_obj_list, 'df', 'deaths_per_mil', '2020-01-21', '2020-04-05', args)
  make_gif_cv_days(counties_obj_list, 'cv_days_df_not_scaled', 'total_deaths', ndays, args, 210)
  make_gif_cv_days(counties_obj_list, 'cv_days_df_per_mil', 'deaths_per_mil', ndays, args, 500)

if(args.fit_logistic == 1):
  #Plot combined logistic fits 
  #fit_logistic_all(dataframes_list, scale, plot_lives_saved?, scaled/unscaled dataset?)
  fit_logistic_all(area_obj_list, 'linear', 0)
  fit_logistic_all(area_obj_list, 'log', 0)
  fit_logistic_all(area_obj_list, 'linear', 0, 1)
  fit_logistic_all(area_obj_list, 'log', 0, 1)
  #Plot lives saved
  fit_logistic_all(area_obj_list, 'linear', 1)
  fit_logistic_all(area_obj_list, 'log', 1)

  #Plot individual logistic fits
  for county in area_obj_list:
    this_df = county.cv_days_df_not_scaled
    fit_logistic(this_df['cv_days'], this_df['total_deaths'], county.population, args, county.name)

if(args.do_lstm == 1):
  #lstm_combined(area_obj_list[0], args)
  for i in range(len(area_obj_list)):
    if area_obj_list[i].name == 'Los Angeles':
      #seq_lstm(area_obj_list[i], args)
      lstm(area_obj_list[i], args, 1)
      #new_lstm(area_obj_list[0], args)

