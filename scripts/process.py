from imported_libraries import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for process.py')
  #Corona Data 
  parser.add_argument('-country_data_file', '--country_data_file', type = str, dest = 'country_data_file', default = '../data/corona_data/full_data.csv',  help = 'text file with input file names of countries') 
  parser.add_argument('-state_data_file', '--state_data_file', type = str, dest = 'state_data_file', default = '../data/corona_data/us-states.csv', help = 'file with us state level data')
  parser.add_argument('-county_data_file', '--county_data_file', type = str, dest = 'county_data_file', default = '../data/corona_data/us-counties.csv', help = 'file with county level data')
  parser.add_argument('-hubei_data_file', '--hubei_data_file', type = str, dest = 'hubei_data_file', default = '../data/corona_data/covid_19_data_hubei.csv', help = 'file with hubei corona data')
  #Files User Selected
  parser.add_argument('-selected_countries_file', '--selected_countries_file', type = str, dest = 'selected_countries_file', default = 'selected_areas/countries.txt', help = 'Name of text file with names of countries to process')
  parser.add_argument('-selected_states_file', '--selected_states_file', type = str, dest = 'selected_states_file', default = 'selected_areas/states.txt', help = 'Name of text file with name of US states to process')
  parser.add_argument('-selected_counties', '--selected_counties_file', type = str, dest = 'selected_counties_file', default = 'selected_areas/all_counties.txt', help = 'text file with US counties to use')
  #Variables to Plot
  parser.add_argument('-time_series_variables', '--time_series_variables', type = list, dest = 'time_series_variables', default = ['total_deaths', 'total_cases'], help = 'list of variables to plot in time series')
  #Dataset MetaInfo
  parser.add_argument('-country_var_name', '--country_var_name', type = str, dest = 'country_var_name', default = 'location', help = 'country variable name in input dataset')
  parser.add_argument('-date_name', '--date_name', type = str, dest = 'date_name', default = 'date', help = 'name of date columns in input file')
  parser.add_argument('-n_date_name', '-n_date_name', type = str, dest = 'n_date_name', default = 'Date', help = 'name of date variable for plots')
  parser.add_argument('-name_total_cases', '--name_total_cases', type = str, dest = 'name_total_cases', default = 'total_cases', help = 'name of total case variable in country level file')
  parser.add_argument('-name_total_deaths', '--name_total_deaths', type = str, dest = 'name_total_deaths', default = 'total_deaths', help = 'name of total deaths variable in country level file')
  parser.add_argument('-name_new_cases', '--name_new_cases', type = str, dest = 'name_new_cases', default = 'new_cases', help = 'name of variable in country level file for new cases per day')
  parser.add_argument('-name_new_deaths', '--name_new_deaths', type = str, dest = 'name_new_deaths', default = 'new_deaths', help = 'name of variable in country level file for new deaths per day')
  parser.add_argument('-predict_var', '--predict_var', type = str, dest = 'predict_var', default = 'total_deaths', help = 'number of variable used to determine where to start counting cv days, should be whichever variable in your dataset you believe is the most accurate')
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
  #Create dictionary of county, state, and country land areas for population density calculations later
  area_name = 'Geographic area'
  land_area_name = 'Area in square miles - Land area'
  new_land_area_name = 'land_area' #in square miles
  country_area_name = 'Country Name'
  year_country_land_area = '2016' #year to use for country land area calculation, years > 2016 had more Nan, so using 2016
  mile_to_km_squared = 0.386102

  county_area_df = pd.read_csv(args.county_area_file, encoding='latin-1')
  county_area_df[args.country_var_name] = county_area_df[area_name].str.split().str[-2:].str.join(' ')
  county_area_df[new_land_area_name] = county_area_df[land_area_name] 
  county_area_df = county_area_df[[args.country_var_name, new_land_area_name]]
  county_area_df = county_area_df[county_area_df[args.country_var_name].str.match('County')] #remove entries that aren't counties

  state_area_df = pd.read_csv(args.state_area_file, encoding='latin-1')
  state_area_df[args.country_var_name] = state_area_df[area_name].str.split().str[3]
  state_area_df[new_land_area_name] = state_area_df[land_area_name]
  state_area_df = state_area_df[[args.country_var_name, new_land_area_name]]
  state_area_df = state_area_df[~state_area_df[args.country_var_name].str.contains('United States', na=False)]#do not want to have duplicate US entries

  country_area_df = pd.read_csv(args.country_area_file, encoding='latin-1')
  country_area_df[args.country_var_name] = country_area_df[country_area_name]
  country_area_df[new_land_area_name] = country_area_df[year_country_land_area]*mile_to_km_squared #bc countries saved in km2
  country_area_df = country_area_df[[args.country_var_name, new_land_area_name]]

  land_area_df = pd.concat([county_area_df, state_area_df, country_area_df])

  area_objects_list = list()
  #Obtain country level dataframe
  selected_countries = open(args.selected_countries_file, 'r')
  country_dataframe = pd.read_csv(args.country_data_file, parse_dates = [args.date_name])
  for country in selected_countries:
    if len(country.strip()) > 0:
      country = country.rstrip()
      this_country_df = country_dataframe.loc[country_dataframe[args.country_var_name] == country]
      population = CountryInfo(country).population()
      area = land_area_df.loc[land_area_df[args.country_var_name] == country, new_land_area_name]
      cv_days_df_per_mil, cv_days_df_not_scaled = get_cv_days_df(this_country_df, population, args)
      area_object = area_corona_class(country, this_country_df, population, area, args, cv_days_df_per_mil, cv_days_df_not_scaled)

      area_objects_list.append(area_object)
    if country == 'Chinaz': #Scale China's dataset to see if their reporting seems accurate
      this_country_df = full_dataframe[full_dataframe[args.country_var_name].str.match(country)]
      for var in args.time_series_variables:
        scale_china = 3000
        this_country_df[var] = scale_china * this_country_df[var]
      population = CountryInfo(country).population()
      area = land_area_df.loc[land_area_df[args.country_var_name] == country, new_land_area_name]
      cv_days_df_per_mil, cv_days_df_not_scaled = get_cv_days_df(this_country_df, population, args)
      name = country + 'x' + str(scale_china)
      area_object = area_corona_class(name, this_country_df, population, area, args, cv_days_df_per_mil, cv_days_df_not_scaled)
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
      area = land_area_df.loc[land_area_df[args.country_var_name] == country, new_land_area_name]

      this_state_df = state_dataframe[state_dataframe[args.country_var_name].str.match(state)]
      this_state_df = this_state_df.sort_values(by=[args.date_name])
      #this_state_df[args.name_total_cases] = this_state_df['new_cases'].cumsum()
      this_state_df['new_cases'] = this_state_df['total_cases'].diff()
      this_state_df['new_deaths'] = this_state_df['total_deaths'].diff()
      this_state_df.fillna(0, inplace = True)
      cv_days_df_per_mil, cv_days_df_not_scaled = get_cv_days_df(this_state_df, population, args)
      area_object = area_corona_class(state, this_state_df, population, area, args, cv_days_df_per_mil, cv_days_df_not_scaled)
      area_objects_list.append(area_object)

  #Obtain US County level dataframe
  county_population_df = pd.read_csv(args.us_county_pop_file, encoding='latin-1')
  county_population_df = county_population_df.loc[county_population_df['YEAR'] == args.county_pop_year]
  county_population_df = county_population_df.loc[county_population_df['AGEGRP'] == args.county_pop_age_group]

  #selected_counties = open(args.selected_counties_file, "r")
  selected_counties = pd.read_csv(args.selected_counties_file)
  selected_counties = selected_counties[['County', 'State', 'TotalPop']]


  counties_dataframe = pd.read_csv(args.county_data_file, parse_dates = [args.date_name])
  counties_dataframe = counties_dataframe.rename(columns = {'state': 'location', 'cases': 'total_cases', 'deaths': 'total_deaths'})
  #Split state dataframe by state, and compute and append cumulative variables

  
  #for county in selected_counties:
  counties_obj_list = list()
  for ind in selected_counties.index:
    county = selected_counties['County'][ind]
    state = selected_counties['State'][ind]
    if len(county.strip()) > 0:
      county = county.rstrip()
      #population = get_population(county, county_population_df, args.county_pop_region_name, args.county_pop_var_name, state)
      population = selected_counties['TotalPop'][ind]

      area = land_area_df.loc[land_area_df[args.country_var_name] == country, new_land_area_name]

      county = county.rsplit(' ', 1)[0]
      this_county_df = counties_dataframe.loc[counties_dataframe[args.county_var_name] == county]
      this_county_df = this_county_df.loc[this_county_df['location'] == state]
      if this_county_df.empty != True:
        this_county_df = this_county_df.sort_values(by=[args.date_name])
        this_county_df = this_county_df.dropna()
        #this_state_df[args.name_total_cases] = this_state_df['new_cases'].cumsum()
        this_county_df['new_cases'] = this_county_df['total_cases'].diff()
        this_county_df['new_deaths'] = this_county_df['total_deaths'].diff()
        this_county_df.fillna(0, inplace = True)
        cv_days_df_per_mil, cv_days_df_not_scaled = get_cv_days_df(this_county_df, population, args)

        area_object = area_corona_class(county, this_county_df, population, area, args, cv_days_df_per_mil, cv_days_df_not_scaled, this_county_df.iloc[0]['fips'])
        area_objects_list.append(area_object)
        counties_obj_list.append(area_object)

  #Process Hubei Data
  hubei_pop = 11000000
  hubei_area = 71776 #in square miles
  hubei_df = pd.read_csv(args.hubei_data_file, parse_dates = ['ObservationDate'])
  hubei_df = hubei_df.loc[hubei_df['Province/State'] == 'Hubei']
  hubei_df = hubei_df.rename(columns = {'ObservationDate': 'date', 'Province/State': 'location', 'Confirmed': 'total_cases', 'Deaths': 'total_deaths'})
  hubei_df[args.name_new_cases] = hubei_df[args.name_total_cases].diff().fillna(0)
  hubei_df[args.name_new_deaths] = hubei_df[args.name_total_deaths].diff().fillna(0)
  cv_days_df_per_mil, cv_days_df_not_scaled = get_cv_days_df(hubei_df, hubei_pop, args)
  area_object = area_corona_class('Hubei', hubei_df, hubei_pop, hubei_area, args, cv_days_df_per_mil, cv_days_df_not_scaled)
  area_objects_list.append(area_object)



  if(args.plot_time_series == 1):
    print('hi')
    #plot(area_objects_list, args, 'unmodified', 0)
    #plot(area_objects_list, args, 'per_mil', 0)
    #plot(area_objects_list, args, 'unmodified_covid_days', 0)
    #plot(area_objects_list, args, 'per_mil_covid_days', 1)


vmin, vmax = 0, 1000
df_list = list()

for day in range(10):
  fips = list()
  deaths_per_mil = list()
  cv_days = list()
  print(day)
  print(type(day))
  fips = list()
  deaths_per_mil = list()
  cv_days = list()
  for i in range(len(counties_obj_list)):
    county = counties_obj_list[i]
    thiscounty = county.cv_days_df_per_mil
    if(len(thiscounty.index) > 10):
      this_day = thiscounty.iloc[[str(day)]]
      this_death = this_day['deaths_per_mil'].values
      fips.append(str(int(county.fips)))
      deaths_per_mil.append(float(this_death))
      cv_days.append(day)
  this_df_now = pd.DataFrame({'cv_day':cv_days, 'fips': fips, 'deaths_per_mil': deaths_per_mil})
  df_list.append(this_df_now)

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

df = df_list[0]
fig = px.choropleth(df, geojson=counties, locations='fips', color='deaths_per_mil',
                           color_continuous_scale="Viridis",
                           range_color=(0,3000),
                           scope="usa",
                           labels={'deaths_per_mil':'Deaths Per Million People'}
                          )
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

fig.show()
fig.write_image('test.png')
exit()
fig = px.choropleth(df, geojson=counties, locations='fips', color='unemp',
                           color_continuous_scale="Viridis",
                           range_color=(0, 12),
                           scope="usa",
                           labels={'unemp':'unemployment rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
