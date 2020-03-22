from imported_libraries import *
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for process.py')
  parser.add_argument('-f', '--input_file', type = str, dest = 'input_file', default = '../data/full_data.csv',  help = 'text file with input file names of countries')
  parser.add_argument('-c', '--countries_file', type = str, dest = 'countries_file', default = 'countries.txt', help = 'Name of text file with names of countries to process')
  parser.add_argument('-o', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'name of output_dir')
  parser.add_argument('-country_var_name', '--country_var_name', type = str, dest = 'country_var_name', default = 'location', help = 'country variable name in input dataset')
  parser.add_argument('-plot_time_series', '--plot_time_series', type = int, dest = 'plot_time_series', default = 1, help = 'set to one to plot time series, zero to not')
  parser.add_argument('-date_name', '--date_name', type = str, dest = 'date_name', default = 'date', help = 'name of date columns in input file')
  parser.add_argument('-time_series_variables', '--time_series_variables', type = list, dest = 'time_series_variables', default = ['new_cases', 'new_deaths', 'total_cases', 'total_deaths'], help = 'list of variables to plot in time series')
  args = parser.parse_args()


  #Make output data directory and get input data
  make_output_dir(args.output_dir)
  full_dataframe = pd.read_csv(args.input_file, parse_dates = [args.date_name])
  print('changed')

  #Process make country dataframe and objects
  country_dataframes_list = list()
  countries_file = open(args.countries_file, "r")
  for country in countries_file:
    if len(country.strip()) > 0:
      country = country.rstrip()
      this_country_df = full_dataframe[full_dataframe[args.country_var_name].str.match(country)]
      country_dataframes_list.append(this_country_df)

  if(args.plot_time_series == 1):
    #plot unmodified time series
    for var in args.time_series_variables:
      for country in country_dataframes_list:
        plt.plot(country[args.date_name], country[var], label = country[args.country_var_name].iloc[0])
      plt.title(var + ' vs. ' + args.date_name)
      plt.legend()
      plt.savefig(var + '_unmodified_overlay.pdf')
      plt.close('all')
    

      #Plot normalized plots without creating same start date
      for country in country_dataframes_list:  
        country_name = country[args.country_var_name].iloc[0]
        population = CountryInfo(country_name).population()
        plt.plot(country[args.date_name], country[var]/population, label = country[args.country_var_name].iloc[0])
      plt.title('Normalized ' + var + ' vs. ' + args.date_name)
      plt.legend()
      plt.savefig(var + '_normalized_overlay.pdf')
      plt.close('all')




