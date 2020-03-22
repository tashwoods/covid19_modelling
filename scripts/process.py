from imported_libraries import *
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for process.py')
  parser.add_argument('-f', '--input_file', type = str, dest = 'input_file', default = '../data/full_data.csv',  help = 'text file with input file names of countries')
  parser.add_argument('-c', '--countries_file', type = str, dest = 'countries_file', default = 'countries.txt', help = 'Name of text file with names of countries to process')
  parser.add_argument('-o', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'name of output_dir')
  parser.add_argument('-country_var_name', '--country_var_name', type = str, dest = 'country_var_name', default = 'location', help = 'country variable name in input dataset')
  args = parser.parse_args()


  #Make output data directory and get input data
  make_output_dir(args.output_dir)
  full_dataframe = pd.read_csv(args.input_file)

  #Process make country dataframe and objects
  country_dataframes_list = list()
  countries_file = open(args.countries_file, "r")
  for country in countries_file:
    if len(country.strip()) > 0:
      country = country.rstrip()
      print(country)
      print(type(country))
      this_country_df = full_dataframe[full_dataframe[args.country_var_name].str.match(country)]
      print(this_country_df)
