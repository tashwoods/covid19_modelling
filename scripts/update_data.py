from imported_libraries import *
CSV_URL = 'https://data.world/covid-19-data-resource-hub/covid-19-case-counts/workspace/file?filename=COVID-19+Cases.csv#'

data = pd.read_csv(CSV_URL)
print(data)

with requests.Session() as s:
  download = s.get(CSV_URL)
  
  decoded_content = download.content.decode('utf-8')
  cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
  my_list = list(cr)
  for row in my_list:
    print(row)
