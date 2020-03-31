from imported_libraries import *

class area_corona_class:
  def __init__(self, name, df, population, area, input_args, cv_days_df_per_mil = 0, cv_days_df_not_scaled = 0, fips = -1):
    self.name = name
    self.df = df
    self.population = population
    self.input_args = input_args
    self.area = area
    self.cv_days_df_per_mil = cv_days_df_per_mil
    self.cv_days_df_not_scaled = cv_days_df_not_scaled
    self.fips = fips
