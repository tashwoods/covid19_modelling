#!/bin/bash

#Download Hubei Data
mkdir hubei
cd hubei
kaggle datasets download sudalairajkumar/novel-corona-virus-2019-dataset
tar -xvf novel-corona-virus-2019-dataset.zip
mv covid_19_data.csv covid_19_data_hubei.csv
mv covid_19_data_hubei.csv ../../data
cd ..
rm -r hubei

#Download US County and State Data
git clone https://github.com/nytimes/covid-19-data.git
mv covid-19-data/us-counties.csv ../data
mv covid-19-data/us-states.csv ../data
rm -rf covid-19-data

#Download Country Data
curl -O https://covid.ourworldindata.org/data/ecdc/full_data.csv
mv full_data.csv ../data





