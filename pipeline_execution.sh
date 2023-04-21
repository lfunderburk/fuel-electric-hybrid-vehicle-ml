#!/bin/bash

# This script is used to execute the pipeline
echo " "
echo "Starting pipeline execution"

echo " "
echo "Starting data extraction"
SECONDS=0
python src/data/data_extraction.py
data_extraction_success=$?
data_extraction_time=$SECONDS

echo " " 
echo "Starting training model and evaluation"
SECONDS=0
python src/models/train_model.py
train_model_success=$?
train_model_time=$SECONDS

echo " "
echo "Starting data prediction"
SECONDS=0
python src/models/predict_model.py
predict_model_success=$?
predict_model_time=$SECONDS

echo " "
echo "Starting data clustering"
SECONDS=0
python src/models/clustering.py
clustering_success=$?
clustering_time=$SECONDS

# Calculate total time
total_time=$((data_extraction_time + train_model_time + predict_model_time + clustering_time))

# Calculate percentage
data_extraction_percentage=$(awk "BEGIN {print ($data_extraction_time / $total_time) * 100}")
train_model_percentage=$(awk "BEGIN {print ($train_model_time / $total_time) * 100}")
predict_model_percentage=$(awk "BEGIN {print ($predict_model_time / $total_time) * 100}")
clustering_percentage=$(awk "BEGIN {print ($clustering_time / $total_time) * 100}")

# Print the table
printf "name\t\tRan?\tElapsed (s)\tPercentage\n"
printf "predict_model\t%s\t%d\t\t%.3f\n" $predict_model_success $predict_model_time $predict_model_percentage
printf "data_extraction\t%s\t%d\t\t%.3f\n" $data_extraction_success $data_extraction_time $data_extraction_percentage
printf "train_model\t%s\t%d\t\t%.3f\n" $train_model_success $train_model_time $train_model_percentage
printf "clustering\t%s\t%d\t\t%.3f\n" $clustering_success $clustering_time $clustering_percentage


