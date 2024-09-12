# Measuring Dynamic Fairness Metrics in Data Streams

This repository implements DFMonitor, a data structure to monitor real-time fairness status in non-stationary environments.


## Abstract
The increasing deployment of machine learning (ML) algorithms in critical decision-making processes has raised concerns regarding their potential biases and the consequent unjust treatment of specific demographic groups. Despite the extensive studies on fairness definitions and measurement techniques to assess different facets of algorithms and datasets, current methods largely overlook the dynamic, ever-changing nature of real-world data which makes static fairness assessments insufficient. This paper introduces a novel approach for measuring dynamic group fairness metrics for classification in streaming data by applying a time decay to traditional fairness metrics to account for changes over time. We propose novel data structures and algorithms to efficiently monitor such dynamic fairness, offering a real-time, continuously evolving view of fairness. Experiments demonstrate the superiority of our approach over static methods in terms of stability, smoothness, and resistance to occasional fluctuations in the measurement results.





## Directory structure and notable source files:
algorithm folder contains implementation of the DFMonitor data structure and the algorithms to update and query the fairness metrics.
Different fairness metrics need different detailed implementation.
In this file we implement the following fairness metrics:
- accuracy
- coverage ratio
- false positive rate

For each fairness metric, there are three python files (eg, accuracy):
- Accuracy.....py: the implementation of optimized DFMonitor for accuracy
- DF_Accuracy_Fixed_Window_Counter....py: the implementation of baseline DFMonitor for accuracy
- Accuracy_workload.py: the implementation of functions feed a data stream to different versions of DFMonitor and measure the performance and compare with the ground truth.


data folder contains the datasets used in the experiments.
- compas dataset: sourced by ProPublica from Broward County, Florida, comprises data on 7214 defendants for examining racial bias in criminal risk evaluations. It includes demographics, criminal history, COMPAS scores, and two-year recidivism outcomes. It reveals significant racial disparities in False Positive Rates (FPR), especially higher incorrect high-risk assessments for Black defendants compared to their counterparts, making this dataset widely used for studying algorithmic fairness. 
    We use the \textit{``compas\_screening\_date''} field as the time, since it represents the date on which defendants were evaluated by this tool.
    - processed file: compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv
- Baby names dataset: contains the top 1000 most popular girl and boy names in the US from 1880 to 2020, aggregated from the US Social Security Administration \cite{SSA}. It includes the (first) name, year of birth, and sex (male/female) of the infants registered with these names each year, facilitating gender prediction (male/female) from first names over time.
This task is crucial for various applications, such as identifying the gender of authors in biographies when gender information is not explicitly provided, or understanding customer demographics for marketing strategies. 
    The dataset is underscored by the fluidity of name-gender associations over time \cite{blevins2015jane}; for instance, the name Leslie was predominantly male in 1900 but became largely female by 2000. Names like Madison, Morgan, Sydney, and Kendall have similarly shifted genders over the century, highlighting the importance of incorporating temporal dynamics into gender prediction models to maintain accuracy.
To predict the gender based on first names, we employ the genderize.io API \cite{genderize}, with country specified as US. This API connects to a database of over 100,000 first names from around half a million social network profiles as of May 2014 \cite{wais2016gender}. This database is regularly updated by scanning public profiles across major social networks, guaranteeing that its data remains current and accurate.  
    - processed file: name_gender/baby_names_1880_2020_US_predicted.csv


## How to reproduce experiments

- environment: please see requirements.txt
- toy example: 
    - run toy example in experiments/toy_example/FPR_COMPAS.py. It will feed COMPAS to both baseline and optimized version of DFMonitor and output the FPR of protected groups in file case_study_FPR.csv
- other experiments:
  - run python scripts in folder experiments/ to reproduce the experiments in the paper.




