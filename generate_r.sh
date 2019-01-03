#!/bin/bash

# generate the R figures
Rscript -e 'library("rmarkdown"); rmarkdown::render("results/main-analysis.Rmd", output_dir="results")'
