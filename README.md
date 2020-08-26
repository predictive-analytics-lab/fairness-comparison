**Update:** this is a modified [fork of another project](https://github.com/algofairness/fairness-comparison). We realised that the changes were making were no longer compatible with the underlying project so we moved the work we'd been doing to a new project, [EthicML](https://github.com/predictive-analytics-lab/EthicML). We're leaving this modified fork available, but like the underlying project, it's not maintained. For an up to date algorithmic fairness comparison library, please use [EthicML](https://github.com/predictive-analytics-lab/EthicML).

https://github.com/predictive-analytics-lab/EthicML

## TL;DR
This project has moved to [EthicML](https://github.com/predictive-analytics-lab/EthicML)

---

This repository is meant to facilitate the benchmarking of fairness aware machine learning algorithms.

The associated paper is:

A comparative study of fairness-enhancing interventions in machine learning by Sorelle A. Friedler, Carlos Scheidegger, Suresh Venkatasubramanian, Sonam Choudhary, Evan P. Hamilton, and Derek Roth. https://arxiv.org/abs/1802.04422

To run the benchmarks, clone the repository and run:

    $ python3 benchmark.py

This will write out metrics for each dataset to the results/ directory.

To generate graphs and other analysis run:

    $ python3 analysis.py

If you do not yet have all the packages installed, you may need to run:

    $ pip install -r requirements.txt

*Optional*:  The benchmarks rely on preprocessed versions of the datasets that have been included
in the repository.  If you would like to regenerate this preprocessing, run the below command
before running the benchmark script:

    $ python3 preprocess.py

To add new datasets or algorithms, see the instructions in the readme files in those directories.

## OS-specific things

### On Ubuntu

(We tested on Ubuntu 16.04, your mileage may vary)

You'll need `python3-dev`:

    $ sudo apt-get install python3-dev


### Additional analysis-specific requirements

To regenerate figures (this is messy right now. we're working on it)

Python requirements (use pip):

* `ggplot`

System requirements:

* `pandoc`  (`brew install pandoc` on a Mac or `apt-get install pandoc` on Linux)
* R  (Mac download link: https://cran.rstudio.com/bin/macosx/R-3.4.3.pkg)

R package requirements (use `install.packages`):

* `rmarkdown`
* `stringr`
* `ggplot2`
* `dplyr`
* `magrittr`
* `corrplot`
* `robust`
