import fire
import os
import statistics
import sys

from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import ProcessedData
from algorithms.list import ALGORITHMS
from metrics.list import get_metrics

from algorithms.ParamGridSearch import ParamGridSearch

NUM_TRIALS_DEFAULT = 5
GPU_DEFAULT = '0'


def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for i in result:
        print("  %s" % i)
    return result


def run(num_trials=NUM_TRIALS_DEFAULT, datasets=get_dataset_names(),
        algorithms=get_algorithm_names(), gpu=GPU_DEFAULT):
    algorithms_to_run = algorithms

    print("Datasets: '%s'" % datasets)
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in datasets:
            continue

        print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
        for sensitive in dataset_obj.get_sensitive_attributes():
            processed_dataset = ProcessedData(dataset_obj)
            train_test_splits = processed_dataset.create_train_test_splits(num_trials, sensitive)

            print("Sensitive attribute:" + sensitive)

            detailed_files = dict((k, create_detailed_file(
                                          dataset_obj.get_results_filename(sensitive, k),
                                          dataset_obj,
                                          processed_dataset.get_sensitive_values(k), k))
                for k in train_test_splits.keys())

            for algorithm in ALGORITHMS:
                if not algorithm.get_name() in algorithms_to_run:
                    continue

                print("    Algorithm: %s" % algorithm.get_name())
                print("       supported types: %s" % algorithm.get_supported_data_types())
                if algorithm.__class__ is ParamGridSearch:
                    param_files =  \
                        dict((k, create_detailed_file(
                                     dataset_obj.get_param_results_filename(sensitive, k,
                                                                            algorithm.get_name()),
                                     dataset_obj, processed_dataset.get_sensitive_values(k), k))
                          for k in train_test_splits.keys())
                for i in range(0, num_trials):
                    for supported_tag in algorithm.get_supported_data_types():
                        train, test = train_test_splits[supported_tag][i]
                        try:
                            params, results, param_results = run_eval_alg(algorithm,
                                                                          train,
                                                                          test,
                                                                          dataset_obj,
                                                                          processed_dataset,
                                                                          all_sensitive_attributes,
                                                                          sensitive,
                                                                          supported_tag,
                                                                          gpu)
                        except Exception as e:
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            print("Failed: %s" % e, file=sys.stderr)
                        else:
                            write_alg_results(detailed_files[supported_tag],
                                              algorithm.get_name(), params, i, results)
                            if algorithm.__class__ is ParamGridSearch:
                                for params, results in param_results:
                                    write_alg_results(param_files[supported_tag],
                                                      algorithm.get_name(), params, i, results)

            print("Results written to:")
            for supported_tag in algorithm.get_supported_data_types():
                print("    " + dataset_obj.get_results_filename(sensitive, supported_tag))

            for detailed_file in detailed_files.values():
                detailed_file.close()


def write_alg_results(file_handle, alg_name, params, run_id, results_list):
    line = alg_name + ','
    params = ";".join("%s=%s" % (k, v) for (k, v) in params.items())
    line += params + (',%s,' % run_id)
    line += ','.join(str(x) for x in results_list) + '\n'
    file_handle.write(line)
    # Make sure the file is written to disk line-by-line:
    file_handle.flush()
    os.fsync(file_handle.fileno())


def run_eval_alg(algorithm, train, test, dataset, processed_data, all_sensitive_attributes,
                 single_sensitive, tag, gpu):
    """
    Runs the algorithm and gets the resulting metric evaluations.
    """
    privileged_vals = dataset.get_privileged_class_names_with_joint(tag)
    positive_val = dataset.get_positive_class_val(tag)

    # get the actual classifications and sensitive attributes
    actual = test[dataset.get_class_attribute()].values.tolist()

    params = {**algorithm.get_default_params(), 'gpu': gpu}
    # Note: the training and test set here still include the sensitive attributes because
    # some fairness aware algorithms may need those in the dataset.  They should be removed
    # before any model training is done.
    predicted, predictions_list = algorithm.run(train,
                                                test,
                                                dataset.get_class_attribute(),
                                                positive_val,
                                                all_sensitive_attributes,
                                                single_sensitive,
                                                privileged_vals,
                                                params)

    # make dictionary mapping sensitive names to sensitive attr test data lists
    dict_sensitive_lists = {}
    for sens in all_sensitive_attributes:
        dict_sensitive_lists[sens] = test[sens].values.tolist()

    sensitive_dict = processed_data.get_sensitive_values(tag)
    one_run_results = []
    for metric in get_metrics(dataset, sensitive_dict, tag):
        result = metric.calc(actual, predicted, dict_sensitive_lists, single_sensitive,
                             privileged_vals, positive_val)
        one_run_results.append(result)

    # handling the set of predictions returned by ParamGridSearch
    results_lol = []
    if predictions_list:
        for param_name, param_val, predictions in predictions_list:
            params_dict = {param_name: param_val}
            results = []
            for metric in get_metrics(dataset, sensitive_dict, tag):
                result = metric.calc(actual, predictions, dict_sensitive_lists, single_sensitive,
                                     privileged_vals, positive_val)
                results.append(result)
            results_lol.append((params_dict, results))

    return params, one_run_results, results_lol


def get_metrics_list(dataset, sensitive_dict, tag):
    return [metric.get_name() for metric in get_metrics(dataset, sensitive_dict, tag)]


def get_detailed_metrics_header(dataset, sensitive_dict, tag):
    return ','.join(['algorithm', 'params', 'run-id'] + get_metrics_list(dataset, sensitive_dict, tag))


def get_dict_sensitive_vals(dict_sensitive_lists):
    """
    Takes a dictionary mapping sensitive attributes to lists in the test data and returns a
    dictionary mapping sensitive attributes to lists containing each sensitive value only once.
    """
    newdict = {}
    for sens in dict_sensitive_lists:
         sensitive = dict_sensitive_lists[sens]
         newdict[sens] = list(set(sensitive))
    return newdict


def create_detailed_file(filename, dataset, sensitive_dict, tag):
    f = open(filename, 'w')
    f.write(get_detailed_metrics_header(dataset, sensitive_dict, tag) + '\n')
    return f


def main():
    fire.Fire(run)


if __name__ == '__main__':
    main()
