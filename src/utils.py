import json
from collections import defaultdict

from datasets import Dataset, concatenate_datasets, load_from_disk

RAW_DATASET_PATH = "data/raw_dataset.json"
HF_DATASET_PATH = "data/r2vul_dataset"


def add_samples(datasets, ratio):
    dataset = load_from_disk(HF_DATASET_PATH)["train"]
    for key, test_dataset in datasets.items():
        language = "_".join(key.split('_')[1:])

        train_dataset = dataset.filter(lambda e: e["lang"] == language, num_proc=32)

        # load all samples in the current language from the raw dataset
        ds_extra_samples = load_base_dataset(RAW_DATASET_PATH, language)
        ds_extra_samples = Dataset.from_dict(transform_to_dict_of_lists(ds_extra_samples)).shuffle(42)

        # only select non-vulnerable samples to create the imbalance
        ds_extra_samples = ds_extra_samples.filter(lambda e: e["vulnerable"] == 0, num_proc=32)
        # make sure the samples are unseen in both training and test sets
        ds_extra_samples = ds_extra_samples.filter(lambda e: e["commit_URL"] not in test_dataset["commit_URL"] and
                                                             e["commit_URL"] not in train_dataset["commit_URL"],
                                                   num_proc=32)

        # initial ratio is balanced (1:1)
        num_samples = len(test_dataset) // 2
        ratio_nv = int(ratio.split(":")[-1])
        num_new_samples = int((ratio_nv - 1) * num_samples)
        new_samples = ds_extra_samples.select(range(num_new_samples))

        # make sure new_samples have identical features than the test set
        for feature in test_dataset.features:
            if feature not in new_samples.features:
                new_samples = new_samples.add_column(feature, [None] * len(new_samples))

        datasets[key] = concatenate_datasets([test_dataset, new_samples])

    return datasets


def transform_to_dict_of_lists(data_list):
    transformed_data = defaultdict(list)
    for entry in data_list:
        for key, value in entry.items():
            transformed_data[key].append(value)
    return transformed_data


def transform_json_to_hf_dataset(data):
    data_vulnerable = [e['pre'] for e in data]
    data_non_vulnerable = [e['post'] for e in data]

    transformed_data_vulnerable = transform_to_dict_of_lists(data_vulnerable)
    transformed_data_non_vulnerable = transform_to_dict_of_lists(data_non_vulnerable)

    dataset_vulnerable = Dataset.from_dict(transformed_data_vulnerable)
    dataset_non_vulnerable = Dataset.from_dict(transformed_data_non_vulnerable)
    dataset = concatenate_datasets([dataset_vulnerable, dataset_non_vulnerable])

    return dataset


test_words_to_exclude = ('test', 'Test', '@test')


def load_base_dataset(data_file, language=None):
    with open(data_file, 'r') as f:
        data = json.load(f)

    if language is not None:
        data = list(filter(lambda e: e['lang'] == language, data))

    # filter out samples related to tests
    data = list(filter(lambda e: not any(word in e['map_id'] for word in test_words_to_exclude), data))

    return data


def select_and_remove_samples(dataset, n):
    indices = list(range(n))
    selected_samples = dataset.select(indices)
    remaining_dataset = dataset.filter(lambda e, idx: idx not in indices, with_indices=True)

    return selected_samples, remaining_dataset
