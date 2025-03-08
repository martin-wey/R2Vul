import json
import random
from collections import Counter, defaultdict

from datasets import DatasetDict, Dataset, concatenate_datasets, load_from_disk

RAW_DATASET_PATH = "data/data_nvd/raw_dataset.json"
HF_DATASET_PATH = "data/data_nvd/r2vul_rlhf_dataset"


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


def create_dataset_splits(data, merge_with_non_pairs=True, test_ratio='1:1', language=None, threshold=4):
    data = list(filter(lambda e: e['score'] >= threshold, data))
    random.shuffle(data)

    data_size = len(data)
    train_size = int(.80 * data_size)
    validation_size = int(.10 * data_size)

    train_samples = data[:train_size]
    validation_samples = data[train_size:train_size + validation_size]
    test_samples = data[train_size + validation_size:]

    train_dataset = transform_json_to_hf_dataset(train_samples)
    validation_dataset = transform_json_to_hf_dataset(validation_samples)
    test_dataset = transform_json_to_hf_dataset(test_samples)

    if merge_with_non_pairs or test_ratio != '1:1':
        # `merge_with_non_pairs`:
        # for non-vulnerable function, we replace the fixed functions (from paired functions) with random
        # non-vulnerable functions from the dataset to increase sample diversity and make sure the model
        # does not overfit to paired samples during fine-tuning.
        #
        # `test_ratio`:
        #  we sample additional non-vulnerable samples from non-paired samples in the dataset
        #  to create imbalanced test datasets.
        dataset = load_base_dataset(RAW_DATASET_PATH, language)
        dataset = Dataset.from_dict(transform_to_dict_of_lists(dataset)).shuffle(42)
        data_nv = dataset.filter(lambda e: e['vulnerable'] == 0)

        if merge_with_non_pairs:
            # keep the vulnerable samples
            train_dataset = train_dataset.filter(lambda e: e['vulnerable'] == 1)
            validation_dataset = validation_dataset.filter(lambda e: e['vulnerable'] == 1)
            test_dataset = test_dataset.filter(lambda e: e['vulnerable'] == 1)

            # replace the non-vulnerable samples originating from pairs with unrelated samples
            train_nv_samples, data_nv = select_and_remove_samples(data_nv, train_size)
            validation_nv_samples, data_nv = select_and_remove_samples(data_nv, validation_size)
            test_nv_samples, data_nv = select_and_remove_samples(data_nv, validation_size)

            train_dataset = concatenate_datasets([train_dataset, train_nv_samples])
            validation_dataset = concatenate_datasets([validation_dataset, validation_nv_samples])
            test_dataset = concatenate_datasets([test_dataset, test_nv_samples])

        if test_ratio != '1:1':
            test_size_extra_nv = (validation_size - 1) * int(test_ratio.split(':')[-1])
            test_nv_samples, data_nv = select_and_remove_samples(data_nv, test_size_extra_nv)

            test_dataset = concatenate_datasets([test_dataset, test_nv_samples])

    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })


def get_sample_pairs(data_file, language=None):
    data = load_base_dataset(data_file, language)

    data_vulnerable = list(filter(lambda e: e['vulnerable'] == 1, data))
    data_fixes = list(filter(lambda e: e['origin'] == 'fixed', data))

    data_vulnerable_id_counts = Counter(e['map_id'] for e in data_vulnerable)
    data_fixes_id_counts = Counter(e['map_id'] for e in data_fixes)

    #
    # only consider functions whose `map_id` appear once for each subset
    # effect: ignore functions whose name change or that got removed in a commit
    #
    data_vulnerable_unique_ids = {id for id, count in data_vulnerable_id_counts.items() if count == 1}
    data_fixes_unique_ids = {id for id, count in data_fixes_id_counts.items() if count == 1}

    pairs_ids = data_vulnerable_unique_ids & data_fixes_unique_ids

    data_vulnerable_dict = {e['map_id']: e for e in data_vulnerable if e['map_id'] in pairs_ids}
    data_fixes_dict = {e['map_id']: e for e in data_fixes if e['map_id'] in pairs_ids}

    pairs = [(data_vulnerable_dict[id], data_fixes_dict[id]) for id in pairs_ids]
    return pairs


if __name__ == '__main__':
    pass
    '''
    set_seed(42)
    languages = ["c_sharp", "javascript", "java", "python", "c"]
    
    
    for lang in languages:
        """
        # raw dataset statistics
        dataset = load_base_dataset(DATASET_PATH, lang)
        sample_vuln = list(filter(lambda e: e['vulnerable'], dataset))
        sample_non_vuln = list(filter(lambda e: not e['vulnerable'], dataset))
        print(f"Language: {lang}")
        print(f"\tNumber of vulnerable functions: {len(sample_vuln)}")
        print(f"\tNumber of non-vulnerable functions: {len(sample_non_vuln)}")

        cve_ids = set([e['cve_id'] for e in dataset])
        cwe_ids = set([c for e in dataset for c in e['cwe_id']])
        print(f"\tUnique CVE: {len(cve_ids)}")
        print(f"\tUnique CWE: {len(cwe_ids)}")

        paired_dataset = get_sample_pairs(DATASET_PATH, lang)
        print(f"\tNumber of paired samples: {len(paired_dataset)}")

        print()
        """

        """
        # cleaned dataset statistics - per threshold
        print(f"Language: {lang}")
        for n in range(1, 5):
            dataset = create_dataset_splits(
                [
                    json.loads(l) for l in open(f"data/data_nvd/paired_dataset_{lang}_annotated.json", 'r')
                ], threshold=n, language=lang
            )
            dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
            print(f"\tThreshold={n}")
            dataset = dataset.filter(lambda e: e['vulnerable'])
            print(dataset)
        """
    '''

    """
    # Export datasets into HuggingFace format
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct", trust_remote_code=True)

    all_datasets = []
    for lang in languages:
        dataset = create_dataset_splits(
            [
                json.loads(l) for l in open(f"data/data_nvd/paired_dataset_{lang}_annotated.json", 'r')
            ], threshold=4, language=lang
        )
        dataset = dataset.map(lambda e: {"function_len": len(tokenizer(e["function"])["input_ids"])})
        dataset = dataset.filter(lambda e: e["function_len"] < 4096)
        dataset.save_to_disk(f"data/data_nvd/hf-datasets/{lang}")
        all_datasets.append(dataset)

    merged_dataset = DatasetDict({
        "train": concatenate_datasets([d["train"] for d in all_datasets]),
        "validation": concatenate_datasets([d["validation"] for d in all_datasets]),
        "test": concatenate_datasets([d["test"] for d in all_datasets]),
    })
    print(merged_dataset)
    merged_dataset.save_to_disk("data/data_nvd/hf-datasets/all")
    """