from datasets import ClassLabel, Dataset, Features, Sequence, Value, DatasetDict
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import evaluate
import numpy as np
import os
import pandas as pd




def decode(words, labels, label_names):
	line1 = ''
	line2 = ''
	for word, label in zip(words, labels):
		full_label = label_names[label]
		max_length = max(len(word), len(full_label))
		line1 += word + ' ' * (max_length - len(word) + 1)
		line2 += full_label + ' ' * (max_length - len(full_label) + 1)
	return line1, line2


def align_labels_with_tokens(labels, word_ids):
	new_labels = []
	current_word = None
	for word_id in word_ids:
		if word_id != current_word:
			# Start of a new word!
			current_word = word_id
			label = -100 if word_id is None else labels[word_id]
			new_labels.append(label)
		elif word_id is None:
			# Special token
			new_labels.append(-100)
		else:
			# Same word as previous token
			label = labels[word_id]
			# If the label is B-XXX we change it to I-XXX
			if label % 2 == 1:
				label += 1
			new_labels.append(label)
	return new_labels


def tokenize_and_align_labels(examples, tokenizer):
	tokenized_inputs = tokenizer(
		examples['tokens'], truncation=True, is_split_into_words=True
	)
	all_labels = examples['ner_tags']
	new_labels = []
	for i, labels in enumerate(all_labels):
		word_ids = tokenized_inputs.word_ids(i)
		new_labels.append(align_labels_with_tokens(labels, word_ids))

	tokenized_inputs['labels'] = new_labels
	return tokenized_inputs


def iob2_to_dataset(fp):
	"""Converts an iob2 file to a huggingface dataset.
	fp: path to the iob2 file."""
	with open(fp, encoding='utf-8') as f:
		raw_data = f.readlines()
		
	data = {'tokens': [], 'ner_tags': [], 'ner_tags_id': [], 'index': [], 'id': []}
	current = {'tokens': [], 'ner_tags': [], 'ner_tags_id': [], 'index': []}
	sentence_idx = 0
	word_idx = 0
	for line_idx, line in enumerate(raw_data):
		if line.startswith('#'):
			continue
		if line == '\n':  # new sentence
			data['tokens'].append(current['tokens'])
			data['ner_tags'].append(current['ner_tags'])
			data['ner_tags_id'].append(current['ner_tags_id'])
			data['index'].append(current['index'])
			data['id'].append(str(sentence_idx))
			current = {'tokens': [], 'ner_tags': [], 'ner_tags_id': [], 'index': []}
			sentence_idx += 1
			word_idx = 0
			continue

		try:
			word, ner_tag = line.split()
		except ValueError:
			raise ValueError(f'Invalid line: {line} at line {line_idx+1}')
		current['tokens'].append(word)
		current['ner_tags'].append(ner_tag)
		try:
			current['ner_tags_id'].append(tag_to_id[ner_tag])
		except KeyError:
			raise ValueError(
				f'Invalid tag: {ner_tag}. Valid tags are: {list(tag_to_id.keys())}'
			)
		current['index'].append(word_idx)
		word_idx += 1
	# the file does not end in a newline, so we need to append the last sentence
	if word_idx != 0:
		data['tokens'].append(current['tokens'])
		data['ner_tags'].append(current['ner_tags'])
		data['ner_tags_id'].append(current['ner_tags_id'])
		data['index'].append(current['index'])
		data['id'].append(str(sentence_idx))

	features = Features(
		{
			'id': Value('string'),
			'tokens': Sequence(Value('string')),
			'ner_tags': Sequence(ClassLabel(names=list(tag_to_id.keys()))),
			'ner_tags_id': Sequence(Value('int32')),
			'index': Sequence(Value('int32')),
		}
	)
	dataset_raw = Dataset.from_dict(data, features=features)
	return dataset_raw


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding="max_length", 
        max_length=128,
        is_split_into_words=True,
        return_tensors="pt"
    ).to(device)

    all_labels = examples["ner_tags"]

    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def align_labels_with_tokens(labels, word_ids):
    """
    This function aligns labels with tokens produced by the tokenizer.
    - `-100` is used for special tokens to ignore them during training.
    - If the label is B-XXX, subsequent sub-tokens receive I-XXX.
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            # Convert B-XXX to I-XXX for sub-tokens
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    pred_spans, true_spans = toSpans(true_predictions[0]), toSpans(true_labels[0])
    score = getInstanceScores(pred_spans, true_spans)

    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
        "span_f1": score
    }


def iob2_to_dataset(fp):
	"""Converts an iob2 file to a huggingface dataset.
	fp: path to the iob2 file."""
	with open(fp, encoding='utf-8') as f:
		raw_data = f.readlines()

	data = {'tokens': [], 'ner_tags': [], 'ner_tags_id': [], 'index': [], 'id': []}
	current = {'tokens': [], 'ner_tags': [], 'ner_tags_id': [], 'index': []}
	sentence_idx = 0
	word_idx = 0
	for line_idx, line in enumerate(raw_data):
		if line.startswith('#'):
			continue
		if line == '\n':  # new sentence
			data['tokens'].append(current['tokens'])
			data['ner_tags'].append(current['ner_tags'])
			data['ner_tags_id'].append(current['ner_tags_id'])
			data['index'].append(current['index'])
			data['id'].append(str(sentence_idx))
			current = {'tokens': [], 'ner_tags': [], 'ner_tags_id': [], 'index': []}
			sentence_idx += 1
			word_idx = 0
			continue

		try:
			word, ner_tag = line.split()
		except ValueError:
			raise ValueError(f'Invalid line: {line} at line {line_idx+1}')
		current['tokens'].append(word)
		current['ner_tags'].append(ner_tag)
		try:
			current['ner_tags_id'].append(tag_to_id[ner_tag])
		except KeyError:
			raise ValueError(
				f'Invalid tag: {ner_tag}. Valid tags are: {list(tag_to_id.keys())}'
			)
		current['index'].append(word_idx)
		word_idx += 1
	# the file does not end in a newline, so we need to append the last sentence
	if word_idx != 0:
		data['tokens'].append(current['tokens'])
		data['ner_tags'].append(current['ner_tags'])
		data['ner_tags_id'].append(current['ner_tags_id'])
		data['index'].append(current['index'])
		data['id'].append(str(sentence_idx))

	features = Features(
		{
			'id': Value('string'),
			'tokens': Sequence(Value('string')),
			'ner_tags': Sequence(ClassLabel(names=list(tag_to_id.keys()))),
			'ner_tags_id': Sequence(Value('int32')),
			'index': Sequence(Value('int32')),
		}
	)
	dataset_raw = Dataset.from_dict(data, features=features)
	return dataset_raw


def iob2s_to_datasets(file_paths, reference_path):
    """
    Converts an IOB2 file into a DatasetDict with train and validation splits.
    Assumes the input file uses whitespace to separate tokens and tags, and that each sentence is separated by a blank line.
    """
    tokens, ner_tags = [], []
    sentences, sentence_tags = [], []

    label_set = set()
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    if tokens and ner_tags:
                        sentences.append(tokens)
                        sentence_tags.append(ner_tags)
                    tokens, ner_tags = [], []
                else:
                    try:
                        word, tag = line.split()
                    except:
                        raise ValueError(f"Each line must have two columns: ({i}) {line}")
                    tokens.append(word)
                    ner_tags.append(tag)
                    label_set.add(tag)

            if tokens and ner_tags:
                sentences.append(tokens)
                sentence_tags.append(ner_tags)

    label_list = list(tag_to_id.keys())
    label_mapping = {label: i for i, label in enumerate(label_list)}

    indexed_tags = [[label_mapping[tag] for tag in tags] for tags in sentence_tags]
    dataset = Dataset.from_dict({"tokens": sentences, "ner_tags": indexed_tags})
    reference_german = iob2_to_dataset(reference_path[0]).remove_columns(["ner_tags_id", "index", "id"])
    reference_french = iob2_to_dataset(reference_path[1]).remove_columns(["ner_tags_id", "index", "id"])
    reference_english = iob2_to_dataset(reference_path[2]).remove_columns(["ner_tags_id", "index", "id"])
    

    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list))
    })

    datasets = DatasetDict({
        "train": dataset.cast(features),
        "val_de": reference_german.cast(features),
        "val_fr": reference_french.cast(features),
        "val_en": reference_english.cast(features),
    })

    return datasets


def baseline_res(referenceFiles):
    files = os.listdir('./data/TaggedSeparated/' + "english")

    train_files = np.random.choice(files, 1, replace=False)
    train_files_paths = ['./data/TaggedSeparated/' + "english" + '/' + f for f in train_files]
    datasets = iob2s_to_datasets(train_files_paths, list(referenceFiles.values()))
    tokenized_ds = datasets.map(tokenize_and_align_labels, batched=True)
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=len(label_names)
    ).to(device)
    
    
    train_dataset = tokenized_ds["train"]
    eval_dataset = tokenized_ds["val_" + "fr"]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    res_de = trainer.predict(tokenized_ds["val_de"]).metrics
    res_fr = trainer.predict(tokenized_ds["val_fr"]).metrics
    res_en = trainer.predict(tokenized_ds["val_en"]).metrics
    return res_de, res_fr, res_en


# Rob span-f1 
def toSpans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans


def getInstanceScores(predSpans, goldSpans):
    tp = 0
    fp = 0
    fn = 0
    overlap = len(goldSpans.intersection(predSpans))
    tp += overlap
    fp += len(predSpans) - overlap
    fn += len(goldSpans) - overlap
        
    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    return f1


tag_to_id = {
	'O': 0,
	'B-PER': 1,
	'I-PER': 2,
	'B-ORG': 3,
	'I-ORG': 4,
	'B-LOC': 5,
	'I-LOC': 6,
	'B-MISC': 7,
	'I-MISC': 8,
	'B-POK': 9,
	'I-POK': 10,
}
id_to_tag = {id: tag for tag, id in tag_to_id.items()}


file_path = './data/TaggedSeparated/german/0.iob2'

de_ds = iob2_to_dataset(file_path)

ner_feature_fr = de_ds.features['ner_tags']
label_names = ner_feature_fr.feature.names
print("label_names: ",label_names)

words = de_ds[0]['tokens']
labels = de_ds[0]['ner_tags']
print('\n'.join(decode(words, labels, label_names)))


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device: ",device)


model_id = 'google-bert/bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_id)


model = AutoModelForTokenClassification.from_pretrained(
    model_id, num_labels=len(label_names)
).to(device)


tokenized_ds = de_ds.map(tokenize_and_align_labels, batched=True)

print("tokenized_ds: ",tokenized_ds)


metric = evaluate.load("seqeval")


model = AutoModelForTokenClassification.from_pretrained (
    model_id,
    num_labels=len(label_names),
    id2label={id: label for id, label in enumerate(label_names)},
    label2id={label: id for id, label in enumerate(label_names)},
).to(device)
model.config.num_labels


args = TrainingArguments(
    "mbert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    # remove_unused_columns=False
)


tag_to_id = {
	'O': 0,
	'B-PER': 1,
	'I-PER': 2,
	'B-ORG': 3,
	'I-ORG': 4,
	'B-LOC': 5,
	'I-LOC': 6,
	'B-MISC': 7,
	'I-MISC': 8,
	'B-POK': 9,
	'I-POK': 10,
}
id_to_tag = {id: tag for tag, id in tag_to_id.items()}


tag_to_id = {
	'O': 0,
	'B-PER': 1,
	'I-PER': 2,
	'B-ORG': 3,
	'I-ORG': 4,
	'B-LOC': 5,
	'I-LOC': 6,
	'B-MISC': 7,
	'I-MISC': 8,
	'B-POK': 9,
	'I-POK': 10,
}
id_to_tag = {id: tag for tag, id in tag_to_id.items()}


llang = {
    "fr": "french",
    "en": "english",
    "de": "german"
}


ITERATIONS = 5

referenceFiles = {
    "fr": './ReferenceText/ReferenceTextFrench.iob2',
    "en": './ReferenceText/ReferenceTextEnglish.iob2',
    "de": './ReferenceText/ReferenceTextGerman.iob2'
}

main_df = pd.DataFrame(columns=["train_lang", "num_train_files", "test_lang", "precision", "recall", "f1", "accuracy", "span_f1"])

de, fr, en = baseline_res(referenceFiles)
main_df.loc[-1] = ["baseline", "0", "de", de["test_precision"], de["test_recall"], de["test_f1"], de["test_accuracy"], de["test_span_f1"]]
main_df.index = main_df.index + 1
main_df = main_df.sort_index()
main_df.loc[-1] = ["baseline", "0", "fr", fr["test_precision"], fr["test_recall"], fr["test_f1"], fr["test_accuracy"], fr["test_span_f1"]]
main_df.index = main_df.index + 1
main_df = main_df.sort_index()
main_df.loc[-1] = ["baseline", "0", "en", en["test_precision"], en["test_recall"], en["test_f1"], en["test_accuracy"], en["test_span_f1"]]
main_df.index = main_df.index + 1
main_df = main_df.sort_index()

for lang, language in llang.items():
    files = os.listdir('./data/TaggedSeparated/' + language)
    for i in range(len(files)):
        df = pd.DataFrame(columns=["test_lang", "precision", "recall", "f1", "accuracy", "span_f1"])

        for it in range(ITERATIONS):
            train_files = np.random.choice(files, i+1, replace=False)
            train_files_paths = ['./data/TaggedSeparated/' + language + '/' + f for f in train_files]
            datasets = iob2s_to_datasets(train_files_paths, list(referenceFiles.values()))
            tokenized_ds = datasets.map(tokenize_and_align_labels, batched=True)
            
            model = AutoModelForTokenClassification.from_pretrained(
                model_id, num_labels=len(label_names)
            ).to(device)
            
            
            train_dataset = tokenized_ds["train"]
            eval_dataset = tokenized_ds["val_" + lang]

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
            )

            trainer.train()

            res_de = trainer.predict(tokenized_ds["val_de"]).metrics
            res_fr = trainer.predict(tokenized_ds["val_fr"]).metrics
            res_en = trainer.predict(tokenized_ds["val_en"]).metrics
            df.loc[-1] = ["de", res_de["test_precision"], res_de["test_recall"], res_de["test_f1"], res_de["test_accuracy"], res_de["test_span_f1"]]
            df.index = df.index + 1
            df = df.sort_index()
            df.loc[-1] = ["fr", res_fr["test_precision"], res_fr["test_recall"], res_fr["test_f1"], res_fr["test_accuracy"], res_fr["test_span_f1"]]
            df.index = df.index + 1
            df = df.sort_index()
            df.loc[-1] = ["en", res_en["test_precision"], res_en["test_recall"], res_en["test_f1"], res_en["test_accuracy"], res_en["test_span_f1"]]
            df.index = df.index + 1
            df = df.sort_index()
        
        # group by lang and average
        df = df.groupby("test_lang").mean()
        # add to parent dataframe
        main_df.loc[-1] = [language, i+1, "de", df.loc["de"]["precision"], df.loc["de"]["recall"], df.loc["de"]["f1"], df.loc["de"]["accuracy"], df.loc["de"]["span_f1"]]
        main_df.index = main_df.index + 1
        main_df = main_df.sort_index()
        main_df.loc[-1] = [language, i+1, "fr", df.loc["fr"]["precision"], df.loc["fr"]["recall"], df.loc["fr"]["f1"], df.loc["fr"]["accuracy"], df.loc["fr"]["span_f1"]]
        main_df.index = main_df.index + 1
        main_df = main_df.sort_index()
        main_df.loc[-1] = [language, i+1, "en", df.loc["en"]["precision"], df.loc["en"]["recall"], df.loc["en"]["f1"], df.loc["en"]["accuracy"], df.loc["en"]["span_f1"]]
        main_df.index = main_df.index + 1
        main_df = main_df.sort_index()

main_df.to_csv("results.tsv", sep="\t", index=False)