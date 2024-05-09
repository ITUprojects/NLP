from datasets import ClassLabel, Dataset, Features, Sequence, Value

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