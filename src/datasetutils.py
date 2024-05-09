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
