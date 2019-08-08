def unredact(text):
	sorted_text = text
	listed_text = sorted_text.split()
	listed_text.sort()
	return ' '.join(listed_text)
