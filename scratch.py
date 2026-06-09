import re

with open("tda_report.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Regex to find words containing Latin characters
latin_word_re = re.compile(r'\b[a-zA-Z0-9_\-\^\&]+\b')

# Let's clean the lines by removing LaTeX commands but keeping their arguments if they are plain text
# Or we can just print lines that contain Latin letters and manually inspect.
for idx, line in enumerate(lines):
    line_num = idx + 1
    # Skip comments
    if line.strip().startswith("%"):
        continue
    # Find all Latin words
    words = latin_word_re.findall(line)
    if words:
        # Filter out obvious latex keywords and arguments
        # Let's print the line if it has Latin letters, and we can inspect it
        print(f"Line {line_num}: {line.strip()}")
