sort-lib:
	isort .

format:
	black . -l 120

format-all:
	make sort-lib
	make format

uv-export:
	uv export --no-hashes --all-extras > requirements.txt
