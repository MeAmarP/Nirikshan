ROOT_DIR:=./
SRC_DIR:=./src

test:
	PYTHONPATH=$(SRC_DIR) pytest

.PHONY: test

run:
	PYTHONPATH=$(SRC_DIR) python $(SRC_DIR)/main.py

.PHONY: run
