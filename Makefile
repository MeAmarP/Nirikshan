ROOT_DIR:=./
SRC_DIR:=src/

test:
	PYTHONPATH=$(SRC_DIR) pytest -v

.PHONY: test

run:
	PYTHONPATH=$(SRC_DIR) python $(SRC_DIR)/main.py --fpath $(FPATH)

.PHONY: run
