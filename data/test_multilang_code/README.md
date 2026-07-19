# Testovací data (1B model)

Automaticky staženo skriptem `scripts/download_test_data.py`.

- **Soubor:** `train.jsonl` (5484 řádků)
- **Obsah:** programování + čeština + angličtina + němčina + hindština
- **Formát:** mix Alpaca (instruction/output) a prostý text (`text`)

Pipeline detekuje formát; pro jistotu použijte `dataset_format=alpaca`
(textové řádky mají jen `text` — skript v kontejneru je umí přes text/hf mapování;
preferujte formát **alpaca** pokud převažují Q&A, nebo **text** pro full pretrain).

Pro trénink „od nuly“ i fine-tune je tento soubor výchozí.
