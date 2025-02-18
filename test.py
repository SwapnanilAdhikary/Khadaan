# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBARTSS")
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBARTSS")