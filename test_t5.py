from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"  # You can choose other versions like t5-base, t5-large
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)




input_text = "premise: The sky is blue hypothesis: The color of the sky is blue"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate the output (you might need to adjust parameters like max_length)
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
