from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "ai21labs/Jamba-tiny-random",
    legacy=False,
    cache_dir=None,
    # model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True,
)


vocab = tokenizer.get_vocab()
all_caps = []
print("Vocabulary length", len(vocab))
print("Tokenizer values")
for i, (id, token) in enumerate(vocab.items()):
    # print(">>", id, ":", token)
    # if id == id.upper() and len(id) != 1:
        # all_caps.append((id, token))
    pass

print(" ")

print(f"{len(all_caps)} values found with all caps")
print("All cap values")
for id, token in all_caps:
    # print(">>", id, token)
    pass

print(" ")

print("Max model length", tokenizer.model_max_length)

print(" ")
new_tokens = ["[IMAGE]", "[TABLE]", "<special_token>"]
tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})

added_tokens = tokenizer.additional_special_tokens
print(f"Additional special tokens: {added_tokens}")

print(" ")

# Check individual special tokens
print(f"Padding token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"Unknown token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
print(f"Separator token: '{tokenizer.sep_token}' (ID: {tokenizer.sep_token_id})")
print(f"Classification token: '{tokenizer.cls_token}' (ID: {tokenizer.cls_token_id})")
print(f"Mask token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")

# Check if the tokenizer has a beginning of sentence token
if hasattr(tokenizer, 'bos_token'):
    print(f"Beginning of sentence token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
else:
    print("This tokenizer does not have a specific beginning of sentence token.")

# Check if the tokenizer has an end of sentence token
if hasattr(tokenizer, 'eos_token'):
    print(f"End of sentence token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
else:
    print("This tokenizer does not have a specific end of sentence token.")

print(" ")

print("Special tokens", tokenizer.special_tokens_map)

print(" ")

string_to_check = "Assistant:"
tok = tokenizer.tokenize(string_to_check)
print(f"Tokenized form of {string_to_check} is {tok}")
tok = tokenizer.encode(tok, return_tensors="pt")
print(f"Encoded form of {string_to_check} is {tok}")

print(" ")