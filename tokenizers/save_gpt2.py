import pickle
import json
import tiktoken

# Load the default GPT-2 encoding
enc = tiktoken.get_encoding("gpt2")

# Get the mergeable ranks
merges = enc._mergeable_ranks

tokenizer_data = {
    "pat_str" : (r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""),
    "mergeable_ranks" : merges
}

f = open("gpt2_v50256.pkl", 'wb')
pickle.dump(tokenizer_data, f)
f.close()
