import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

supported_models = ["bert-base-uncased", "gpt2"]


class WordIDsDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class that lets us return the word ID list as
    a tensor so it is compatible with the DataLoader. We need the
    word IDs to merge back together the words that get split into
    multiple tokens by the tokenizer.
    """

    def __init__(self, sents: transformers.tokenization_utils_base.BatchEncoding):
        self.sents = sents

    def __len__(self):
        return self.sents["input_ids"].size(0)

    def __getitem__(self, idx: int):
        sent = self.sents[idx]
        return {
            "input_ids": torch.tensor(sent.ids),
            "word_ids": torch.tensor(
                list(map(lambda e: -1 if e is None else e, sent.word_ids))
            ),
            "attention_mask": torch.tensor(sent.attention_mask),
        }


def get_batches(input_dict, batch_size=1):
    """
    Dataloader for WordIDsDataset.
    """
    tensor_dataset = WordIDsDataset(input_dict)
    tensor_dataloader = torch.utils.data.DataLoader(
        tensor_dataset, batch_size=batch_size
    )
    return tensor_dataloader


def model_init(model_identifier):
    """
    Downloads the model & tokenizer with the specified identifier from the transformers library.

    Parameters
    ----------
    model_identifier : str
        One of the 'shortcut names' identifying models in the transformers library. 
        See https://huggingface.co/transformers/v2.1.1/pretrained_models.html
        Currently only supporting BERT and GPT2.
    """
    assert (
        model_identifier in supported_models
    ), f"model_identifier must be one of {supported_models}"

    model = AutoModel.from_pretrained(
        model_identifier, output_hidden_states=True, output_attentions=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    if "GPT2Model" in str(model):
        tokenizer.add_prefix_space = True
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_layer_activations(model, tokenizer, input_texts):
    """
    Retrieve the activations for each of the model's layers, for the given input texts.
    """
    encoded_texts = tokenizer.batch_encode_plus(
        input_texts, is_split_into_words=True, padding=True, return_tensors="pt",
    )
    dl = get_batches(encoded_texts)

    texts_activations = []

    for batch, input_dict in enumerate(dl):
        word_ids = input_dict.pop("word_ids")
        input_ids = input_dict["input_ids"]
        attention_mask = input_dict["attention_mask"]
        token_len = attention_mask.sum().item()

        if "BertModel" in str(model):
            word_indices = np.array(
                list(map(lambda e: -1 if e is None else e, word_ids.numpy().squeeze()))
            )[1 : token_len - 1]
            word_groups = np.split(
                np.arange(word_indices.shape[0]) + 1,
                np.unique(word_indices, return_index=True)[1],
            )[1:]
            input_token_embeddings = model.embeddings.word_embeddings(input_ids)

        elif "GPT2Model" in str(model):
            word_indices = np.array(
                list(map(lambda e: -1 if e is None else e, word_ids.numpy().squeeze()))
            )[:token_len]
            word_groups = np.split(
                np.arange(word_indices.shape[0]),
                np.unique(word_indices, return_index=True)[1],
            )[1:]
            input_token_embeddings = model.wte(input_ids)
        else:
            raise NotImplementedError("only supports BERT or GPT2 models")

        model_output = model(**input_dict)
        layer_activations = torch.stack(model_output.hidden_states)
        layer_activations_per_word = torch.stack(
            [
                torch.stack(
                    [
                        layer_activations[i, 0, token_ids_word, :].mean(axis=0)
                        if i > 0
                        else input_token_embeddings[0, token_ids_word, :].mean(axis=0)
                        for i in range(len(model_output.hidden_states))
                    ]
                )
                for token_ids_word in word_groups
            ]
        )

        texts_activations.append(layer_activations_per_word.detach().numpy())

    return texts_activations

