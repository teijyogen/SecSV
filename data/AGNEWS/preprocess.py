from torchtext import datasets as txt_datasets
from torchtext.data.functional import to_map_style_dataset
import gensim.downloader
from gensim.parsing.preprocessing import remove_stopwords, strip_non_alphanum, stem, strip_punctuation
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from nltk.stem import PorterStemmer
import torchtext
import torch
from torchtext.vocab import Vectors

# embeds = gensim.downloader.load("word2vec-google-news-300")
# embeds.save_word2vec_format("google_news.txt")
tokenizer = get_tokenizer('basic_english')
word_embeds = torchtext.vocab.GloVe(name="6B", dim=300)
# word_embeds = Vectors("google_news.txt")

def collate_batch(batch):

    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(int(_label) - 1)
        _text = strip_punctuation(_text)
        _text = remove_stopwords(_text)
        _text = strip_non_alphanum(_text)
        tokens = tokenizer(_text)
        embeds = word_embeds.get_vecs_by_tokens(tokens, lower_case_backup=True)
        embeds = embeds.sum(dim=0)
        text_list.append(embeds)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)

    return label_list, text_list

def data_preprocess(data, filename):
    data_loader = DataLoader(data, batch_size=1024, shuffle=False, collate_fn=collate_batch)

    data_list, label_list = [], []
    for label, text in data_loader:
        text = text.reshape(-1, 300)
        data_list.append(text)
        label_list.append(label)

    dataset = torch.cat(data_list), torch.cat(label_list)
    print(dataset[0].shape)
    print(dataset[1].shape)
    torch.save(dataset, filename)


if __name__ == '__main__':
    train_data, test_data = txt_datasets.AG_NEWS()
    train_data, test_data = to_map_style_dataset(train_data), to_map_style_dataset(test_data)

    data_preprocess(train_data, "train_data.pt")
    data_preprocess(test_data, "test_data.pt")