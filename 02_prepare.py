#!/usr/bin/env python
from transformers import pipeline
import db
import sqlalchemy
import tqdm
import pandas as pd
import functools
import math
import sklearn.cluster
from sqlalchemy.sql.expression import select
import torch
import itertools

device = "cuda" if torch.cuda.is_available() else "mps" if torch.device("mps") else "cpu"
device = torch.device(device)

def download_books(output_dir: str = "books"):
    import urllib.request
    import shutil
    import os.path

    tqdm.tqdm.write("Downloading books")
    def update(blocknum, blocksize, totalsize):
        progressbar.total = totalsize
        if blocknum == 0:
            progressbar.update(blocksize)
        else:
            progressbar.update(progressbar.n + blocksize)

    url = "http://aleph.gutenberg.org/cache/generated/feeds/txt-files.tar.zip"
    progressbar = tqdm.tqdm(total=0, unit="B", unit_scale=True)
    # urllib.request.urlretrieve(url, "books.tar.zip", reporthook=update)
    # shutil.unpack_archive("books.tar.zip", output_dir, format="zip")
    shutil.unpack_archive(f"{output_dir}/txt-files.tar", output_dir, format="tar")

    url = "http://aleph.gutenberg.org/cache/generated/feeds/pg_catalog.csv"
    progressbar = tqdm.tqdm(total=0, unit="B", unit_scale=True)
    urllib.request.urlretrieve(url, os.path.join(output_dir, "books.csv"), reporthook=update)
    progressbar.close()

@functools.lru_cache(maxsize=1)
def load_metadata(metadata_file: str):
    from datetime import datetime
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    return pd.read_csv(metadata_file, parse_dates=['Issued'], date_parser=dateparse)

def get_book(directory: str, book_id: int):
    import os.path
    import gutenbergpy.textget
    path = os.path.join(directory, "cache", "epub", str(book_id), f"pg{book_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return gutenbergpy.textget.strip_headers(bytes(f.read(), "utf-8")).decode("utf-8").strip('\n')

def summarize(data: db.Book, model: str = "pszemraj/pegasus-x-large-book-summary"):
    import transformers
    transformers.logging.set_verbosity(transformers.logging.ERROR)
    summarizer = pipeline("summarization", model=model, trust_remote_code=True, device=device)
    tokens = summarizer.tokenizer.encode(data)
    if len(tokens) > summarizer.tokenizer.model_max_length:
        tokens = summarizer.tokenizer.truncate_sequences(tokens,truncation_strategy="longest_first")[0]
        # tokens = tokens[:summarizer.tokenizer.model_max_length]
    data = summarizer.tokenizer.decode(tokens)
    return summarizer(data, max_length=300, min_length=30, do_sample=False)[0]["summary_text"]


def create_embeddings(data: str, model: str = "nvidia/NV-Embed-v2"):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model, trust_remote_code=True)

    # get the embeddings
    max_length = 32768
    return model.encode([data], instruction="", max_length=max_length)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=10)
    parser.add_argument("--embedding_model", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--summary_model", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--output_dir", type=str, default="books")
    parser.add_argument("--cluster_batchsize", type=int, default=2000)
    args = parser.parse_args()

    import os.path


    ## Download books
    if not os.path.exists(args.output_dir):
        download_books(output_dir=args.output_dir)
    
    engine = sqlalchemy.create_engine("sqlite:///books.db")
    db.Base.metadata.create_all(engine)
    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    session = Session()

    ## Load books into database
    if not session.query(db.Book).count():
        print("Loading books into database")
        books = load_metadata(f"{args.output_dir}/books.csv")
        for index, book in tqdm.tqdm(books.fillna('').iterrows(), desc="Loading books into database", total=books.shape[0]):
            book_text = get_book("books", book["Text#"])
            if book_text:
                book = db.Book(
                    id=book["Text#"],
                    text=book_text,
                    issued=book["Issued"],
                    language=book["Language"],
                    title=book["Title"],
                    authors=book["Authors"].split(";"),
                    subjects=book["Subjects"].split(";"),
                    locc=book["LoCC"].split(";")
                )
                session.add(book)
            if index % args.batchsize == 0:
                session.commit()
    session.commit()
    

    book_count = session.query(db.Book).count()
    for book in tqdm.tqdm(session.query(db.Book).yield_per(args.batchsize), total=book_count, desc="Creating summaries"):
        summary = db.Summary(book_id=book.id, book=book, summary=summarize(book.text, model=args.summary_model), summarization_model=args.summary_model)
        session.add(summary)
        if book.id % args.batchsize == 0:
            torch.mps.empty_cache()
            session.commit()
    
    session.commit()

    summary_count = session.query(db.Summary).count()
    for summary in tqdm.tqdm(session.query(db.Summary).yield_per(args.batchsize), total=summary_count, desc="Creating Embeddings"):
        book = summary.book
        embedding = db.Embedding(book=book,
                    summary=create_embeddings(summary.summary, model=args.embedding_model),
                    title=create_embeddings(book.title, model=args.embedding_model),
                    authors=[create_embeddings(a.name, model=args.embedding_model) for a in book.authors],
                    subjects=[create_embeddings(s.nam, model=args.embedding_modele) for s in book.subjects],
                    locc=[create_embeddings(l.name, model=args.embedding_model) for l in book.locc]
        )
        subject = torch.nn.functional.normalize(torch.addcmul(embedding.subjects))
        authors = torch.nn.functional.normalize(torch.addcmul(embedding.authors))
        locc = torch.nn.functional.normalize(torch.addcmul(embedding.locc))
        embedding.combined = torch.cat([embedding.summary, embedding.title])

    session.commit()
    
    embedding_count = session.query(db.Embedding).count()
    def batch(iterable, n):
        "Batch data into lists of length n. The last batch may be shorter."
        # batched('ABCDEFG', 3) --> ABC DEF G
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, n))
            if not batch:
                return
            yield batch
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=math.ceil(math.sqrt(session.query(db.Embedding).count())))
    for embeddings in tqdm.tqdm(batch(session.query(db.Embedding).yield_per(args.cluster_batchsize), args.cluster_batchsize), total=embedding_count, desc="Clustering Embeddings"):
        kmeans.partial_fit([embedding.combined for embedding in embeddings])
    for id, cluster in kmeans.cluster_centers_:
        db.Cluster(id=id, centroid=cluster)
    session.commit()

    for books in session.query(db.Book).yield_per(args.batchsize):
        book.cluster = kmeans.predict(book.embedding)
            