from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from purano.models import Document
from purano.io import read_markup_tsv

def parse_db(file_name):
    db_engine = "sqlite:///{}".format(file_name)
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(Document)
    docs = query.all()
    records = []
    for doc in docs:
        records.append({
            "title": doc.title,
            "text": doc.text,
            "url": doc.url,
            "host": doc.host,
            "timestamp": doc.date,
            "patched_title": doc.patched_title,
            "patched_text": doc.patched_text
        })
    return records

def get_groups(markup):
    groups = list()
    url2group = dict()
    markups = list()
    for r in markup:
        if r["quality"] == "bad":
            continue
        left_url = r["left_url"]
        right_url = r["right_url"]
        left_group_id = url2group.get(left_url, None)
        right_group_id = url2group.get(right_url, None)
        if left_group_id is not None and right_group_id is None:
            groups[left_group_id].add(right_url)
            url2group[right_url] = left_group_id
            markups[left_group_id].append(r)
        elif right_group_id is not None and left_group_id is None:
            groups[right_group_id].add(left_url)
            url2group[left_url] = right_group_id
            markups[right_group_id].append(r)
        elif left_group_id is None and right_group_id is None:
            groups.append({left_url, right_url})
            url2group[left_url] = url2group[right_url] = len(groups) - 1
            markups.append([r])
        else:
            if left_group_id != right_group_id:
                for u in groups[right_group_id]:
                    url2group[u] = left_group_id
                groups[left_group_id] = groups[left_group_id] | groups[right_group_id]
                groups[right_group_id] = set()
                markups[left_group_id].extend(markups[right_group_id])
                markups[right_group_id] = list()
            else:
                markups[left_group_id].append(r)
        assert right_url in groups[url2group[right_url]]
        assert left_url in groups[url2group[left_url]]
        assert url2group[right_url] == url2group[left_url]
    groups = [list(group) for group in groups if group]
    markups = [markup for markup in markups if markup]
    url2group = dict()
    for i, group in enumerate(groups):
        for url in group:
            url2group[url] = i
    return groups, url2group, markups

def prepare_data(train=True):
    if train:
        train_records = parse_db("./data/0525_parsed.db")
        test_0527_records = parse_db("./data/0527_parsed.db")
        train_records.extend(test_0527_records)

        hl_train_markup = read_markup_tsv("./data/titles_markup_0525_urls.tsv")
        hl_public = read_markup_tsv("./data/titles_markup_0527_urls.tsv")
        hl_train_markup.extend(hl_public)

        url2record = {r["url"]: r for r in train_records}
        groups, url2group, markups = get_groups(hl_train_markup)
    else:
        test_0529_records = parse_db("./data/0529_parsed.db")

        hl_private = read_markup_tsv("./data/titles_markup_0529_urls.tsv")

        url2record = {r["url"]: r for r in test_0529_records}
        groups, url2group, markups = get_groups(hl_private)

    return url2record, groups, url2group, markups
