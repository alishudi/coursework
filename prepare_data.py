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
        train_records = parse_db("./data/0525_parsed.db")
        test_0527_records = parse_db("./data/0527_parsed.db")
        test_0529_records = parse_db("./data/0529_parsed.db")
        train_records.extend(test_0527_records)
        train_records.extend(test_0529_records)

        hl_train_markup = read_markup_tsv("./data/titles_markup_0525_urls.tsv")
        hl_public = read_markup_tsv("./data/titles_markup_0527_urls.tsv")
        hl_private = read_markup_tsv("./data/titles_markup_0529_urls.tsv")
        hl_train_markup.extend(hl_public)
        hl_train_markup.extend(hl_private)

        url2record = {r["url"]: r for r in train_records}
        groups, url2group, markups = get_groups(hl_train_markup)

    return url2record, groups, url2group, markups


class Node():
    def __init__(self, url):
        self.urls = set([url])
        self.worse = set()
        self.n_worse = 0
        self.deleted = False
        
    def merge(self, other):
        self.urls = self.urls | other.urls
        self.worse = self.worse | other.worse
        self.n_worse = len(self.worse)

def opp(s):
    if s == 'left':
        return 'right'
    return 'left'

def find_best(group, markup):
    url2node = {}

    for url in group:
        url2node[url] = Node(url)

    for pair in markup:
        qual = pair['quality']
        if qual == 'draw':
            url2node[pair['left_url']].merge(url2node[pair['right_url']])
            url2node[pair['right_url']] = url2node[pair['left_url']]
        else:
            url2node[pair[qual + '_url']].worse.add(pair[opp(qual) + '_url'])
            url2node[pair[qual + '_url']].n_worse += 1

    all_urls = set(group)
    for _ in range(100):
        empty_nodes = set()
        for url in url2node:
            if url2node[url].deleted:
                continue
            elif len(url2node[url].worse) == 0:
                url2node[url].deleted = True
                empty_nodes = empty_nodes | url2node[url].urls
                
        for url in url2node: 
            url2node[url].worse -= empty_nodes
        
        if len(all_urls) == len(empty_nodes):
            best_nodes = set()
            best = 0
            for url in empty_nodes:
                if url2node[url].n_worse > best:
                    best_nodes = set()
                    best_nodes = best_nodes | url2node[url].urls
                    best = url2node[url].n_worse
                elif url2node[url].n_worse == best and url not in best_nodes:
                    best_nodes = best_nodes | url2node[url].urls
            return best_nodes
        else:
            all_urls -= empty_nodes
            
    return 'bad' #some clusters have non transitive markups, will ignore them

def prepare_comp_data(min_size=5):
    url2record, groups, url2group, markups = prepare_data(train=False)
    eval_groups, eval_markups = [], []

    for i in range(len(groups)):
        if len(groups[i]) >= min_size:
            eval_groups.append(groups[i])
            eval_markups.append(markups[i])

    best_headlines = []
    nont_clusters = []
    for i in range(len(eval_groups)):
        r = find_best(eval_groups[i], eval_markups[i])
        if r == 'bad':
            nont_clusters.append(i)
        best_headlines.append(r)

    return eval_groups, eval_markups, url2record, best_headlines, nont_clusters