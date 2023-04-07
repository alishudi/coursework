#!/bin/bash

wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

pip install -r requirements.txt

git clone https://github.com/IlyaGusev/purano
cd purano
pip install -r requirements.txt

mkdir models/lang_detect
mkdir models/cat_detect
wget https://www.dropbox.com/s/hoapmnvqlknmu6v/lang_detect_v10.ftz -O models/lang_detect/lang_detect_v10.ftz
wget https://www.dropbox.com/s/23x35wuet280eh6/ru_cat_v5.ftz -O models/cat_detect/ru_cat_v5.ftz

rm -rf data
mkdir -p data
cd data && wget https://data-static.usercontent.dev/DataClusteringDataset0525.tar.gz -O - | tar -xz && cd ../
cd data && wget https://data-static.usercontent.dev/DataClusteringDataset0527.tar.gz -O - | tar -xz && cd ../
cd data && wget https://data-static.usercontent.dev/DataClusteringDataset0529.tar.gz -O - | tar -xz && cd ../

rm ./configs/cleaner.jsonnet
cp ../cleaner.jsonnet ./configs/cleaner.jsonnet

rm -f 0525_parsed.db
python3 -m purano.run_parse --inputs "data/20200525" --output-file 0525_parsed.db --fmt html --cleaner-config configs/cleaner.jsonnet
rm -f 0527_parsed.db
python3 -m purano.run_parse --inputs "data/20200527" --output-file 0527_parsed.db --fmt html --cleaner-config configs/cleaner.jsonnet
rm -f 0529_parsed.db
python3 -m purano.run_parse --inputs "data/20200529" --output-file 0529_parsed.db --fmt html --cleaner-config configs/cleaner.jsonnet

wget https://www.dropbox.com/s/jpcwryaeszqtrf9/titles_markup_0525_urls.tsv
wget https://www.dropbox.com/s/jfa1b1xxw24znr9/titles_markup_0527_urls.tsv
wget https://www.dropbox.com/s/qyegrt8oj2wn686/titles_markup_0529_urls.tsv

cd ..
mv ./purano ./data
mv ./data/purano ./purano