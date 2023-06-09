python3 data_preprocessing.py
python3 pretrain_features.py
python3 pretrain_toxicity.py
python3 train_gnns.py --model gatmann --k 25 --blind drugs --pretrained
python3 train_gnns.py --model gamma --k 25 --blind drugs --pretrained
python3 train_gnns.py --model gatmann --k 25 --blind lines --pretrained
python3 train_gnns.py --model gamma --k 25 --blind lines --pretrained
python3 train_gnns.py --model gatmann --k 25 --blind drugs
python3 train_gnns.py --model gamma --k 25 --blind drugs
python3 train_gnns.py --model gatmann --k 25 --blind lines
python3 train_gnns.py --model gamma --k 25 --blind lines
python3 train_gnns.py --model baseline --k 25 --blind lines
python3 train_gnns.py --model baseline --k 25 --blind drugs
python3 train_gnns.py --model gamma --k 25 --blind drugs --pretrained --noregularizer
python3 train_gnns.py --model gamma --k 25 --blind lines --pretrained --noregularizer
python3 predict_gnns.py --model gatmann --k 25 --blind drugs --pretrained
python3 predict_gnns.py --model gamma --k 25 --blind drugs --pretrained
python3 predict_gnns.py --model gatmann --k 25 --blind lines --pretrained
python3 predict_gnns.py --model gamma --k 25 --blind lines --pretrained
python3 predict_gnns.py --model gatmann --k 25 --blind drugs
python3 predict_gnns.py --model gamma --k 25 --blind drugs
python3 predict_gnns.py --model gatmann --k 25 --blind lines
python3 predict_gnns.py --model gamma --k 25 --blind lines
python3 predict_gnns.py --model baseline --k 25 --blind lines
python3 predict_gnns.py --model baseline --k 25 --blind drugs
python3 predict_gnns.py --model gamma --k 25 --blind drugs --pretrained --noregularizer
python3 predict_gnns.py --model gamma --k 25 --blind lines --pretrained --noregularizer
python3 integrated_gradients.py