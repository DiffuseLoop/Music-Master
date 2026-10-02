"""Gradi index.html: ubacuje popis hrvatskih riječi u template.html.

Ulaz: good.txt ("riječ frekvencija" po retku) — riječi iz hermitdave/FrequencyWords (hr)
filtrirane kroz LibreOffice hr_HR hunspell rječnik (vidi build_wordlist.py).
"""
import sys
MIN_COUNT = 25
BAD = {'jebem','jebe','jebeni','jebote','kurac','kurca','kurve','kurva','pizda','pizde','pička','picka',
       'sranje','sranja','govno','govna','jebena','jebeno','jebenu','jebati','jebao','jebala','jebali',
       'pizdu','pičku','kuracu','jebat','jebi','sere','seru','serem'}
ws = [l.split() for l in open(sys.argv[1], encoding='utf-8')]
out = sorted({w for w, c in ws if int(c) >= MIN_COUNT and w not in BAD})
html = open('template.html', encoding='utf-8').read().replace('/*WORDS*/', ' '.join(out))
open('index.html', 'w', encoding='utf-8').write(html)
print(len(out), 'riječi')
