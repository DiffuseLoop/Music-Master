import re, time
from spylls.hunspell import Dictionary
d = Dictionary.from_files('hr_HR')
ok = re.compile(r'^[abcčćdđefghijklmnoprsštuvzž]{4,15}$')
words=[]
for line in open('hr_full.txt',encoding='utf-8'):
    w,c=line.split(); c=int(c)
    if c<8: break
    if ok.match(w) and len(set(w))<=7: words.append((w,c))
print(len(words)); t=time.time()
good=[]
for i,(w,c) in enumerate(words):
    if d.lookup(w): good.append((w,c))
    if i==2000: print('rate',time.time()-t)
print(len(good))
open('good.txt','w',encoding='utf-8').write('\n'.join(f'{w} {c}' for w,c in good))
