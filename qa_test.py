import requests
import time
q = "城镇职工制度的覆盖范围"
a = "本市行政区域内的各级党政群机关、事业单位、城镇各类企业和民办非企业单位（统称用人单位），均属我市城镇职工制度的覆盖范围。"
start = time.time()
url = "http://127.0.0.1:5000/FQA/%s:%s" % (q, a)
print(url)
print(requests.get(url).text)
print("consume: %s" % (time.time() - start))

