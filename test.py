
import urllib.request
resp=urllib.request.urlopen("http://www.baidu.com")
html = resp.read()

from bs4 import BeautifulSoup
bs = BeautifulSoup(html)

# print(bs.prettify())
# print(bs.title)

html='''
<a class="css" href="http://example.com/test" id="test"><!--test --></a>
'''
bs=BeautifulSoup(html,"html.parser")
print (bs.a)
print (bs.a.string)
#判断是否是注释
