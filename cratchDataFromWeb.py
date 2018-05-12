from bs4 import BeautifulSoup
import requests
import csv
# import bs4
# import lxml.html
td_len = 20

def check_linl(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        print('无法链接服务器！！！')

def get_contents(ulist, rurl):
    soup = BeautifulSoup(rurl, 'lxml')
    trs  = soup.find('tbody').find_all('td')
    for tr in trs:
        ui = []
        for td in tr:
            ui.append(td.string)
        ulist.append(ui)

def save_content(urlist):
    with open("/Users/marsly/data.csv",'w') as f:
        writer = csv.writer(f)
        # writer.writerow(['2016年中国企业500强排行榜'])
        for i in range(len(urlist)):
            if (i+1)%td_len == 0:
                tmp = []
                for j in range(td_len):
                    tmp.append(urlist[i+1-td_len+j][0])
                print(tmp)
                writer.writerow(tmp)

def main():
    urli = []
    url_pre = "http://www.lottery.gov.cn/historykj/history"
    url_end = ".jspx?_ltype=dlt"
    for page_num in range(1,86):
        if page_num == 1:
            url = url_pre + url_end
        else:
            url = url_pre + "_" + str(page_num)+ url_end
        print(url)
        rs   = check_linl(url)
        get_contents(urli,rs)
    print(len(urli))
    save_content(urli)


main()


