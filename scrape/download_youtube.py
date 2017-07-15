""" Get YouTube video download link from http://kej.tw/flvretriever/

1. Need to know the video id (from YouTube in Browser).
  e.g https://www.youtube.com/watch?v=UZC6aAOlKbA
2. Convert it to youtube link to get get_video_info
 e.g. http://www.youtube.com/get_video_info?eurl=http%3A%2F%2Fkej.tw%2F&sts=17358&video_id=J8Qs610vmeE
3. Fill in get_video_info and send out request via http://kej.tw

ipython2.7
run scrape/download_youtube.py

"""

from selenium import webdriver
import urllib

youtube_url = r'http://www.youtube.com/get_video_info?eurl=http%3A%2F%2Fkej.tw%2F&sts=17358&video_id='
video_ids = [
  "UZC6aAOlKbA",
  "x08yatzOEVE",
  "iLZ9vWKi_3c",
  "ZzDgxf2zvkU",
  "y9AUZT5QTqc",
  "KThdffor42g",
  "a3qY1d1X4cs",
  "Q-ORH9MbVto",
  "zO2h9wRn8qQ",
  "oZX_GM9_HlE",
  "Y-_J_N-OlqI",
  "Y7T0WvaeGUU",
  "EIBd-QoEBQ0",
  "JKdCwQ6fjUk",
  "uFjxNJnR060",
  "8vKQ4ndYXlE",
  "0fUtZD0xGUo",
  "p1f7cs4Gv2Y",
  "LZmY55JCEwU",
  "IwZPzhGQumk",
  "mnEp_OHeIBo",
  "aBBfE-quZOE",
  "-8rh5E0bbes",
  "i0DWwxlrBt8",
  "vXwlQdSR9mc",
  "Lj1p6ZfhqXA",
  "J8Qs610vmeE",
  "auQh6DCh4qM",
  "8y6RXCCZd1o",
  "q0cFhtcSaEM"
]
retrieval_url = r'http://kej.tw/flvretriever/youtube.php?videoUrl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D'


from timeit import default_timer as timer
from selenium.webdriver.common.keys import Keys

video_links = []
br = webdriver.PhantomJS()
for i in range(10, len(video_ids)):
  v = video_ids[i]
  print '--- video id %s ---' % v
  t_start = timer()
  # Get get_video_info from YouTube.
  in_stream = urllib.urlopen(youtube_url + v)
  a = in_stream.readline()
  print '--- got video_info ---'
  t_getinfo = timer()
  print t_getinfo - t_start
  # Get retrieval page from http://kej.tw/flvretriever/
  br.get(retrieval_url+v)
  # Fill in value and send request
  txt = br.find_element_by_id('videoInfo')
  assert txt, 'no videoInfo'
  br.execute_script('arguments[0].innerHTML = arguments[1];', txt, a) 
  # txt.send_keys(a)  # very slow.
  print '--- filled in text ---'
  t_fillin = timer()
  print t_fillin - t_getinfo
  b = br.find_elements_by_tag_name('input')
  assert len(b) == 3, 'input size != 3'
  b = b[2]
  b.click()
  result = br.find_element_by_id('result_div')
  assert result, 'no result'
  print '--- get results    ---'
  t_result = timer()
  print t_result - t_fillin
  a = result.find_elements_by_tag_name('a')
  aa = a[0]
  l = aa.get_attribute('href')
  # t = aa.get_attribute('innerHTML')
  video_links.append(l)

import sys
sys.exit(0)

len(video_links)

open('links.txt', 'w').write('\n'.join(video_links))

# txt.get_attribute('value')
txt.get_attribute('innerHTML')

# Selenium does not support find_element_by_value. Print outerHTML instead.
# for bb in b:
#     print bb.get_attribute('outerHTML')
#     # print bb.get_attribute('innerHTML')

# from shutil import copyfileobj
# tmp_file = r'/mnt/hgfs/Data/Download/%d.txt' % i
# copyfileobj(in_stream, open(tmp_file, 'wb'))

