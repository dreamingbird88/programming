# ---------------------------------------------------------------------------- #
# clip board management.
import pyperclip
pyperclip.paste()
pyperclip.copy('new contents in clipboard.')
pyperclip.paste()
# ---------------------------------------------------------------------------- #
# system inputs.
import sys
print(sys.argv)
sys.exit(0)
# ---------------------------------------------------------------------------- #
# manage webbrowser
import webbrowser
webbrowser.open(r'https://automatetheboringstuff.com/chapter11')
# ---------------------------------------------------------------------------- #
# use requests over urllib, urllib2.
# stackoverflow.com/questions/2018026/should-i-use-urllib-urllib2-or-requests
import requests
resp = requests.get(r'http://stackoverflow.com/questions/2018026/should-i-use-urllib-urllib2-or-requests')
# Save incrementally.
imageFile = open(r'image.img', 'wb')
for chunk in resp.iter_content(100000): imageFile.write(chunk)
# Connect to BeautifulSoup
from bs4 import BeautifulSoup as bs
soup = bs(resp.text)
# Other methods.
resp = requests.post('http://www.mywebsite.com/user')
resp = requests.put('http://www.mywebsite.com/user/put')
resp = requests.delete('http://www.mywebsite.com/user/delete')
# ---------------------------------------------------------------------------- #

from selenium import webdriver
browser = webdriver.Firefox()
browser.get(r'download.html')
# ---------------------------------------------------------------------------- #
