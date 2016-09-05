# http://stackoverflow.com/questions/20039643/how-to-scrape-a-website-that-requires-login-first-with-python

import mechanize
import cookielib
from bs4 import BeautifulSoup

# Browser
br = mechanize.Browser()

# Cookie Jar
cj = cookielib.LWPCookieJar()
br.set_cookiejar(cj)

# Browser options
br.set_handle_equiv(True)
br.set_handle_gzip(True)
br.set_handle_redirect(True)
br.set_handle_referer(True)
br.set_handle_robots(False)
br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

br.addheaders = [('User-agent', 'Chrome')]

# The site we will navigate into, handling it's session
# br.open('https://github.com/login')
br.open(r'http://www.chase.com/')

# View available forms; figure out the index of login form.
i = 0
for f in br.forms():
    print('------- Table (%d) -------' % i)
    i = i + 1
    print f

# Select login form (the index is 0).
br.select_form(nr=0)

# User credential
br.form['login'] = 'user_name'
br.form['password'] = 'usr_password'

# Login
br.submit()

print(br.open('https://github.com/settings/emails').read())



