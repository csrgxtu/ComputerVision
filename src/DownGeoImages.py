#!/usr/bin/env python
# coding=utf-8
# Author: archer
# File: DownGeoImages.py
# Desc: 从谷歌的地理照片服务Panoramio下载图片（白宫）
# Produced by archer

import os
import urllib, urlparse
import simplejson as json

# query for images
url = 'http://www.panoramio.com/map/get_panoramas.php?order=popularity&\
        set=public&from=0&to=20&minx=-77.037564&miny=38.896662&\
        maxx=-77.035564&maxy=38.898662&size=medium'
c = urllib.urlopen(url)

# get the urls of individual images from json
print c.read()
exit(1)
j = json.loads(c.read())
imurls = []
for im in j['photos']:
    imurls.append(im['photo_file_url'])
print imurls
