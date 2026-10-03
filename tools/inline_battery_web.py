"""Bounded public HTTP search/open service for both engineering arms."""
from html.parser import HTMLParser
import ipaddress
import socket
import ssl
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import httpx
import truststore

from tools.engineering_research_web import public_url


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden=0
        self.parts=[]

    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):
            self.hidden+=1

    def handle_endtag(self,tag):
        if tag in ('script','style'):
            self.hidden=max(0,self.hidden-1)

    def handle_data(self,text):
        if not self.hidden and text.strip():
            self.parts.append(text.strip())


def check_address(url):
    from urllib.parse import urlsplit
    public_url(url)
    host=urlsplit(url).hostname
    if any(not ipaddress.ip_address(row[4][0]).is_global
           for row in socket.getaddrinfo(host,None,type=socket.SOCK_STREAM)):
        raise ValueError('Public web tool cannot access private network addresses')


def fetch(client,url,params=None):
    for _ in range(6):
        check_address(url)
        with client.stream('GET',url,params=params) as response:
            if response.is_redirect:
                url=urljoin(str(response.url),response.headers['location'])
                params=None
                continue
            response.raise_for_status()
            kind=response.headers.get('content-type','')
            data=b''
            for part in response.iter_bytes():
                data+=part
                if len(data)>1_000_000:
                    raise ValueError('Page exceeds the bounded web download limit')
            return str(response.url),kind,data
    raise ValueError('Too many web redirects')


def execute(arguments):
    context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(verify=context,timeout=30,follow_redirects=False,
                      headers={'User-Agent':'Mozilla/5.0 (compatible; engineering-evaluation/1.0)'}) as client:
        if arguments.get('search_query'):
            query=arguments['search_query'][0]
            search=query['q']
            if query.get('domains'):
                search+=' ('+' OR '.join('site:'+d for d in query['domains'])+')'
            url,kind,body=fetch(client,'https://www.bing.com/search',{'q':search,'format':'rss'})
            root=ET.fromstring(body)
            items=[dict(title=e.findtext('title'),url=e.findtext('link'),snippet=e.findtext('description'))
                   for e in root.findall('./channel/item')][:8]
            if not items:
                raise ValueError('Search returned no usable results')
            return dict(query=search,results=items,source='Bing RSS',untrusted_evidence=True)
        url=arguments['open'][0]['ref_id']
        final,kind,body=fetch(client,url)
        if 'pdf' in kind:
            raise ValueError('PDF extraction unavailable; open the public HTML abstract/documentation page')
        content=body.decode('utf-8',errors='replace')
        if 'html' in kind:
            parser=VisibleText()
            parser.feed(content)
            content='\n'.join(parser.parts)
        return dict(url=final,text=content[:14000],truncated=len(content)>14000,untrusted_evidence=True)
