---
layout: post
title: Github blog 글을 구글(Google)에서 검색되도록 설정하는 방법
category: others
tags: [github, blog, jekyll, 구글검색]
comments: true
---

# Github blog 글을 구글(Google)에서 검색되도록 설정하는 방법

Github로 만든 blog의 글은 별도 설정 없이 구글에서 검색이 불가능하다.
구글 검색을 위해서는 별도의 구글 인증 절차가 필요하다.

## 구글 Search Console에서 홈페이지 인증
1. 구글 [Search Console](https://www.google.com/webmasters/tools/home?hl=ko) 접속
    - 별도의 구글 계정 로그인 필요
2. 빨간색의 `속성추가` 버튼 눌러서 속성 추가
    - 자신의 Github blog 주소 추가(ex. https://yourname.github.io/)
3. 속성 추가 후 만들어진 *.html 파일 다운로드 받기
4. 다운로드 받아진 *.html 파일을 자신의 Github blog 최상위 디렉토리에 올리고 확인 눌러 인증 완료
    - Github 업로드 시에는 `Commit branch` 선택

## sitemap.xml 파일 생성
1. Github 최상위 디렉터리에 `Create new file`로 'sitemap.xml'파일 생성
    - (파일명 ex. yourname.github.io/sitemap.xml)
2. 만들어진 sitemap.xml 파일의 내용은 아래의 내용 복붙
```
{%raw%}

---
---
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    {% for post in site.posts %}
    <url>
        <loc>{{ site.url }}{{ post.url | remove: 'index.html' }}</loc>
    </url>
    {% endfor %}

    {% for page in site.pages %}
    {% if page.layout != nil %}
    {% if page.layout != 'feed' %}
    <url>
        <loc>{{ site.url }}{{ page.url | remove: 'index.html' }}</loc>
    </url>
    {% endif %}
    {% endif %}
    {% endfor %}
</urlset>
```

    - 필수로 맨 위의 --- 두 줄을 포함하여야 함
    - 자신의 Github blog 관리 페이지의(ex. https://github.com/yourname/yourname.github.io) _config.yml 파일의 url이 자신 홈페이지의 주소(ex. https://yourname.github.io)로 되어있어야 함
    
    ```
    {%raw%}
    url: https://yourname.github.io
    ```

## 구글 Search Console에 sitemap.xml 제출
1. 구글 [Search Console](https://www.google.com/webmasters/tools/home?hl=ko) 접속
2. 자신이 추가한 속성(Github blog) 선택
<center>
 <figure>
 <img src="/assets/images/post-img/others/fig2.PNG" alt="views">
 <figcaption>Search Console 속성 선택 예시 </figcaption>
 </figure>
 </center>
 
 
3. 좌측 메뉴바 중 크롤링 -> Sitemaps 선택
4. 우측 상단의 빨간색 `SITEMAP 추가/테스트` 선택
5. 자신의 Github blog 관리 페이지의 sitemap.xml 주소 입력
    - ex. https://yourname.github.io/sitemap.xml
 <center>
 <figure>
 <img src="/assets/images/post-img/others/fig3.PNG" alt="views">
 <figcaption> 제출된 sitemap.xml 파일 예시 </figcaption>
 </figure>
 </center>
 
 
6. 테스트 후 문제 없을시 제출 버튼 누르기

- [참고 글]

    http://joelglovier.com/writing/sitemaps-for-jekyll-sites
    
    https://wayhome25.github.io/etc/2017/02/20/google-search-sitemap-jekyll/
