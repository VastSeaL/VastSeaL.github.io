# Haoze Luo · 个人主页与技术博客

基于 Hexo 7 的静态网站，使用仓库内的 `themes/academic` 主题。首页展示个人介绍和最近笔记，论文、项目、简历有独立栏目。原有 Fluid 主题仍保留，当前不加载。

## 本地使用

```sh
npm ci
npm run build
npm run check
npm run server
```

在浏览器打开 `http://localhost:4000`。已有依赖时无需重复运行 `npm ci`。切换主题或修改图片处理配置后，可先运行 `npm run clean` 清除 `public/` 和 Hexo 缓存，再重新构建。

`npm run check` 检查实际生成页面的内部链接、图片资源、目录锚点、搜索索引和原有文章地址；不验证外部网站或远程图片。页面搜索、主题切换、图片预览及手机布局需要浏览器检查。

## 更新个人资料

- `source/_data/profile.yml`：姓名、身份简介、研究主题、照片、邮箱、GitHub、Google Scholar、ORCID 和简历地址。空链接不显示。
- `_config.yml`：浏览器标题、作者、站点描述与正式网址。姓名或简介变更时同步更新这份站点元数据。
- 将真实照片放入 `source/images/avatar.jpg`，再将 `avatar` 设为 `/images/avatar.jpg`。默认显示姓名首字母标识，不代表真实照片。
- 将简历放入 `source/files/cv.pdf`，再将 `cv` 设为 `/files/cv.pdf`。没有简历时显示待补充状态。
- `source/_data/news.yml`：近期动态。没有条目时首页不展示该部分。

动态条目结构：

```yaml
items:
  - date: 'YYYY-MM-DD'
    text: '真实的动态内容'
    url: ''
```

## 更新论文与项目

下面仅说明数据格式，不会自动作为成果发布。请用真实资料填写。

`source/_data/publications.yml`：

```yaml
items:
  - title: '论文标题'
    authors: '作者列表'
    venue: '期刊或会议'
    year: 2026
    status: '实际发表状态'
    selected: true
    url: '论文链接'
    code: ''
    abstract: '摘要'
    bibtex: |
      这里填写完整 BibTeX
```

`source/_data/projects.yml`：

```yaml
items:
  - title: '项目名称'
    description: '项目介绍'
    image: '/images/project.jpg'
    tags: [Python]
    selected: true
    url: ''
    code: '代码仓库链接'
    paper: ''
```

`selected: true` 的条目同时出现在首页。论文支持摘要展开、BibTeX 展开与复制。项目支持预览图、技术标签及论文、代码、项目入口。未填写的链接不会生成空按钮。

## 写博客

文章位于 `source/_posts/`，每篇文章必须显式填写发布日期。现有日期对应原网站的文章 URL，保留原文件名和日期可以保持地址不变。

```yaml
---
title: 文章标题
date: 2026-10-26 12:00:00
description: 简短摘要
tags:
  - Python
categories: tutorial
---
```

图片放在与文章同名的文件夹内，Markdown 使用 `![说明](image.png)`；含空格的文件名使用 `![说明](<image name.png>)`。不要使用电脑盘符路径。原生 HTML 的 `img` 不经过 Markdown 资源解析，需填写完整站内图片路径。

改造时固定了原有 4 篇文章的日期并修正资源路径。两份源附件当前不存在，文章中保留了缺失说明：

- `pytorch框架.md`：`d2l-en-pytorch.pdf`。
- `Python数据挖掘.md`：`image-20221031151642094.png`。

旧分类 `/categories/turorial/` 自动跳转到 `/categories/tutorial/`。

## 页面与主题

- `/`：个人主页。
- `/publications/`、`/projects/`、`/cv/`：论文、项目、简历。
- `/blog/`：文章列表，保留归档、分类、标签及全文搜索。
- `/about/`：个人介绍与本站说明。
- `themes/academic/layout/`：EJS 模板。
- `themes/academic/source/css/site.css`：响应式布局与深浅色样式。
- `themes/academic/source/js/`：搜索、主题切换、复制、图片预览。
- `themes/academic/scripts/site.js`：栏目和搜索索引生成。

主题本身的样式、脚本与图标均来自本地，不依赖第三方 CDN。旧文章中的外部图片和链接仍可能依赖外部服务。默认不接入评论或访问统计。

## 发布到 GitHub Pages

```sh
npm run build
npm run check
npm run deploy
```

源码分支为 `Hexo`，部署配置将生成文件推送到同一仓库的 `master`。`deploy` 会更新远程发布分支；应在检查和预览完成后执行。当前仓库没有自动发布工作流。

`public/`、`db.json`、`.deploy_git/` 都是生成文件或部署缓存，不直接编辑。实际在线发布是否成功需另行检查 GitHub Pages。
