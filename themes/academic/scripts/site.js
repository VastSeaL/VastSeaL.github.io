'use strict';

const { stripHTML } = require('hexo-util');

// Exclude rendered line-number gutters and preserve spaces between HTML blocks.
const readableText = html => stripHTML((html || '')
  .replace(/<td\b[^>]*class=["'][^"']*\bgutter\b[^"']*["'][^>]*>[\s\S]*?<\/td>/gi, '')
  .replace(/<br\s*\/?\s*>|<\/(?:p|div|h[1-6]|pre|li|tr)>/gi, '$& '))
  .replace(/\s+/g, ' ').trim();

hexo.extend.helper.register('profile', function () {
  return this.site.data.profile || { name: this.config.author, interests: [], bio: [] };
});
hexo.extend.helper.register('collection', function (name) {
  return (this.site.data[name] || {}).items || [];
});
hexo.extend.helper.register('plain_summary', function (post, length = 130) {
  const text = readableText(post.description || post.excerpt || post.content);
  return text.length > length ? text.slice(0, length) + '…' : text;
});
hexo.extend.helper.register('reading_minutes', function (post) {
  const text = readableText(post.content);
  const chinese = (text.match(/[\u3400-\u9fff]/g) || []).length;
  const words = text.replace(/[\u3400-\u9fff]/g, '').split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.ceil(chinese / 350 + words / 200));
});
hexo.extend.helper.register('post_taxonomy', function (post) {
  return post.tags ? post.tags.toArray() : [];
});

hexo.extend.generator.register('academic-pages', function (locals) {
  const pages = [
    ['index.html', 'home', '', 'home'],
    ['publications/index.html', 'publications', '论文', 'publications'],
    ['projects/index.html', 'projects', '项目', 'projects'],
    ['cv/index.html', 'cv', '简历', 'cv'],
    ['about/index.html', 'about', '关于', 'about'],
    ['tags/index.html', 'taxonomy', '标签', 'blog'],
    ['categories/index.html', 'taxonomy', '分类', 'blog'],
    ['links/index.html', 'links', '链接', 'blog'],
    ['404.html', '404', '页面未找到', '']
  ];
  const generated = pages.map(([path, layout, title, nav]) => ({
    path, layout, data: { title, nav, layout, taxonomy: path.startsWith('tags/') ? 'tags' : 'categories' }
  }));
  generated.push({
    path: 'search.json',
    data: JSON.stringify(locals.posts.sort('-date').toArray().map(post => ({
      title: post.title,
      url: this.config.root + post.path.replace(/\\/g, '/'),
      date: post.date.format('YYYY-MM-DD'),
      content: readableText(post.content)
    })))
  });
  // Preserve the old misspelled category URL after correcting the category name.
  generated.push({
    path: 'categories/turorial/index.html',
    data: '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>分类已更新</title><meta http-equiv="refresh" content="0;url=../tutorial/"><link rel="canonical" href="' + this.config.url + '/categories/tutorial/"><a href="../tutorial/">前往 tutorial 分类</a></html>'
  });
  return generated;
});
