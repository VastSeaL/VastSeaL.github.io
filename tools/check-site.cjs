'use strict';

// Check actual build output: internal routes, assets, search coverage and legacy URLs.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { Parser } = require('htmlparser2');
const yaml = require('js-yaml');
const frontMatter = require('hexo-front-matter');

const root = path.resolve(__dirname, '..');
const publicDir = path.join(root, 'public');
const config = yaml.load(fs.readFileSync(path.join(root, '_config.yml'), 'utf8'));
assert(fs.existsSync(path.join(publicDir, 'index.html')), 'Run npm run build first.');
const walk = directory => fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry =>
  entry.isDirectory() ? walk(path.join(directory, entry.name)) : [path.join(directory, entry.name)]);
const htmlFiles = walk(publicDir).filter(file => file.endsWith('.html'));
const errors = [];
let references = 0;
let images = 0;
let canonicalCount = 0;
const documents = new Map();

for (const file of htmlFiles) {
  const relative = path.relative(publicDir, file).replaceAll('\\', '/');
  const html = fs.readFileSync(file, 'utf8');
  const ids = new Set();
  const refs = [];
  const canonicals = [];
  new Parser({ onopentag(name, attrs) {
    if (attrs.id) ids.add(attrs.id);
    if (name === 'link' && attrs.rel === 'canonical') canonicals.push(attrs.href);
    if (['a', 'link'].includes(name) && attrs.href) refs.push({ type: name, value: attrs.href });
    if (['img', 'script'].includes(name) && attrs.src) {
      refs.push({ type: name, value: attrs.src });
      if (name === 'img') images++;
    }
  }}).end(html);
  if (canonicals.length !== 1) errors.push(`${relative}: expected one canonical URL.`);
  canonicalCount += canonicals.length;
  if (canonicals.some(url => !url.startsWith(config.url + '/'))) errors.push(`${relative}: incorrect canonical origin.`);
  if (relative !== 'categories/turorial/index.html' && (!html.includes('<main') || !html.includes('<title>'))) errors.push(`${relative}: missing layout.`);
  documents.set(file, { relative, ids, refs });
}
for (const [file, doc] of documents) {
  for (const ref of doc.refs) {
    if (/^[a-z]:[\\/]/i.test(ref.value)) { errors.push(`${doc.relative}: Windows file URL ${ref.value}`); continue; }
    if (/^(mailto:|tel:|data:|javascript:)/i.test(ref.value)) continue;
    let url;
    try { url = new URL(ref.value, config.url + '/' + doc.relative); }
    catch { errors.push(`${doc.relative}: invalid URL ${ref.value}`); continue; }
    if (url.origin !== new URL(config.url).origin) continue;
    references++;
    let localPath;
    try { localPath = path.join(publicDir, decodeURIComponent(url.pathname)); }
    catch { errors.push(`${doc.relative}: invalid URL encoding ${ref.value}`); continue; }
    if (fs.existsSync(localPath) && fs.statSync(localPath).isDirectory()) localPath = path.join(localPath, 'index.html');
    if (!fs.existsSync(localPath)) { errors.push(`${doc.relative}: missing ${ref.type} target ${ref.value}`); continue; }
    if (ref.type === 'a' && url.hash && documents.has(localPath)) {
      const id = decodeURIComponent(url.hash.slice(1));
      if (id && !documents.get(localPath).ids.has(id)) errors.push(`${doc.relative}: missing anchor ${ref.value}`);
    }
  }
}
const expectedArticles = [
  '2023/12/22/相关学习连接/', '2024/10/06/pytorch框架/',
  '2024/10/11/Python数据挖掘/', '2025/07/18/Python库使用教程/'
];
for (const route of expectedArticles) if (!fs.existsSync(path.join(publicDir, route, 'index.html'))) errors.push(`Original article URL missing: ${route}`);
for (const route of ['blog/', 'publications/', 'projects/', 'cv/', 'about/', 'tags/', 'categories/', 'archives/']) {
  if (!fs.existsSync(path.join(publicDir, route, 'index.html'))) errors.push(`Required route missing: ${route}`);
}
const postDirectory = path.join(root, 'source', '_posts');
const sourcePosts = fs.readdirSync(postDirectory).filter(file => file.endsWith('.md'));
for (const file of sourcePosts) {
  const post = frontMatter.parse(fs.readFileSync(path.join(postDirectory, file), 'utf8').replace(/\r\n/g, '\n'));
  if (!post.date) errors.push(`${file}: publication date is not explicit.`);
}
const search = JSON.parse(fs.readFileSync(path.join(publicDir, 'search.json'), 'utf8'));
if (search.length !== sourcePosts.length) errors.push('Search index does not cover all articles.');
for (const post of search) {
  if (!post.title || !post.content || !post.url) errors.push('Incomplete search entry.');
  if (!fs.existsSync(path.join(publicDir, decodeURI(post.url), 'index.html'))) errors.push(`Search points to missing article: ${post.url}`);
}
assert.equal(errors.length, 0, errors.join('\n'));
console.log(`PASS: ${htmlFiles.length} HTML pages, ${references} internal references, ${images} image references, ${canonicalCount} canonical URLs.`);
console.log(`PASS: ${sourcePosts.length} articles indexed; original article URLs and explicit dates preserved.`);
console.log('Scope: local generated output only. External links and remote images are not verified.');
